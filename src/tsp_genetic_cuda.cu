#include <iostream>
#include <algorithm>
#include <cstdio>
#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "../include/testcases.h"
#include <chrono>

using namespace std;

// Error checking macro
#define CUDA_CHECK(call)                                                                              \
    do                                                                                                \
    {                                                                                                 \
        cudaError_t err = call;                                                                       \
        if (err != cudaSuccess)                                                                       \
        {                                                                                             \
            fprintf(stderr, "CUDA Error %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                       \
        }                                                                                             \
    } while (0)

//------------------------------------------------------
// Kernel Parameters (Adjust as Needed)
#define BLOCK_SIZE 128

//------------------------------------------------------
// Device Kernels

// Setup Curand States
__global__ void setupCurandStates(curandState *states, unsigned long long seed, int populationSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize)
    {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Initialize Population:
// Each individual is a permutation of [0 ... numCities-1].
// A simple approach: start with identity and perform a series of swaps per individual.
__global__ void initPopulationKernel(int *d_population, curandState *states, int populationSize, int numCities)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize)
    {
        int start = idx * numCities;
        // Initialize as identity permutation
        for (int i = 0; i < numCities; i++)
        {
            d_population[start + i] = i;
        }
        // Shuffle using Fisher-Yates
        curandState localState = states[idx];
        for (int i = numCities - 1; i > 0; i--)
        {
            int j = (int)(curand_uniform(&localState) * (i + 1));
            int temp = d_population[start + i];
            d_population[start + i] = d_population[start + j];
            d_population[start + j] = temp;
        }
        states[idx] = localState;
    }
}

// Fitness Kernel: Compute fitness = 1/distance for each individual
__global__ void fitnessKernel(const int *d_distanceMatrix, const int *d_population, double *d_fitness,
                              int numCities, int populationSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize)
    {
        const int start = idx * numCities;
        double dist = 0.0;
        for (int i = 0; i < numCities - 1; i++)
        {
            int from = d_population[start + i];
            int to = d_population[start + i + 1];
            dist += d_distanceMatrix[from * numCities + to];
        }
        // close the loop
        int last = d_population[start + numCities - 1];
        int first = d_population[start];
        dist += d_distanceMatrix[last * numCities + first];
        d_fitness[idx] = 1.0 / dist;
    }
}

// Find Best Individual Kernel
// Single-block reduction for simplicity (assumes populationSize <= blockDim.x *gridDim.x adequately)
__global__ void findBestKernel(const int *d_population, const double *d_fitness, int *d_bestTour, double *d_bestFitness,
                               int *d_blockBestIndex, double *d_blockBestFitness,
                               int populationSize, int numCities)
{
    extern __shared__ double s_fitness[];
    __shared__ int s_indices[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double val = -1.0;
    int bestIdx = -1;
    if (idx < populationSize)
    {
        val = d_fitness[idx];
        bestIdx = idx;
    }

    s_fitness[tid] = val;
    s_indices[tid] = bestIdx;

    __syncthreads();

    // Reduce to find max fitness
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        // reduce all of the larger values to the first half
        if (tid < s && s_fitness[tid] < s_fitness[tid + s])
        {
            s_fitness[tid] = s_fitness[tid + s];
            s_indices[tid] = s_indices[tid + s];
        }
        __syncthreads();
    }

    // Block-level result is stored in shared memory index 0
    if (tid == 0)
    {
        d_blockBestFitness[blockIdx.x] = s_fitness[0];
        d_blockBestIndex[blockIdx.x] = s_indices[0];
    }

    __syncthreads();

    // Perform linear scan of `d_blockBestFitness` in a single thread
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        double maxFitness = -1.0;
        int bestGlobalIndex = -1;

        for (int i = 0; i < gridDim.x; i++)
        {
            if (d_blockBestFitness[i] > maxFitness)
            {
                maxFitness = d_blockBestFitness[i];
                bestGlobalIndex = d_blockBestIndex[i];
            }
        }

        // Update the global best fitness and best tour
        *d_bestFitness = maxFitness;
        for (int i = 0; i < numCities; i++)
        {
            d_bestTour[i] = d_population[bestGlobalIndex * numCities + i];
        }
    }
}

// Tournament Selection Kernel:
// We'll select pairs of parents from top 50% of population based on fitness.
//
// For simplicity, assume populationSize is even, and we produce offspring for the bottom 50%.
// Each thread handles one offspring index in the bottom half.
// We'll pick two parents by tournament from the top half.
__global__ void selectionKernel(const double *d_fitness, int *d_population, int populationSize,
                                int numCities, curandState *states)
{
    int half = populationSize / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // idx indexes into the bottom half
    if (idx < half)
    {
        int offspringIdx = half + idx; // where to place offspring

        curandState localState = states[idx];

        // tournament for parent1
        int p1 = (int)(curand_uniform(&localState) * half);
        int p2 = (int)(curand_uniform(&localState) * half);
        int parent1 = (d_fitness[p1] > d_fitness[p2]) ? p1 : p2;

        // tournament for parent2
        p1 = (int)(curand_uniform(&localState) * half);
        p2 = (int)(curand_uniform(&localState) * half);
        int parent2 = (d_fitness[p1] > d_fitness[p2]) ? p1 : p2;

        // Store chosen parents in place
        // We'll rely on crossover kernel to read from these indices
        // Let's store parent indices at the offspring location temporarily:
        // The crossover kernel will interpret these as parent indices.
        // This is a trick: we temporarily store negative indices to differentiate.
        // Another approach: we could use a separate array for parent indices.
        d_population[offspringIdx * numCities] = parent1;
        d_population[offspringIdx * numCities + 1] = parent2;

        states[idx] = localState;
    }
}

// Crossover Kernel (Order Crossover In-Place):
// We know that the bottom half of the population currently holds temporary parent indices at [offspringIdx*numCities ...],
// specifically at the first two positions. We will read parent tours from the top half and overwrite the offspring tours.
__global__ void crossoverKernel(int *d_population, int populationSize, int numCities, curandState *states)
{
    int half = populationSize / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < half)
    {
        int offspringIdx = half + idx;
        curandState localState = states[idx];

        // Retrieve parents
        int parent1 = d_population[offspringIdx * numCities];
        int parent2 = d_population[offspringIdx * numCities + 1];

        // Perform order crossover
        int start = (int)(curand_uniform(&localState) * numCities);
        int end = (int)(curand_uniform(&localState) * numCities);
        if (start > end)
        {
            int tmp = start;
            start = end;
            end = tmp;
        }

        // Copy parent1 segment
        int *child = &d_population[offspringIdx * numCities];
        int *p1 = &d_population[parent1 * numCities];
        int *p2 = &d_population[parent2 * numCities];

        // Mark child as empty
        for (int i = 0; i < numCities; i++)
        {
            child[i] = -1;
        }

        for (int i = start; i <= end; i++)
        {
            child[i] = p1[i];
        }

        int currentIndex = 0;
        for (int i = 0; i < numCities; i++)
        {
            int val = p2[i];
            bool found = false;
            // Check if val is already in child
            // TODO: For large numCities, consider optimization or a hash table approach
            for (int j = start; j <= end; j++)
            {
                if (child[j] == val)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                // find next available slot
                while (child[currentIndex] != -1)
                    currentIndex++;
                child[currentIndex] = val;
            }
        }

        states[idx] = localState;
    }
}

// Mutation Kernel (Swap mutation):
// Mutate bottom half of the population.
// Randomly swap two cities in each individual.
__global__ void mutationKernel(int *d_population, int populationSize, int numCities, double mutationRate, curandState *states)
{
    int half = populationSize / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < half)
    {
        int offspringIdx = half + idx;
        curandState localState = states[idx];

        if (curand_uniform(&localState) < mutationRate)
        {
            int i = (int)(curand_uniform(&localState) * numCities);
            int j = (int)(curand_uniform(&localState) * numCities);
            int *child = &d_population[offspringIdx * numCities];
            int temp = child[i];
            child[i] = child[j];
            child[j] = temp;
        }

        states[idx] = localState;
    }
}

//------------------------------------------------------
// Host Code

int main()
{
    // Genetic Algorithm parameters
    int generations = 1000;
    double mutationRate = 0.1;
    int populationSize = 512;

    // Iterate over test cases defined in testcases.h
    for (size_t test_i = 0; test_i < testcases.size(); ++test_i)
    {
        const auto &[distanceMatrix, expectedDistance] = testcases[test_i];

        // timing
        auto start = chrono::high_resolution_clock::now();
        int numCities = (int)distanceMatrix.size();

        // Flatten distance matrix
        std::vector<int> h_distanceMatrix(numCities * numCities);
        for (int row = 0; row < numCities; row++)
        {
            for (int col = 0; col < numCities; col++)
            {
                h_distanceMatrix[row * numCities + col] = distanceMatrix[row][col];
            }
        }

        // Device memory
        int *d_population, *d_distanceMatrix, *d_bestTour, *d_blockBestIndex;
        double *d_fitness, *d_bestFitness, *d_blockBestFitness;
        curandState *d_states;

        CUDA_CHECK(cudaMalloc(&d_population, sizeof(int) * populationSize * numCities));
        CUDA_CHECK(cudaMalloc(&d_distanceMatrix, sizeof(int) * numCities * numCities));
        CUDA_CHECK(cudaMalloc(&d_fitness, sizeof(double) * populationSize));
        CUDA_CHECK(cudaMalloc(&d_bestTour, sizeof(int) * numCities));
        CUDA_CHECK(cudaMalloc(&d_bestFitness, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_states, sizeof(curandState) * populationSize));
        CUDA_CHECK(cudaMemcpy(d_distanceMatrix, h_distanceMatrix.data(), sizeof(int) * numCities * numCities, cudaMemcpyHostToDevice));

        int gridDim = (populationSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CUDA_CHECK(cudaMalloc(&d_blockBestFitness, sizeof(double) * gridDim));
        CUDA_CHECK(cudaMalloc(&d_blockBestIndex, sizeof(int) * gridDim));

        // Initialize bestFitness to a very low value
        double initVal = -1e9;
        CUDA_CHECK(cudaMemcpy(d_bestFitness, &initVal, sizeof(double), cudaMemcpyHostToDevice));

        // Setup curand states
        {
            int grid = (populationSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
            setupCurandStates<<<grid, BLOCK_SIZE>>>(d_states, 1234ULL, populationSize);
            cudaDeviceSynchronize();
        }

        // Initialize population
        {
            int grid = (populationSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
            initPopulationKernel<<<grid, BLOCK_SIZE>>>(d_population, d_states, populationSize, numCities);
            cudaDeviceSynchronize();
        }

        // Main GA loop
        for (int gen = 0; gen < generations; gen++)
        {
            // Compute fitness
            {
                int grid = (populationSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
                fitnessKernel<<<grid, BLOCK_SIZE>>>(d_distanceMatrix, d_population, d_fitness, numCities, populationSize);
                cudaDeviceSynchronize();
            }

            // Find best
            {
                int grid = (populationSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
                findBestKernel<<<grid, BLOCK_SIZE, BLOCK_SIZE * (sizeof(double) + sizeof(int))>>>(d_population, d_fitness, d_bestTour, d_bestFitness, d_blockBestIndex, d_blockBestFitness, populationSize, numCities);
                cudaDeviceSynchronize();
            }

            // Selection
            // Select 2 parents from the top 50% of the population
            {
                int half = populationSize / 2;
                int grid = (half + BLOCK_SIZE - 1) / BLOCK_SIZE;
                selectionKernel<<<grid, BLOCK_SIZE>>>(d_fitness, d_population, populationSize, numCities, d_states);
                cudaDeviceSynchronize();
            }

            // Crossover
            // Crossover the selected parents pairs, replacing the bottom 50% of the population
            {
                int half = populationSize / 2;
                int grid = (half + BLOCK_SIZE - 1) / BLOCK_SIZE;
                crossoverKernel<<<grid, BLOCK_SIZE>>>(d_population, populationSize, numCities, d_states);
                cudaDeviceSynchronize();
            }

            // Mutation
            {
                int half = populationSize / 2;
                int grid = (half + BLOCK_SIZE - 1) / BLOCK_SIZE;
                mutationKernel<<<grid, BLOCK_SIZE>>>(d_population, populationSize, numCities, mutationRate, d_states);
                cudaDeviceSynchronize();
            }
        }

        // Copy back best result
        double hostBestFitness;
        std::vector<int> hostBestTour(numCities);
        CUDA_CHECK(cudaMemcpy(&hostBestFitness, d_bestFitness, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostBestTour.data(), d_bestTour, sizeof(int) * numCities, cudaMemcpyDeviceToHost));

        double bestDistance = 1.0 / hostBestFitness;
        // timing
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed_time = end - start;
        std::cout << "Time Taken: " << elapsed_time.count() << " seconds\n";
        std::cout << "Test Case " << test_i << ":\n";
        std::cout << "Best Distance Found: " << bestDistance << "\n";
        std::cout << "Expected Distance: " << expectedDistance << "\n";
        std::cout << "Best Tour: ";
        for (auto c : hostBestTour)
            std::cout << c << " ";
        std::cout << "\n-----------------------------------------\n";

        // Clean up
        CUDA_CHECK(cudaFree(d_population));
        CUDA_CHECK(cudaFree(d_distanceMatrix));
        CUDA_CHECK(cudaFree(d_fitness));
        CUDA_CHECK(cudaFree(d_bestTour));
        CUDA_CHECK(cudaFree(d_bestFitness));
        CUDA_CHECK(cudaFree(d_states));
        CUDA_CHECK(cudaFree(d_blockBestFitness));
        CUDA_CHECK(cudaFree(d_blockBestIndex));
    }

    return 0;
}