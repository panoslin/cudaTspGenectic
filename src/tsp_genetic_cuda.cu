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
// Kernel Parameters
#define BLOCK_SIZE 1024

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
__global__ void initPopulationKernel(
    int *d_population,
    curandState *states,
    int populationSize,
    int numCities)
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
            if (j > i)
                j = i;
            int temp = d_population[start + i];
            d_population[start + i] = d_population[start + j];
            d_population[start + j] = temp;
        }
        states[idx] = localState;
    }
}

// Fitness Kernel: Compute fitness = 1/distance for each individual
__global__ void fitnessKernel(
    const int *d_distanceMatrix,
    const int *d_population,
    double *d_fitness,
    int numCities,
    int populationSize)
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
// Single-block reduction for simplicity (assumes populationSize <= 1024)
__global__ void findBestKernel(
    const int *d_population,
    const double *d_fitness,
    int *d_bestTour,
    double *d_bestFitness,
    int populationSize,
    int numCities)
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

    // The best in this block
    if (tid == 0)
    {
        // Atomically compare with global best
        double oldVal = atomicMax((unsigned long long *)d_bestFitness, __double_as_longlong(s_fitness[0]));
        double currentBestVal = __longlong_as_double(atomicCAS((unsigned long long *)d_bestFitness, __double_as_longlong(oldVal), __double_as_longlong(oldVal)));
        // if our found is better, update best tour
        if (s_fitness[0] > currentBestVal)
        {
            // Copy best individual's tour
            int bestPopIdx = s_indices[0];
            for (int i = 0; i < numCities; i++)
            {
                // TODO: FIX THIS for larger populations (i.e. > 1024)
                // Note: This is not atomic, so there's a race if multiple blocks find better at the same time.
                d_bestTour[i] = d_population[bestPopIdx * numCities + i];
            }
            *d_bestFitness = s_fitness[0];
        }
    }
}

__global__ void copyBestIndividualKernel(
    int *d_bestTour, // Best tour
    double *d_bestFitness,
    int *d_population,  // Population array
    int populationSize, // Number of individuals in the population
    int numCities)      // Number of cities in the TSP
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // Copy the best tour into the first position of the next generation
        for (int j = 0; j < numCities; j++)
        {
            d_population[j] = d_bestTour[j];
        }
    }
}

__global__ void selectionKernel(
    const double *d_fitness,
    int *d_population,
    int populationSize,
    int numCities,
    curandState *states)
{
    int half = populationSize / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Skip the top one individual
    if (idx == 0 || idx >= populationSize - 1)
        return;

    int offspringIdx = idx;

    curandState localState = states[idx];

    // Tournament for parent1
    int p1 = (int)(curand_uniform(&localState) * half);
    if (p1 >= half)
        p1 = half - 1;
    int p2 = (int)(curand_uniform(&localState) * half);
    if (p2 >= half)
        p2 = half - 1;
    int parent1 = (d_fitness[p1] > d_fitness[p2]) ? p1 : p2;

    // Tournament for parent2
    p1 = (int)(curand_uniform(&localState) * half);
    if (p1 >= half)
        p1 = half - 1;
    p2 = (int)(curand_uniform(&localState) * half);
    if (p2 >= half)
        p2 = half - 1;
    int parent2 = (d_fitness[p1] > d_fitness[p2]) ? p1 : p2;

    // Store chosen parents at the offspring location temporarily
    d_population[offspringIdx * numCities] += parent1 * (numCities + 1);
    d_population[offspringIdx * numCities + 1] += parent2 * (numCities + 1);

    states[idx] = localState;
}

__global__ void crossoverKernel(int *d_population, int populationSize, int numCities, curandState *states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Skip the top one individual
    if (idx == 0 || idx >= populationSize)
        return;

    int offspringIdx = idx;
    curandState localState = states[idx];

    // Retrieve parents
    int parent1 = d_population[offspringIdx * numCities] / (numCities + 1);
    int parent2 = d_population[offspringIdx * numCities + 1] / (numCities + 1);

    // Restore gene from masked values
    d_population[offspringIdx * numCities] -= parent1 * (numCities + 1);
    d_population[offspringIdx * numCities + 1] -= parent2 * (numCities + 1);

    // Perform order crossover
    int start = (int)(curand_uniform(&localState) * numCities);
    if (start >= numCities)
        start = numCities - 1;
    int end = (int)(curand_uniform(&localState) * numCities);
    if (end >= numCities)
        end = numCities - 1;
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

    for (int i = start; i <= end; i++)
    {
        child[i] += (p1[i] % (numCities + 1)) * (numCities + 1);
    }

    int currentIndex = 0;
    for (int i = 0; i < numCities; i++)
    {
        int val = p2[i] % (numCities + 1);
        bool found = false;
        // Check if val is already in child
        for (int j = start; j <= end; j++)
        {
            if (child[j] / (numCities + 1) == val)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            // find next available slot
            while (child[currentIndex] >= numCities + 1)
                currentIndex++;
            child[currentIndex] += val * (numCities + 1);
        }
    }

    states[idx] = localState;
}

__global__ void removeMaskKernel(int *d_population, int populationSize, int numCities, curandState *states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Skip the top one individual
    if (idx == 0 || idx >= populationSize)
        return;

    int offspringIdx = idx;
    int *child = &d_population[offspringIdx * numCities];

    for (int i = 0; i < numCities; i++)
    {
        child[i] /= (numCities + 1);
    }
}

// Mutation Kernel (Swap mutation):
// Randomly swap two cities in each individual.
__global__ void mutationKernel(int *d_population, int populationSize, int numCities, double mutationRate, curandState *states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0)
    {
        int offspringIdx = idx;
        curandState localState = states[idx];

        if (curand_uniform(&localState) < mutationRate)
        {
            int i = (int)(curand_uniform(&localState) * numCities);
            if (i >= numCities)
                i = numCities - 1;
            int j = (int)(curand_uniform(&localState) * numCities);
            if (j >= numCities)
                j = numCities - 1;
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
int main(int argc, char **argv)
{
    // Genetic Algorithm parameters
    if (argc != 4)
    {
        printf("Usage: %s <generations> <mutation_rate> <population_size>\n", argv[0]);
        return 1;
    }

    int generations = atoi(argv[1]);
    double mutationRate = atof(argv[2]);
    int populationSize = atoi(argv[3]);

    cout << "Running Genetic Algorithm with the following parameters:\n";
    cout << "Generations: " << generations << "\n";
    cout << "Mutation Rate: " << mutationRate << "\n";
    cout << "Population Size: " << populationSize << "\n\n";

    size_t stagnationCounter = 0;           // Tracks stagnation
    double epsilon = 1e-7;                  // Minimum improvement considered significant
    size_t maxStagnationGenerations = 1000; // Maximum allowed stagnation generations
    double previousBestFitness = 0.0;       // Fitness from the previous generation

    // Iterate over test cases defined in testcases.h
    for (size_t test_i = 0; test_i < testcases.size(); ++test_i)
    {
        const auto &[distanceMatrix, expectedDistance] = testcases[test_i];

        // timing
        auto start = chrono::high_resolution_clock::now();
        int numCities = (int)distanceMatrix.size();

        // Flatten distance matrix
        vector<int> h_distanceMatrix(numCities * numCities);
        for (int row = 0; row < numCities; row++)
        {
            for (int col = 0; col < numCities; col++)
            {
                h_distanceMatrix[row * numCities + col] = distanceMatrix[row][col];
            }
        }

        // Device memory
        int *d_population, *d_distanceMatrix, *d_bestTour;
        double *d_fitness, *d_bestFitness;
        curandState *d_states;

        CUDA_CHECK(cudaMalloc(&d_population, sizeof(int) * populationSize * numCities));
        CUDA_CHECK(cudaMalloc(&d_distanceMatrix, sizeof(int) * numCities * numCities));
        CUDA_CHECK(cudaMalloc(&d_fitness, sizeof(double) * populationSize));
        CUDA_CHECK(cudaMalloc(&d_bestTour, sizeof(int) * numCities));
        CUDA_CHECK(cudaMalloc(&d_bestFitness, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_states, sizeof(curandState) * populationSize));

        CUDA_CHECK(cudaMemcpy(d_distanceMatrix, h_distanceMatrix.data(), sizeof(int) * numCities * numCities, cudaMemcpyHostToDevice));

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
                // shared memory size might need adjustments
                findBestKernel<<<grid, BLOCK_SIZE, BLOCK_SIZE * (sizeof(double) + sizeof(int))>>>(d_population, d_fitness, d_bestTour, d_bestFitness, populationSize, numCities);
                cudaDeviceSynchronize();
            }

            // Copy best individual
            {
                copyBestIndividualKernel<<<1, 1>>>(d_bestTour, d_bestFitness, d_population, populationSize, numCities);
                cudaDeviceSynchronize();
            }

            double currentBestFitness;
            CUDA_CHECK(cudaMemcpy(&currentBestFitness, d_bestFitness, sizeof(double), cudaMemcpyDeviceToHost));

            // Check for improvement
            if (abs(currentBestFitness - previousBestFitness) < epsilon)
            {
                stagnationCounter++;
            }
            else
            {
                stagnationCounter = 0;
            }

            previousBestFitness = currentBestFitness;

            // Terminate if stagnation persists
            if (stagnationCounter >= maxStagnationGenerations)
            {
                cout << "Terminating early due to convergence detection.\n";
                break;
            }

            // Selection
            // Select 2 parents from the top 50% of the population
            {
                int grid = (populationSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
                selectionKernel<<<grid, BLOCK_SIZE>>>(d_fitness, d_population, populationSize, numCities, d_states);
                cudaDeviceSynchronize();
            }

            // Crossover
            // Crossover the selected parents pairs, replacing all population except the best one
            {
                int grid = (populationSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
                crossoverKernel<<<grid, BLOCK_SIZE>>>(d_population, populationSize, numCities, d_states);
                cudaDeviceSynchronize();
            }

            // Remove mask values
            {
                int grid = (populationSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
                removeMaskKernel<<<grid, BLOCK_SIZE>>>(d_population, populationSize, numCities, d_states);
                cudaDeviceSynchronize();
            }

            // Mutation
            {
                int grid = (populationSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
                mutationKernel<<<grid, BLOCK_SIZE>>>(d_population, populationSize, numCities, mutationRate, d_states);
                cudaDeviceSynchronize();
            }
        }

        // Final global best after last generation
        double hostBestFitness;
        vector<int> hostBestTour(numCities);
        CUDA_CHECK(cudaMemcpy(&hostBestFitness, d_bestFitness, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostBestTour.data(), d_bestTour, sizeof(int) * numCities, cudaMemcpyDeviceToHost));

        double bestDistance = 1.0 / hostBestFitness;

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed_time = end - start;

        cout << "Test Case " << test_i << ":\n";
        cout << "Number of Cities: " << numCities << "\n";
        cout << "Time Taken: " << elapsed_time.count() << " seconds\n";
        cout << "Best Distance: " << bestDistance << "\n";
        cout << "Expected Distance: " << expectedDistance << "\n";
        cout << "Best Tour: ";
        for (auto c : hostBestTour)
            cout << c << " ";
        cout << "\n-----------------------------------------\n";

        // Clean up
        CUDA_CHECK(cudaFree(d_population));
        CUDA_CHECK(cudaFree(d_distanceMatrix));
        CUDA_CHECK(cudaFree(d_fitness));
        CUDA_CHECK(cudaFree(d_bestTour));
        CUDA_CHECK(cudaFree(d_bestFitness));
        CUDA_CHECK(cudaFree(d_states));
    }

    return 0;
}