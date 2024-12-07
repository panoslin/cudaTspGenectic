#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <tuple>
#include <chrono>
#include <cassert>
#include "../include/testcases.h"

using namespace std;

/////////////////////////////////////////////
// CUDA KERNELS AND DEVICE FUNCTIONS
/////////////////////////////////////////////

__global__ void calculateFitnessKernel(const int *distanceMatrix, const int *population, double *fitnessArray,
                                       int numCities, int populationSize)
{
    // Each thread corresponds to one individual
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize)
        return;

    // Compute distance of the idx-th individual's tour
    // population is laid out as [tour0_city0, tour0_city1, ..., tour0_cityN-1, tour1_city0, ...]
    const int *individual = &population[idx * numCities];

    double totalDistance = 0.0;
    for (int i = 0; i < numCities - 1; ++i)
    {
        int from = individual[i];
        int to = individual[i + 1];
        totalDistance += distanceMatrix[from * numCities + to];
    }
    // Close the loop
    {
        int from = individual[numCities - 1];
        int to = individual[0];
        totalDistance += distanceMatrix[from * numCities + to];
    }

    // Fitness is inverse of distance
    fitnessArray[idx] = 1.0 / totalDistance;
}

/////////////////////////////////////////////
// GENETIC ALGORITHM CLASS
/////////////////////////////////////////////

class GeneticAlgorithmTSPCuda
{
private:
    vector<vector<int>> weights;
    size_t populationSize;
    size_t generations;
    double mutationRate;

    mt19937 rng;

    double calculateTourDistance(const vector<int> &tour) const
    {
        double totalDistance = 0.0;
        for (size_t i = 0; i < tour.size() - 1; ++i)
            totalDistance += weights[tour[i]][tour[i + 1]];
        totalDistance += weights[tour.back()][tour.front()];
        return totalDistance;
    }

    double calculateFitness(const vector<int> &tour) const
    {
        return 1.0 / calculateTourDistance(tour);
    }

    // Order Crossover (similar to OX)
    vector<int> orderCrossover(const vector<int> &parent1, const vector<int> &parent2)
    {
        size_t n = parent1.size();
        vector<int> child(n, -1);

        uniform_int_distribution<size_t> dist(0, n - 1);
        size_t start = dist(rng);
        size_t end = dist(rng);
        if (start > end)
            swap(start, end);

        // Copy segment from parent1
        for (size_t i = start; i <= end; ++i)
        {
            child[i] = parent1[i];
        }

        // Fill remaining positions from parent2
        size_t currentIndex = 0;
        for (size_t i = 0; i < n; ++i)
        {
            if (find(child.begin(), child.end(), parent2[i]) == child.end())
            {
                while (child[currentIndex] != -1)
                    currentIndex++;
                child[currentIndex] = parent2[i];
            }
        }
        return child;
    }

    // Swap Mutation
    void swapMutation(vector<int> &tour)
    {
        uniform_int_distribution<size_t> dist(0, tour.size() - 1);
        size_t i = dist(rng);
        size_t j = dist(rng);
        std::swap(tour[i], tour[j]);
    }

    // Device-related helper functions
    // Flatten distance matrix
    void flattenDistanceMatrix(const vector<vector<int>> &matrix, vector<int> &flat)
    {
        int n = (int)matrix.size();
        flat.resize(n * n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                flat[i * n + j] = matrix[i][j];
    }

public:
    GeneticAlgorithmTSPCuda(const vector<vector<int>> &distanceMatrix,
                            size_t popSize = 100,
                            size_t gen = 1000,
                            double mutation = 0.1)
        : weights(distanceMatrix),
          populationSize(popSize),
          generations(gen),
          mutationRate(mutation),
          rng(random_device{}()) {}

    tuple<vector<int>, double> solve()
    {
        size_t numCities = weights.size();
        vector<vector<int>> population(populationSize, vector<int>(numCities));

        // Initialize population
        for (auto &tour : population)
        {
            iota(tour.begin(), tour.end(), 0);
            shuffle(tour.begin(), tour.end(), rng);
        }

        vector<int> bestTour;
        double bestFitness = 0.0;

        // Flatten distance matrix for GPU
        vector<int> flatDist;
        flattenDistanceMatrix(weights, flatDist);
        int n = (int)numCities;
        int popSize = (int)populationSize;

        // Allocate GPU memory
        int *d_distanceMatrix = nullptr;
        int *d_population = nullptr;
        double *d_fitnessArray = nullptr;

        cudaMalloc((void **)&d_distanceMatrix, n * n * sizeof(int));
        cudaMalloc((void **)&d_population, popSize * n * sizeof(int));
        cudaMalloc((void **)&d_fitnessArray, popSize * sizeof(double));

        cudaMemcpy(d_distanceMatrix, flatDist.data(), n * n * sizeof(int), cudaMemcpyHostToDevice);

        for (size_t gen = 0; gen < generations; ++gen)
        {
            // Flatten population
            vector<int> flatPop(popSize * n);
            for (int i = 0; i < popSize; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    flatPop[i * n + j] = population[i][j];
                }
            }

            cudaMemcpy(d_population, flatPop.data(), popSize * n * sizeof(int), cudaMemcpyHostToDevice);

            // Run kernel to compute fitness
            int blockSize = 128;
            int gridSize = (popSize + blockSize - 1) / blockSize;
            calculateFitnessKernel<<<gridSize, blockSize>>>(d_distanceMatrix, d_population, d_fitnessArray, n, popSize);
            cudaDeviceSynchronize();

            // Copy fitness results back
            vector<double> fitnessScoresHost(popSize);
            cudaMemcpy(fitnessScoresHost.data(), d_fitnessArray, popSize * sizeof(double), cudaMemcpyDeviceToHost);

            // Pair fitness with tours
            vector<pair<double, vector<int>>> fitnessScores;
            fitnessScores.reserve(popSize);
            for (int i = 0; i < popSize; i++)
            {
                double f = fitnessScoresHost[i];
                if (f > bestFitness)
                {
                    bestFitness = f;
                    bestTour = population[i];
                }
                fitnessScores.push_back({f, population[i]});
            }

            // Sort by fitness descending
            sort(fitnessScores.rbegin(), fitnessScores.rend());

            // Elitism
            vector<vector<int>> newPopulation;
            newPopulation.push_back(fitnessScores[0].second);

            // Selection/Crossover/Mutation
            uniform_int_distribution<size_t> dist(0, populationSize / 2);
            while (newPopulation.size() < populationSize)
            {
                size_t p1 = dist(rng);
                size_t p2 = dist(rng);
                auto child = orderCrossover(fitnessScores[p1].second, fitnessScores[p2].second);

                if (uniform_real_distribution<double>(0.0, 1.0)(rng) < mutationRate)
                {
                    swapMutation(child);
                }

                newPopulation.push_back(child);
            }

            population = move(newPopulation);
        }

        // Cleanup GPU memory
        cudaFree(d_distanceMatrix);
        cudaFree(d_population);
        cudaFree(d_fitnessArray);

        double bestDistance = 1.0 / bestFitness;
        return {bestTour, bestDistance};
    }
};

int main()
{
    for (size_t i = 0; i < testcases.size(); ++i)
    {
        const auto &[distanceMatrix, expected] = testcases[i];
        // timing for cuda kernel
        auto start_time = chrono::high_resolution_clock::now();

        GeneticAlgorithmTSPCuda ga(distanceMatrix);

        auto [bestTour, bestDistance] = ga.solve();

        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed_time = end_time - start_time;
        cout << "Time Taken: " << elapsed_time.count() << " seconds\n";
        cout << "Best Tour: ";
        for (int city : bestTour)
            cout << city << " ";
        cout << "\nBest Distance: " << bestDistance << endl;
        cout << "Expected Distance: " << expected << endl;
        cout << "-----------------------------------------\n";
    }
    return 0;
}