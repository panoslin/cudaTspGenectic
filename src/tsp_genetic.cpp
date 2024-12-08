#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <tuple>
#include <limits>
#include <chrono>
#include "../include/testcases.h"

using namespace std;

class GeneticAlgorithmTSP
{
private:
    vector<vector<int>> weights;
    size_t populationSize;
    size_t generations;
    double mutationRate;

    // Random generator
    mt19937 rng;

    // Calculate the total distance of a tour
    double calculateTourDistance(const vector<int> &tour) const
    {
        double totalDistance = 0.0;
        for (size_t i = 0; i < tour.size() - 1; ++i)
        {
            totalDistance += weights[tour[i]][tour[i + 1]];
        }
        totalDistance += weights[tour.back()][tour.front()];
        return totalDistance;
    }

    // Calculate fitness (inverse of the tour distance)
    double calculateFitness(const vector<int> &tour) const
    {
        return 1.0 / calculateTourDistance(tour);
    }

    // Partially Mapped Crossover (PMX)
    vector<int> orderCrossover(const vector<int> &parent1, const vector<int> &parent2)
    {
        size_t n = parent1.size();
        vector<int> child(n, -1);

        // Select random crossover points
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
        swap(tour[i], tour[j]);
    }

public:
    // Constructor
    GeneticAlgorithmTSP(
        const vector<vector<int>> &distanceMatrix,
        size_t popSize = 100,
        size_t gen = 1000,
        double mutation = 0.1) : weights(distanceMatrix),
                                 populationSize(popSize),
                                 generations(gen),
                                 mutationRate(mutation),
                                 rng(random_device{}()) {}

    // Solve the TSP using Genetic Algorithm
    CostPathPair solve()
    {
        size_t numCities = weights.size();
        vector<vector<int>> population(populationSize, vector<int>(numCities));

        // Initialize population with random tours
        for (auto &tour : population)
        {
            iota(tour.begin(), tour.end(), 0);
            shuffle(tour.begin(), tour.end(), rng);
        }

        vector<int> bestTour;
        double bestFitness = 0.0;

        size_t stagnationCounter = 0;         // Tracks stagnation
        double epsilon = 1e-7;                // Minimum improvement considered significant
        size_t maxStagnationGenerations = 100; // Maximum allowed stagnation generations
        double previousBestFitness = 0.0;     // Fitness from the previous generation

        for (size_t gen = 0; gen < generations; ++gen)
        {
            // Calculate fitness for all individuals
            vector<CostPathPair> fitnessScores;
            for (const auto &tour : population)
            {
                double fitness = calculateFitness(tour);
                fitnessScores.emplace_back(fitness, tour);
                if (fitness > bestFitness)
                {
                    bestFitness = fitness;
                    bestTour = tour;
                }
            }

            // Sort population by fitness (descending order)
            sort(fitnessScores.rbegin(), fitnessScores.rend());

            double currentBestFitness = fitnessScores[0].first; // Assuming fitnessScores is sorted
            // Check for improvement
            if (abs(currentBestFitness - previousBestFitness) < epsilon)
            {
                stagnationCounter++; // Increment stagnation count
            }
            else
            {
                stagnationCounter = 0; // Reset stagnation count
            }

            // Update the previous fitness value
            previousBestFitness = currentBestFitness;

            // Terminate if stagnation persists
            if (stagnationCounter >= maxStagnationGenerations)
            {
                cout << "Terminating early due to convergence detection.\n";
                break;
            }

            // Elitism: Retain the best individual
            vector<vector<int>> newPopulation = {fitnessScores[0].second};

            // Selection, Crossover, and Mutation
            uniform_int_distribution<size_t> dist(0, populationSize / 2);
            while (newPopulation.size() < populationSize)
            {
                size_t p1 = dist(rng);
                size_t p2 = dist(rng);

                // Perform order crossover
                auto child = orderCrossover(fitnessScores[p1].second, fitnessScores[p2].second);

                // Apply mutation with a given probability
                if (uniform_real_distribution<double>(0.0, 1.0)(rng) < mutationRate)
                {
                    swapMutation(child);
                }

                newPopulation.push_back(child);
            }

            population = move(newPopulation);
        }

        return {calculateTourDistance(bestTour), bestTour};
    }
};

int main(int argc, char *argv[])
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

    for (size_t i = 0; i < testcases.size(); ++i)
    {
        const auto &[distanceMatrix, expected] = testcases[i];

        auto start_time = chrono::high_resolution_clock::now();

        // Create GeneticAlgorithmTSP instance
        GeneticAlgorithmTSP ga(distanceMatrix, populationSize, generations, mutationRate);

        // Solve the TSP
        auto [bestDistance, bestTour] = ga.solve();

        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed_time = end_time - start_time;

        cout << "Test Case " << i << ":\n";
        cout << "Number of Cities: " << distanceMatrix.size() << "\n";
        cout << "Time Taken: " << elapsed_time.count() << " seconds\n";
        cout << "Best Distance: " << bestDistance << "\n";
        cout << "Expected Distance: " << expected << "\n";
        cout << "Best Tour: ";
        for (auto c : bestTour)
            cout << c << " ";
        cout << "\n-----------------------------------------\n";
    }
    return 0;
}