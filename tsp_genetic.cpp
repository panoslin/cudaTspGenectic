#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <tuple>
#include <limits>
#include <chrono>
#include "testcases.h"

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
        totalDistance += weights[tour.back()][tour.front()]; // Return to start
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
    tuple<vector<int>, double> solve()
    {
        size_t numCities = weights.size();
        vector<vector<int>> population(populationSize, vector<int>(numCities));

        // Initialize population with random tours
        for (auto &tour : population)
        {
            iota(tour.begin(), tour.end(), 0); // [0, 1, 2, ..., n-1]
            shuffle(tour.begin(), tour.end(), rng);
        }

        vector<int> bestTour;
        double bestFitness = 0.0;

        for (size_t gen = 0; gen < generations; ++gen)
        {
            // Calculate fitness for all individuals
            vector<pair<double, vector<int>>> fitnessScores;
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

            // Elitism: Retain the best individual
            vector<vector<int>> newPopulation = {fitnessScores[0].second};

            // Selection, Crossover, and Mutation
            while (newPopulation.size() < populationSize)
            {
                uniform_int_distribution<size_t> dist(0, populationSize / 2); // Select from top 50%
                size_t p1 = dist(rng);
                size_t p2 = dist(rng);

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

        return {bestTour, calculateTourDistance(bestTour)};
    }
};

int main()
{
    for (size_t i = 0; i < testcases.size(); ++i)
    {
        const auto &[distanceMatrix, expected] = testcases[i];
        // Create GeneticAlgorithmTSP instance
        GeneticAlgorithmTSP ga(distanceMatrix);

        // Solve the TSP
        auto [bestTour, bestDistance] = ga.solve();

        // Output the results
        cout << "Best Tour: ";
        for (int city : bestTour)
            cout << city << " ";
        cout << "\nBest Distance: " << bestDistance << endl;
        cout << "Expected Distance: " << expected << endl;
        cout << "-----------------------------------------\n";
    }
    return 0;
}