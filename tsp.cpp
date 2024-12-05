#include <iostream>
#include <vector>
#include <limits>
#include <tuple>
#include <chrono>
using namespace std;

using CostPathPair = pair<double, vector<int>>;

class TSP
{
public:
    static CostPathPair travel(const vector<vector<int>> &weights)
    {
        int start_node = 0;
        int n = weights.size();

        // dp[state][node] -> pair of minimal cost and path
        vector<vector<CostPathPair>> dp(1 << n, vector<CostPathPair>(n, {numeric_limits<double>::infinity(), {}}));

        // Initialize dp for first step
        for (int i = 0; i < n; ++i)
        {
            int state_only_i_visited = 1 << i;
            dp[state_only_i_visited][i] = {static_cast<double>(weights[start_node][i]), {i}};
        }

        // Rest of the function remains unchanged...
        // Iterate through all possible masks
        for (int current_mask = 0; current_mask < (1 << n); ++current_mask)
        {
            for (int dest = 0; dest < n; ++dest)
            {
                if (!(current_mask & (1 << dest)))
                    continue;

                int mask_dest_not_visited = current_mask ^ (1 << dest);
                for (int src = 0; src < n; ++src)
                {
                    if (!(current_mask & (1 << src)) || src == dest)
                        continue;

                    auto [prev_cost, prev_path] = dp[mask_dest_not_visited][src];
                    double new_cost = prev_cost + weights[src][dest];
                    if (new_cost < dp[current_mask][dest].first)
                    {
                        dp[current_mask][dest] = {new_cost, prev_path};
                        dp[current_mask][dest].second.push_back(dest);
                    }
                }
            }
        }

        // All nodes visited, return to the start node
        return dp[(1 << n) - 1][0];
    }
};

int main()
{
    // Define test cases as a vector of pairs (weights, expected_cost)
    vector<pair<vector<vector<int>>, int>> testcases = {
        // Test Case 1
        {{{0, 5, 8}, {4, 0, 8}, {4, 5, 0}}, 17},
        // Test Case 2
        {{{0, 10, 15, 20}, {10, 0, 35, 25}, {15, 35, 0, 30}, {20, 25, 30, 0}}, 80},
        // Test Case 3
        {{{0, 10, 15, 20, 25}, {10, 0, 40, 45, 50}, {15, 40, 0, 65, 70}, {20, 45, 65, 0, 85}, {25, 50, 70, 85, 0}}, 200},
        // Test Case 4
        {{{0, 10000, 10000, 8, 10000, 10}, {10000, 0, 10000, 10000, 2, 12}, {10000, 10000, 0, 6, 4, 10000}, {8, 10000, 6, 0, 10000, 10000}, {10000, 2, 4, 10000, 0, 10000}, {10, 12, 10000, 10000, 10000, 0}}, 42},
        // Test Case 5
        {{{0, 1, 1, 1, 1, 1, 1}, {1, 0, 1, 1, 1, 1, 1}, {1, 1, 0, 1, 1, 1, 1}, {1, 1, 1, 0, 1, 1, 1}, {1, 1, 1, 1, 0, 1, 1}, {1, 1, 1, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 1, 0}}, 7},
        // Test Case 6
        {{{0, 1, 2, 3, 4, 5, 6, 7}, {1, 0, 8, 9, 10, 11, 12, 13}, {2, 8, 0, 14, 15, 16, 17, 18}, {3, 9, 14, 0, 19, 20, 21, 22}, {4, 10, 15, 19, 0, 23, 24, 25}, {5, 11, 16, 20, 23, 0, 26, 27}, {6, 12, 17, 21, 24, 26, 0, 28}, {7, 13, 18, 22, 25, 27, 28, 0}}, 108},
        // Test Case 7
        {{{0, 1, 2, 3, 4, 5, 6, 7, 8}, {1, 0, 9, 10, 11, 12, 13, 14, 15}, {2, 9, 0, 16, 17, 18, 19, 20, 21}, {3, 10, 16, 0, 22, 23, 24, 25, 26}, {4, 11, 17, 22, 0, 27, 28, 29, 30}, {5, 12, 18, 23, 27, 0, 31, 32, 33}, {6, 13, 19, 24, 28, 31, 0, 34, 35}, {7, 14, 20, 25, 29, 32, 34, 0, 36}, {8, 15, 21, 26, 30, 33, 35, 36, 0}}, 154},
        // Test Case 8
        {{{0, 10, 15, 20, 25, 30, 35, 40, 45, 50}, {10, 0, 55, 60, 65, 70, 75, 80, 85, 90}, {15, 55, 0, 95, 100, 105, 110, 115, 120, 125}, {20, 60, 95, 0, 130, 135, 140, 145, 150, 155}, {25, 65, 100, 130, 0, 160, 165, 170, 175, 180}, {30, 70, 105, 135, 160, 0, 190, 195, 200, 205}, {35, 75, 110, 140, 165, 190, 0, 220, 225, 230}, {40, 80, 115, 145, 170, 195, 220, 0, 235, 240}, {45, 85, 120, 150, 175, 200, 225, 235, 0, 245}, {50, 90, 125, 155, 180, 205, 230, 240, 245, 0}}, 1100},
        // Test Case 9
        {{{0, 17, 15, 16, 16, 15, 19, 19, 16, 18, 20}, {10, 0, 15, 16, 15, 12, 19, 19, 16, 18, 20}, {10, 17, 0, 16, 16, 16, 19, 19, 16, 18, 20}, {10, 17, 14, 0, 16, 16, 19, 19, 16, 3, 8}, {10, 17, 15, 1, 0, 16, 19, 19, 16, 4, 9}, {10, 17, 15, 16, 16, 0, 19, 19, 6, 18, 20}, {10, 17, 15, 3, 2, 16, 0, 19, 15, 6, 11}, {10, 11, 15, 16, 15, 16, 19, 0, 16, 18, 20}, {8, 17, 15, 16, 16, 16, 19, 19, 0, 18, 20}, {10, 17, 11, 16, 16, 16, 19, 19, 16, 0, 5}, {10, 17, 6, 16, 16, 16, 19, 18, 16, 18, 0}}, 92},
    };

    // Run test cases
    for (size_t i = 0; i < testcases.size(); ++i)
    {
        const auto &[weights, expected] = testcases[i];
        auto start_time = chrono::high_resolution_clock::now();
        auto [cost, path] = TSP::travel(weights);
        auto end_time = chrono::high_resolution_clock::now();

        cout << "Test Case " << i + 1 << ":\n";
        cout << "Expected Cost: " << expected << "\n";
        cout << "Calculated Cost: " << cost << "\n";
        cout << "Path: ";
        for (int node : path)
            cout << node << " ";
        cout << "\n";

        chrono::duration<double> elapsed_time = end_time - start_time;
        cout << "Time Taken: " << elapsed_time.count() << " seconds\n";
        cout << "-----------------------------------------\n";
    }

    return 0;
}