#include <iostream>
#include <vector>
#include <limits>
#include <tuple>
#include <chrono>
#include "../include/testcases.h"

using namespace std;

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