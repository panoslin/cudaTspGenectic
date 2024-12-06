#ifndef TESTCASES_H
#define TESTCASES_H

#include <vector>
#include <utility>

using namespace std;

const vector<pair<vector<vector<int>>, int>> testcases = {
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

        {{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
          {1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
          {2, 3, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
          {3, 4, 5, 0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
          {4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
          {5, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 16, 17, 18, 19},
          {6, 7, 8, 9, 10, 11, 0, 13, 14, 15, 16, 17, 18, 19, 20},
          {7, 8, 9, 10, 11, 12, 13, 0, 15, 16, 17, 18, 19, 20, 21},
          {8, 9, 10, 11, 12, 13, 14, 15, 0, 17, 18, 19, 20, 21, 22},
          {9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 19, 20, 21, 22, 23},
          {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24},
          {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 0, 23, 24, 25},
          {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 25, 26},
          {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0, 27},
          {14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 0}},
         210},
        {{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
          {1, 0, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37},
          {2, 20, 0, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54},
          {3, 21, 38, 0, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70},
          {4, 22, 39, 55, 0, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85},
          {5, 23, 40, 56, 71, 0, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99},
          {6, 24, 41, 57, 72, 86, 0, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112},
          {7, 25, 42, 58, 73, 87, 100, 0, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124},
          {8, 26, 43, 59, 74, 88, 101, 113, 0, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135},
          {9, 27, 44, 60, 75, 89, 102, 114, 125, 0, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145},
          {10, 28, 45, 61, 76, 90, 103, 115, 126, 136, 0, 146, 147, 148, 149, 150, 151, 152, 153, 154},
          {11, 29, 46, 62, 77, 91, 104, 116, 127, 137, 146, 0, 155, 156, 157, 158, 159, 160, 161, 162},
          {12, 30, 47, 63, 78, 92, 105, 117, 128, 138, 147, 155, 0, 163, 164, 165, 166, 167, 168, 169},
          {13, 31, 48, 64, 79, 93, 106, 118, 129, 139, 148, 156, 163, 0, 170, 171, 172, 173, 174, 175},
          {14, 32, 49, 65, 80, 94, 107, 119, 130, 140, 149, 157, 164, 170, 0, 176, 177, 178, 179, 180},
          {15, 33, 50, 66, 81, 95, 108, 120, 131, 141, 150, 158, 165, 171, 176, 0, 181, 182, 183, 184},
          {16, 34, 51, 67, 82, 96, 109, 121, 132, 142, 151, 159, 166, 172, 177, 181, 0, 185, 186, 187},
          {17, 35, 52, 68, 83, 97, 110, 122, 133, 143, 152, 160, 167, 173, 178, 182, 185, 0, 188, 189},
          {18, 36, 53, 69, 84, 98, 111, 123, 134, 144, 153, 161, 168, 174, 179, 183, 186, 188, 0, 190},
          {19, 37, 54, 70, 85, 99, 112, 124, 135, 145, 154, 162, 169, 175, 180, 184, 187, 189, 190, 0}},
         1670},

    };

#endif