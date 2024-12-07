project(TSP_GeneticAlgorithm LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.18) # Use 3.18 or higher for CUDA support

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 61)

add_subdirectory(src)

include_directories(include)