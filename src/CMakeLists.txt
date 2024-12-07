add_executable(tsp tsp.cpp)
add_executable(tsp_genetic tsp_genetic.cpp)
add_executable(tsp_genetic_cuda tsp_genetic_cuda.cu)

# Link CUDA libraries to the CUDA executable
target_link_libraries(tsp_genetic_cuda PRIVATE cuda)