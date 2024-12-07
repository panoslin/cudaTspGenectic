# CUDA TSP Genectic Algorithm


# Pseudo Code
initialize population with random valid tours;
evaluate fitness of each individual;

while (termination condition not met) {
    select parents based on fitness;
    perform crossover to generate offspring;
    apply mutation to offspring;
    evaluate fitness of offspring;
    replace old population with new offspring, keeping elites;
}

return best solution;

