import numpy as np
from run_this import run_this

class GeneticAlgorithm:
    def __init__(self, pop_size, gene_length, crossover_rate, mutation_rate, generations):
        if gene_length < 2:
            raise ValueError("Gene length must be greater than 2 for crossover to work.")
        self.pop_size = pop_size
        self.gene_length = gene_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations


    def initialize_population(self):
        population = np.random.rand(self.pop_size, self.gene_length)
        population[:, 0] = np.random.uniform(0, 1, size=self.pop_size)  # 学习率在0到1之间
        population[:, 1] = np.random.randint(50, 2001, size=self.pop_size)  # 经验池容量在50到2000之间
        return population


    def fitness(self, individual):
        lr = individual[0]
        capacity = int(individual[1])
        return run_this(lr, capacity)


    def select(self, population, fitness_values):
        selected = population[np.argsort(fitness_values)][-self.pop_size//2:]
        return selected

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate and self.gene_length > 2:
            crossover_point = np.random.randint(1, self.gene_length - 1)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        else:
            return parent1, parent2

    def mutate(self, individual):
        for i in range(self.gene_length):
            if np.random.rand() < self.mutation_rate:
                if i == 0:
                    individual[i] = np.random.rand()
                else:
                    individual[i] = np.random.randint(50, 2001)
        return individual

    def optimize(self):
        population = self.initialize_population()
        best_fitness = -np.inf
        no_improvement_generations = 0
        max_no_improvement = 20  # 设定一个早停阈值

        for generation in range(self.generations):
            fitness_values = np.array([self.fitness(individual) for individual in population])
            selected_population = self.select(population, fitness_values)
            new_population = []

            while len(new_population) < self.pop_size:
                if len(selected_population) < 2:
                    new_population.extend(selected_population)
                    break

                parents_indices = np.random.choice(len(selected_population), 2, replace=False)
                parent1, parent2 = selected_population[parents_indices[0]], selected_population[parents_indices[1]]
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                if len(new_population) < self.pop_size:
                    new_population.append(self.mutate(child2))

            population = np.array(new_population[:self.pop_size])
            current_best_fitness = np.max(fitness_values)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                no_improvement_generations = 0
            else:
                no_improvement_generations += 1

            if no_improvement_generations >= max_no_improvement:
                break

        fitness_values = np.array([self.fitness(individual) for individual in population])
        best_individual = population[np.argmax(fitness_values)]
        return best_individual

