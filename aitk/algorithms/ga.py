# -*- coding: utf-8 -*-
# ****************************************************************
# aitk.algorithms: Algorithms for AI
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.algorithms
#
# ****************************************************************

import random
import math
import io

import matplotlib.pyplot as plt

from aitk.utils import progress_bar

class GeneticAlgorithm(object):
    """
    A genetic algorithm is a model of biological evolution.  It
    maintains a population of chromosomes.  Each chromosome is
    represented as a list.  A fitness function must be
    defined to score each chromosome.  Initially, a random population
    is created. Then a series of generations are executed.  Each
    generation, parents are selected from the population based on
    their fitness.  More highly fit chromosomes are more likely to be
    selected to create children.  With some probability crossover will
    be done to model sexual reproduction.  With some very small
    probability mutations will occur.  A generation is complete once
    all of the original parents have been replaced by children.  This
    process continues until the maximum generation is reached or when
    the is_done method returns True.
    """

    def __init__(self, length, popSize, verbose=False):
        """
        Create a GeneticAlgorithm instance.

        Args:
            length (int): length of chromosome
            popSize (int): number of chromosomes in population
            verbose (bool): optional, show details if True
        """
        self.verbose = verbose      # Set to True to see more info displayed
        self.progress_type = "notebook" # "notebook" or "tqdm"
        self.length = length        # Length of the chromosome
        self.popSize = popSize      # Size of the population
        self.generations = None     # Maximum generation
        self.crossover_rate = None  # Probability of crossover
        self.mutation_rate = None   # Probability of mutation (per bit)
        self.elite_percent = None   # Percent elite
        self.generation = 0         # Current generation of evolution
        self._widget = None
        self.title = "Genetic Algorithm Population Statistics"
        self.bestList = []          # Best fitness per generation
        self.avgList = []           # Avg fitness per generation
        print("Genetic algorithm")
        print(f"  Chromosome length: {self.length}")
        print(f"  Population size: {self.popSize}")

    def initialize_population(self):
        """
        Initialize each chromosome in the population with a random
        chromosome.

        Returns: None
        Result: Initializes self.population
        """
        self.bestEver = None        # Best member ever in this evolution
        self.bestEverScore = 0      # Fitness of best member ever
        self.population = None      # Population is a list of chromosomes
        self.totalFitness = None    # Total fitness in entire population
        self.bestList.clear()          # Best fitness per generation
        self.avgList.clear()           # Avg fitness per generation
        self.scores = [0] * self.popSize  # Fitnesses of all members of population
        self.population = []
        for i in range(self.popSize):
            chromosome = self.make_random_chromosome()
            self.population.append(chromosome)

    def reset(self):
        self.generation = 0

    def evaluate_population(self, **kwargs):
        """
        Computes the fitness of every chromosome in population.  Saves the
        fitness values to the list self.scores.  Checks whether the
        best fitness in the current population is better than
        self.bestEverScore. If so, updates this variable and saves the
        chromosome to self.bestEver.  Computes the total fitness of
        the population and saves it in self.totalFitness. Saves the
        current bestEverScore and the current average score to the
        lists self.bestList and self.avgList.

        Returns: None
        """
        for i, chromosome in enumerate(self.population):
            self.scores[i] = self.fitness(chromosome, index=i, **kwargs)
        bestScore = max(self.scores)
        best = self.population[self.scores.index(bestScore)]
        if bestScore > self.bestEverScore:
            self.bestEver = best[:]
            self.bestEverScore = bestScore
        self.totalFitness = sum(self.scores)
        self.bestList.append(self.bestEverScore)
        self.avgList.append(sum(self.scores)/float(self.popSize))

    def report(self):
        print(f"Generation {self.generation:4d} Best fitness {self.bestEverScore:4.2f}")

    def selection(self):
        """
        Each chromosome's chance of being selected for reproduction is
        based on its fitness.  The higher the fitness the more likely
        it will be selected.  Uses the roulette wheel strategy.

        Returns: A COPY of the selected chromosome.
        """
        spin = random.random() * self.totalFitness
        partialSum = 0
        index = 0
        for i in range(self.popSize):
            partialSum += self.scores[i]
            if partialSum > spin:
                break
        return self.population[i][:]

    def crossover(self, parent1, parent2):
        """
        With probability self.crossover_rate, recombine the genetic
        material of the given parents at a random location between
        1 and the length-1 of the chromosomes. If no crossover is
        performed, then return the original parents.

        Returns: Two children
        """
        if random.random() < self.crossover_rate:
            crossPoint = random.randrange(1, self.length)
            if self.verbose:
                print(f"Crossing over at position {crossPoint}")
            child1 = parent1[0:crossPoint] + parent2[crossPoint:]
            child2 = parent2[0:crossPoint] + parent1[crossPoint:]
            return child1, child2
        else:
            if self.verbose:
                print("No crossover performed")
            return parent1, parent2

    def mutation(self, chromosome):
        """
        With probability self.mutation_rate, mutate positions in the
        chromosome.

        Returns: None
        Result: Modifies the given chromosome
        """
        for i in range(self.length):
            if random.random() < self.mutation_rate:
                if self.verbose:
                    print(f"Mutating at position {i}")
                gene = self.mutate_gene(chromosome[i])
                while chromosome[i] == gene:
                    gene = self.mutate_gene(chromosome[i])
                chromosome[i] = gene

    def one_generation(self):
        """
        Execute one generation of the evolution. Each generation,
        repeatedly select two parents, call crossover to generate
        two children.  Call mutate on each child.  Finally add both
        children to the new population.  Continue until the new
        population is full.

        Returns: None
        Result: Replaces self.pop with a new population.
        """
        # First, select the most elite to carry on unchanged:
        elite_size = math.floor(self.elite_percent * self.popSize)
        fittest = sorted(list(enumerate(self.scores)), key=lambda item: item[1],
                         reverse=True)
        newPop = []
        for i in range(elite_size):
            index, score = fittest[i]
            newPop.append(self.population[index])
        # Next, fill up rest of population:
        while len(newPop) < self.popSize:
            parent1 = self.selection()
            parent2 = self.selection()
            if self.verbose:
                print("Parents:")
                print(parent1)
                print(parent2)
            child1, child2 = self.crossover(parent1, parent2)
            self.mutation(child1)
            self.mutation(child2)
            if self.verbose:
                print("Children:")
                print(child1)
                print(child2)
            newPop.append(child1)
            newPop.append(child2)
        if len(newPop) > self.popSize:
            newPop.pop(random.randrange(len(newPop)))
        self.population = newPop
        self.generation += 1

    def evolve(self, generations, crossover_rate=0.7, mutation_rate=0.001,
               elite_percent=0.0, **kwargs):
        """
        Run a series of generations until a maximum generation is
        reached or self.is_done() returns True.

        Returns the best chromosome ever found over the course of
        the evolution, which is stored in self.bestEver.
        """
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_percent = elite_percent
        elite_count = math.floor(self.elite_percent * self.popSize)

        print(f"Maximum number of generations: {self.generations}")
        print(f"  Elite percentage {self.elite_percent} ({elite_count}/{self.popSize} chromosomes per generation)")
        print(f"  Crossover rate: {self.crossover_rate} (~{int((self.popSize - elite_count) * self.crossover_rate)}/{self.popSize - elite_count} crossovers per generation)")
        print(f"  Mutation rate: {self.mutation_rate} (~{int((self.popSize - elite_count) * self.length * self.mutation_rate)}/{(self.popSize - elite_count) * self.length} genes per generation)")

        if self.generation == 0:
            print("Evaluating initial population...")
            self.initialize_population()
            self.evaluate_population(**kwargs)
            self.update()
            print("Done!")

        progress = progress_bar(
            range(self.generation, self.generations),
            progress_type=self.progress_type,
        )
        for generation in progress:
            progress.set_description("Best fitness %.2f" % self.bestEverScore)
            progress.refresh() # to show immediately the update
            try:
                self.one_generation()
                self.evaluate_population(**kwargs)
            except KeyboardInterrupt:
                break

            self.update()
            if self.is_done():
                break
            self.report()

        if self.generation >= self.generations:
            print("Max generations reached")
        elif self.is_done():
            print("Solution found")
        else:
            print("Manually interrupted")
            self.update()

        return self.bestEver

    def make_plot(self):
        gens = range(len(self.bestList))
        plt.plot(gens, self.bestList, label="Best")
        plt.plot(gens, self.avgList, label="Average")
        plt.legend()
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        if self.title:
            plt.title(self.title)

    def plot_stats(self, title=None):
        """
        Plots a summary of the GA's progress over the generations.
        """
        if title:
            self.title = title

        self.make_plot()
        plt.show()

    def get_svg(self):
        self.make_plot()
        bytes = io.BytesIO()
        plt.savefig(bytes, format="svg")
        plt.close()
        img_bytes = bytes.getvalue()
        return img_bytes.decode()

    def get_widget(self):
        from ipywidgets import HTML

        if self._widget is None:
            svg = self.get_svg()
            self._widget = HTML(svg)

        return self._widget

    def watch(self, title=None):
        from IPython.display import display

        # Watched items get a border
        # Need width and height; we get it out of svg:
        #header = svg.split("\n")[0]
        #width = int(re.match('.*width="(\d*)px"', header).groups()[0])
        #height = int(re.match('.*height="(\d*)px"', header).groups()[0])
        #div = """<div style="outline: 5px solid #1976D2FF; width: %spx; height: %spx;">%s</div>""" % (width, height, svg)
        #self._widget.value = div

        if title:
            self.title = title

        display(self.get_widget())

    def update(self):
        if self._widget is not None:
            svg = self.get_svg()
            self._widget.value = svg

    def make_random_chromosome(self):
        """
        Function to generate a new random chromosome.
        """
        return [self.make_random_gene() for i in range(self.length)]

    def mutate_gene(self, gene):
        """
        Function to mutate gene.
        """
        # Override this if needed
        return self.make_random_gene()

    def is_done(self):
        """
        If there is a stopping critera, it will be different for
        each problem. As a default, we do not stop until max
        epochs are reached.
        """
        # Override this if needed
        return False

    def fitness(self, chromosome, **kwargs):
        """
        The fitness function will change for each problem.  Therefore
        it is not defined here.  To use this class to solve a
        particular problem, inherit from this class and define this
        method.
        """
        # Override this
        pass

    def make_random_gene(self):
        """
        Function to generate a new random gene.
        """
        # Override this
        pass
