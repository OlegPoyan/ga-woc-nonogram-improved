from functools import reduce
# from itertools import chain
from random import random, randint, uniform

from math import ceil, floor
import os
from PIL import Image, ImageDraw
import numpy as np
import time

# EXAMPLE PUZZLE: PANDA 35X47
ROW_CONSTRAINTS = [
    (1, 3), (1, 2, 2), (3, 3), (5, 1), (5, 2, 6), (11, 2, 4), (10, 7), (9, 3),
    (9, 2), (9, 1), (8, 2), (8, 3, 1), (9, 4, 1), (9, 5, 2), (8, 2, 2, 1),
    (8, 4, 1), (8, 1), (8, 1), (8, 1, 2), (9, 1, 2, 3), (9, 1, 2, 3),
    (9, 1, 2, 3), (11, 1, 2, 2, 3), (12, 3, 2), (1, 6, 1, 2, 3), (1, 2, 4, 6),
    (1, 2, 2, 5, 2), (1, 9), (2, 8, 1), (1, 10), (3, 8, 1), (4, 9), (4, 5, 1),
    (5, 1, 4, 3), (8, 3, 5), (10, 1, 8), (11, 14), (12, 2, 15), (13, 16, 1),
    (16, 12, 1,
     2), (16, 11, 4), (14, 3, 14), (15, 2, 13), (13, 4,
                                                 12), (13, 16), (17, ), (15, )
]
COL_CONSTRAINTS = [
    (5, ), (10, ), (2, 16), (10, 16), (12, 16), (16, 13), (18, 14), (20, 13),
    (23, 12), (1, 23, 12), (24, 11), (2, 22, 10), (1, 2, 9, 3, 9),
    (3, 6, 4, 2, 9), (7, 1, 2, 3), (2, 2, 3, 2, 4, 3), (1, 5, 4, 1, 3, 2, 2),
    (2, 3, 1, 1, 2, 2, 4), (1, 5, 1, 3, 6), (1, 4, 3, 8), (1, 2, 2, 3, 5, 1),
    (1, 4, 18), (2, 2, 9, 9), (2, 10, 9), (1, 9, 10), (2, 9, 11), (2, 8, 12),
    (3, 3, 5, 12), (4, 2, 3, 1,
                    14), (5, 2, 6, 11), (4, 5, 4,
                                         4), (9, 10), (2, 2, 5), (7, ), (1, 3)
]

# EXAMPLE PUZZLE: FACE 10X10
# ROW_CONSTRAINTS = [(3, 3), (2, 4, 2), (1, 1), (1, 2, 2, 1), (1, 1, 1),
#                    (2, 2, 2), (1, 1), (1, 2, 1), (2, 2), (6, )]
# COL_CONSTRAINTS = [(5, ), (2, 4), (1, 1, 2), (2, 1, 1), (1, 2, 1, 1),
#                    (1, 1, 1, 1, 1), (2, 1, 1), (1, 2), (2, 4), (5, )]

# EXAMPLE PUZZLE: TANK 3X3
# ROW_CONSTRAINTS = [(2, ), (2, ), (2, )]
# COL_CONSTRAINTS = [(1, 1), (3, ), (1, )]

# EXAMPLE PUZZLE: TEST 3X5
# ROW_CONSTRAINTS = [(1, ), (1, 1), (2, ), (2, ), (1, 1)]
# COL_CONSTRAINTS = [(2, 1), (2, ), (4, )]

FILLED = 1
EMPTY = 0

# GA & WOC related global parameters
SQUARE_PENALTY = 1
GROUP_PENALTY = 6

POPULATION_SIZE = 3000
GEN_ITERATIONS = 100
REJECTION_PERCENTAGE = .1
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.05
THRESHOLD = 0.7

# in order to use roullete wheel select we need to convert to maximization
# problem. This variable allows for that.
WORST_POSSIBLE_FIT = len(COL_CONSTRAINTS) * (
    SQUARE_PENALTY * len(ROW_CONSTRAINTS) +
    GROUP_PENALTY * ceil(len(ROW_CONSTRAINTS) / 2))


class Nonogram(object):
    """ Represents a n x m Nonogram grid suitable for use in GA algorithm.
    Includes methods for easy fitness calculation, encoding and condensed
    encoding """

    def __init__(self, row_constraints, column_constraints, grid=None):
        """ Return nonogram individual with size of len(nonogram_constraints)
        """
        # self.row_constraints = row_constraints
        # self.column_constraints = column_constraints
        # Note that grid_width constraints corresponds to number of
        # column constraints given, the opposite applies to grid_height
        self.grid_width = len(column_constraints)
        self.grid_height = len(row_constraints)
        if grid is None:
            self.grid = Nonogram.create_rand_grid(row_constraints,
                                                  self.grid_width)
        else:
            self.grid = grid
        self.fitness = Nonogram.calc_fitness(self.grid, column_constraints)

    @staticmethod
    def generate_seg_line(constraints_i, length):
        """ Generates a segmentation line in the condensed encoding based on
        constraints. Constraints can be either i'th row numbers or i'th
        column numbers from Nonogram puzzle problem. Line represent a row in
        the nonogram puzzle. Condensed encoding differs from binary encoding by
        the fact, that it hides information about number of 1 in each segment
        of consecutive 1, and focuses on correct segmentation based on row
        constraints.

        Example: line with length 5 and encoding 10110 is repesented as 1010
        using condensed encoding; 11100->100, 10101->10101 etc. """
        # calculate number of 0's
        zeroes = length - sum(n for n in constraints_i)
        # generate an array of 1's according to number of segments in the
        # given constraint
        string = [1] * len(constraints_i)

        # no consecutive 1's are allowed per our encoding, so insert 0 in
        # between
        for i in range(1, 2 * len(string) - 1, 2):
            string.insert(i, 0)
            zeroes -= 1

        # if there are still 0's left to fill, randomly pick a position
        while zeroes:
            string.insert(randint(0, len(string)), 0)
            zeroes -= 1
        return string

    @staticmethod
    def generate_seg_grid(constraints, size):
        """ Generates a list of segmentation lines according to the
        constraints. If you choose col_constraints make sure to use
        len(row_constraints) for size"""
        return [
            Nonogram.generate_seg_line(constraints[x], size)
            for x in range(0, len(constraints))
        ]

    @staticmethod
    def create_rand_grid(constraints, size):
        """ Creates a Nonogram grid with every row having perfect fitness, and
        random columns"""
        seg_grid = Nonogram.generate_seg_grid(constraints, size)
        # print(seg_grid)
        nonogram_grid = []
        for line, constraint in zip(seg_grid, constraints):
            decoded_line = []
            i = 0
            for element in line:
                if element:
                    decoded_line[len(decoded_line):] = [1] * constraint[i]
                    i += 1
                else:
                    decoded_line.append(0)
            nonogram_grid.append(decoded_line)
        return nonogram_grid

    @staticmethod
    def calc_fitness(grid, constraints):
        """ Returns the fitness score for a particular grid. Since nonogram_grid
        is generated with rows according to the row_constraints, fitness is
        only calculated for columns """

        score = 0

        matrix = np.array(grid)
        # print(matrix)
        # print(matrix.T)
        for index, column in enumerate(matrix.T):
            group_flag = False
            filled_count = 0
            group_count = 0
            column_square_number = 0

            for square in column:
                # Calculate number of groups and number of FILLED squares
                # present in the column
                if square == FILLED:
                    if not group_flag:
                        group_flag = True
                        group_count += 1
                    filled_count += 1
                else:
                    if group_flag:
                        group_flag = False
            for number in constraints[index]:
                column_square_number += number
            # print(
            #     str(filled_count) + " - " + str(column_square_number) + " " +
            #     str(group_count) + " - " + str(
            #         len(self.column_numbers[index])))
            # TODO it will count len((0,)) to be one, needs to be 0
            score += SQUARE_PENALTY * abs(
                filled_count - column_square_number) + GROUP_PENALTY * abs(
                    group_count - len(constraints[index]))

        return score

    def draw_nonogram(self):
        """ Create an PNG format image of grid"""
        image = Image.new("RGB", (self.grid_width * 10, self.grid_height * 10),
                          (255, 255, 255))
        draw = ImageDraw.Draw(image)

        for index, square in enumerate(
                reduce(lambda x, y: x + y, self.grid), 0):

            # print(square)
            x = index % self.grid_width
            y = index // self.grid_width
            coord = [(x * 10, y * 10), ((x + 1) * 10, (y + 1) * 10)]
            if square == EMPTY:
                draw.rectangle(coord, fill=(255, 255, 255))
            if square == FILLED:
                draw.rectangle(coord, fill=(0, 0, 0))
        return image


def draw_population(population, path, filename):
    for index, board in enumerate(population):
        # Draw a picture of each individual in initial population
        image = board.draw_nonogram()
        if not os.path.exists(path):
            os.makedirs(path)
        image.save(path + filename + "_%d.png" % index)
        # print("Board #" + str(index) + " " + str(board.fitness))


def create_population(population_size, row_constraints, col_constraints):
    """Returns a list of nonogram grids"""
    return [
        Nonogram(row_constraints, col_constraints)
        for x in range(0, population_size)
    ]


def roulette_wheel_select(candidates):
    """ Returns an individual from population and its index in a list.
    The chance of being selected is proportional to the individual fitness."""
    # convert to maximization problem
    # to do that I aproximized worst_possbile_fitness to be

    roullete_range = sum(
        WORST_POSSIBLE_FIT - chromosome.fitness for chromosome in candidates)
    roullete_pick = uniform(0, roullete_range)
    current = 0
    for chromosome in candidates:
        current += WORST_POSSIBLE_FIT - chromosome.fitness
        if current > roullete_pick:
            return chromosome


def mate(candidates, row_constraints, col_constraints):
    """ Returns 2 offsprings by mating 2 randomly choosen candidates.
    Make sure pass a copy of a list. """

    # print(candidates)

    # randomly choose 2 candidates, remover selected from candidates to
    # prevent mating 2 identical chromosomes
    candidate1 = roulette_wheel_select(candidates)
    candidates.remove(candidate1)
    candidate2 = roulette_wheel_select(candidates)

    offspring1, offspring2 = single_point_crossover(candidate1.grid,
                                                    candidate2.grid)
    board1 = Nonogram(row_constraints, col_constraints, grid=offspring1)
    board2 = Nonogram(row_constraints, col_constraints, grid=offspring2)
    if board1.fitness < board2.fitness:
        return board1
    else:
        return board2


def single_point_crossover(chromosome1, chromosome2):
    """ Returns 2 chromosomes by randomly swapping genes """
    # print("\nStarting crossover")
    # print(chromosome1, chromosome2)
    chromosome_len = len(chromosome1)
    crossover_point = randint(0, chromosome_len)
    # print(crossover_point)

    offspring1 = chromosome1[0:crossover_point] + chromosome2[crossover_point:
                                                              chromosome_len]
    offspring2 = chromosome2[0:crossover_point] + chromosome1[crossover_point:
                                                              chromosome_len]
    # print("CROSSOVER RESULT")
    # print(offspring1, offspring2)
    return offspring1, offspring2


# def mutation(chromosome):
#     """ Mutation of a gene can occur during mate function, based on
#     probability MUTATION_RATE"""
#
#     n = randint(0, (chromosome.grid_width * chromosome.grid_height) - 1)
#     x, y = n % chromosome.grid_width, n // chromosome.grid_height
#     if chromosome[x][y]:


def population_metrics(boards, generation):
    population = len(boards)
    best = boards[0].fitness
    worst = boards[population - 1].fitness
    average = 0
    median = 0
    buffer = 0
    standard_deviation = 0
    fitnesses = []
    # find average
    for pop_size in range(0, population):
        buffer += boards[pop_size].fitness
        fitnesses.append(boards[pop_size].fitness)
    average = buffer / population
    # calculate median
    if (population % 2 == 0):
        median = (boards[int(population / 2)].fitness +
                  boards[int(population / 2 + 1)].fitness) / 2
    else:
        median = boards[int(population / 2)].fitness
    standard_deviation = np.std(fitnesses, ddof=1)
    # print(standard_deviation)
    file = open('nonogram.log', 'a')
    line_of_text = str(generation) + " " + str(best) + " " + str(
        average) + " " + str(worst) + " " + str(median) + " " + str(
            standard_deviation) + "\n"
    file.write(line_of_text)
    file.close()


def ga_algorithm(population_size, row_constraints, col_constraints):
    """ga algorithm to find a solution for Nonogram puzzle"""

    # Start timer to measure performance
    t0 = time.time()

    population = create_population(population_size, row_constraints,
                                   col_constraints)
    population.sort(key=lambda individual: individual.fitness)
    draw_population(population, 'pics/gen_0/population/', 'nono')
    population_metrics(population, 0)

    for i in range(0, GEN_ITERATIONS):
        # print("Rejecting unfit candidates \n")
        # population = population[0:floor((
        #     1 - REJECTION_PERCENTAGE) * len(population))]
        # path = 'pics/gen_' + str(i) + '/'
        # draw_population(population, path + 'fit_population/', 'fit_nono')

        # Create new chromosomes until reaching POPUlATION_SIZE
        next_gen = []
        # print("GEN", i)
        while len(next_gen) < population_size:
            if random() > CROSSOVER_RATE:
                next_gen.append(
                    mate(population[:], row_constraints, col_constraints))
        # print("Create Adj Matrix\n")
        # adj_matrix = wisdom_of_crowds(population)
        # print(adj_matrix)
        # board = Nonogram()
        # board.grid = wisdom_create_board(adj_matrix, THRESHOLD)
        # board.fitness = Nonogram.calc_fitness(board)
        # next_gen.append(board)
        next_gen.sort(key=lambda individual: individual.fitness)
        # print("Create new board and extend to population")
        # print("NEW POPULATION")
        population_metrics(next_gen, i + 1)
        # path = 'pics/gen_' + str(i + 1) + '/'
        # draw_population(next_gen, path + 'population/', 'nono')
        population = next_gen

    draw_population(next_gen, 'pics/last_gen/population/', 'nono')

    t1 = time.time()
    file = open('nonogram.log', 'a')
    file.write("Running time: " + str(t1 - t0) + "\n")
    file.write("POPULATION_SIZE " + str(POPULATION_SIZE) + "\n")
    file.write("GEN_ITERATIONS " + str(GEN_ITERATIONS) + "\n")
    file.write("REJECTION_RATE " + str(REJECTION_PERCENTAGE) + "\n")
    file.write("MUTATION_RATE " + str(MUTATION_RATE) + "\n")
    file.write("\nSQUARE_PENALTY " + str(SQUARE_PENALTY) + "\n")
    file.write("GROUP_PENALTY " + str(GROUP_PENALTY) + "\n")
    file.write("WISDOM_TRHESHOLD " + str(THRESHOLD) + "\n")
    file.write("END RUN\n\n")
    file.close()


ga_algorithm(POPULATION_SIZE, ROW_CONSTRAINTS, COL_CONSTRAINTS)
