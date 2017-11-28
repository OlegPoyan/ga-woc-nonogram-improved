from functools import reduce
# from itertools import chain
from random import random, randint, uniform

from math import ceil
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
UNDECIDED = 2

# Designates possible modes of GA algorithm
ROW_MODE = 1
COL_MODE = 2
RAND_MODE = 0
SCATER_MODE = (10, 1)
GROUP_MODE = (1, 10)
# Use this mode only outside GA algorithms
EVAL_MODE = (1, 1)

# GA & WOC related global parameters
POPULATION_SIZE = 100
GEN_ITERATIONS = 30
EXPERT_THRESHOLD = .1
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.05
WOAC_UPPER_THRESHOLD = 0.5
WOAC_LOWER_THRESHOLD = 0.5

mode_dict_g = {
    RAND_MODE: "RAND_MODE",
    ROW_MODE: "ROW_MODE",
    COL_MODE: "COL_MODE",
    GROUP_MODE: "GROUP_MODE",
    SCATER_MODE: "SCATER_MODE"
}


class Nonogram(object):
    """ Represents a n x m Nonogram grid suitable for use in GA algorithm.
    Includes methods for easy fitness calculation, encoding and condensed
    encoding. Based on the mode, grid can be either list of rows, or list of
    columns."""

    def __init__(self,
                 row_constraints,
                 col_constraints,
                 mode,
                 weights,
                 grid=None):
        """ Return nonogram individual with size of len(nonogram_constraints)

        If grid is not supplied, generates grid with either row's segmentation
        solved when row_flag=True, or columns sovled when row_flag=False

        Note: if in COL_MODE grid will be generated in the form of list of
        rows. The opposite will hapen if ROW_MODE. Not a bug, but a feature."""
        # Note that grid_width constraints corresponds to number of
        # column constraints given, the opposite applies to grid_height
        self.grid_width = len(col_constraints)
        self.grid_height = len(row_constraints)
        self.row_constraints = row_constraints
        self.col_constraints = col_constraints
        self.weights = weights
        if grid is None:
            if mode is ROW_MODE:
                self.grid = Nonogram.create_bias_rand_grid(
                    row_constraints, self.grid_width)
                self.mode = mode
            elif mode is COL_MODE:
                self.grid = Nonogram.create_bias_rand_grid(
                    col_constraints, self.grid_height)
                self.mode = mode
            else:
                self.grid = Nonogram.create_rand_grid(self.grid_width,
                                                      self.grid_height)
                self.mode = mode
        else:
            self.grid = grid
            self.mode = mode
        self.fitness = Nonogram.calc_fitness(
            self.grid, row_constraints, col_constraints, self.mode, weights)

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
    def create_bias_rand_grid(constraints, size):
        """ Creates a randomly filled grid with either rows, or columns
        solved"""
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
    def create_rand_grid(grid_width, grid_height):
        """Returns two dimensional array with squares in the grid filled in
        randomly."""

        return [[randint(0, 1) for x in range(0, grid_width)]
                for y in range(0, grid_height)]

    @staticmethod
    def calc_bias_fitness(grid, constraints, weights):
        """Returns total fitnes score of just one dimension."""
        SQUARE_PENALTY = weights[0]
        GROUP_PENALTY = weights[1]

        score = 0
        for index, row in enumerate(grid):
            group_flag = False
            filled_count = 0
            group_count = 0
            row_square_number = 0

            for square in row:
                # Calculate number of groups and number of FILLED squares
                # present in the row
                if square == FILLED:
                    if not group_flag:
                        group_flag = True
                        group_count += 1
                    filled_count += 1
                else:
                    if group_flag:
                        group_flag = False

            for number in constraints[index]:
                row_square_number += number
            # print(
            #     str(filled_count) + " - " + str(row_square_number) + " " +
            #     str(group_count) + " - " + str(len(self.row_numbers[index])))
            # TODO it will count len((0,)) to be one, needs to be 0
            score += SQUARE_PENALTY * abs(
                filled_count - row_square_number) + GROUP_PENALTY * abs(
                    group_count - len(constraints[index]))

        return score

    @staticmethod
    def calc_fitness(grid, row_constraints, col_constraints, mode, weights):
        """ Returns the fitness score for a particular grid

        Depending on the mode, calculates either row_fitness, col_fitness, or
        both."""
        # if grid was generated in biased mode, only calculate fitnes of one
        # dimension
        if (mode is ROW_MODE):
            return Nonogram.calc_bias_fitness(
                list(map(list, zip(*grid))), col_constraints, weights)
        elif (mode is COL_MODE):
            return Nonogram.calc_bias_fitness(
                list(map(list, zip(*grid))), row_constraints, weights)
        else:
            # in a case when mode=None, we can use calc_bias_fitness() to
            # calculate both col_fitness and row_fitnes of a grid
            # make sure to to supply transpose of grid when calculating
            # col_fitness
            return Nonogram.calc_bias_fitness(
                # transpose of a grid
                list(map(list, zip(*grid))),
                col_constraints,
                weights) + Nonogram.calc_bias_fitness(grid, row_constraints,
                                                      weights)

    def draw_nonogram(self):
        """ Create an PNG format image of grid"""
        image = Image.new("RGB", (self.grid_width * 10, self.grid_height * 10),
                          (255, 255, 255))
        draw = ImageDraw.Draw(image)

        # transpose of that matrix to get it in list of rows form
        if self.mode is COL_MODE:
            grid = list(map(list, zip(*self.grid)))
        else:
            grid = self.grid

        for index, square in enumerate(reduce(lambda x, y: x + y, grid), 0):

            # print(square)
            x = index % self.grid_width
            y = index // self.grid_width
            coord = [(x * 10, y * 10), ((x + 1) * 10, (y + 1) * 10)]
            if square == EMPTY:
                draw.rectangle(coord, fill=(255, 255, 255))
            elif square == FILLED:
                draw.rectangle(coord, fill=(0, 0, 0))
            elif square == UNDECIDED:
                draw.rectangle(coord, fill=(255, 102, 153))
        return image


def draw_population(population, path, filename):
    for index, board in enumerate(population):
        # Draw a picture of each individual in initial population
        image = board.draw_nonogram()
        if not os.path.exists(path):
            os.makedirs(path)
        image.save(path + filename + "_%d.png" % index)
        # print("Board #" + str(index) + " " + str(board.fitness))


def create_population(population_size, row_constraints, col_constraints, mode,
                      weights):
    """Returns a list of nonogram grids"""
    return [
        Nonogram(row_constraints, col_constraints, mode, weights)
        for x in range(0, population_size)
    ]


def roulette_wheel_select(candidates, worst_possible_fit):
    """ Returns an individual from population and its index in a list.
    The chance of being selected is proportional to the individual fitness."""
    # convert to maximization problem
    # to do that I aproximized worst_possbile_fitness to be

    roullete_range = sum(
        worst_possible_fit - chromosome.fitness for chromosome in candidates)
    roullete_pick = uniform(0, roullete_range)
    current = 0
    for chromosome in candidates:
        current += worst_possible_fit - chromosome.fitness
        if current > roullete_pick:
            return chromosome


def mate(candidates, mode, worst_possible_fit):
    """ Returns 2 offsprings by mating 2 randomly choosen candidates.
    Make sure pass a copy of a list. """

    # randomly choose 2 candidates, remover selected from candidates to
    # prevent mating 2 identical chromosomes
    candidate1 = roulette_wheel_select(candidates, worst_possible_fit)
    row_constraints = candidate1.row_constraints
    col_constraints = candidate1.col_constraints
    weights = candidate1.weights
    candidates.remove(candidate1)
    candidate2 = roulette_wheel_select(candidates, worst_possible_fit)

    offspring1, offspring2 = single_point_crossover(candidate1.grid,
                                                    candidate2.grid)
    board1 = Nonogram(
        row_constraints, col_constraints, mode, weights, grid=offspring1)
    board2 = Nonogram(
        row_constraints, col_constraints, mode, weights, grid=offspring2)
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


def ga_algorithm(population_size, row_constraints, col_constraints, mode,
                 run_name, weights):
    """ga algorithm to find a solution for Nonogram puzzle. Depending on the
    mode selected generates populations either with solved rows, column, or
    generates random grids if mode=None"""

    file = open('nonogram.log', 'a')
    file.write("START GA RUN\nGA mode: " + mode_dict_g[mode] + " " +
               mode_dict_g[weights] + "\n")
    file.close()
    t0 = time.time()

    population = create_population(population_size, row_constraints,
                                   col_constraints, mode, weights)

    # in order to use roullete wheel select we need to convert to maximization
    # problem. This variable allows for that.
    worst_possible_fit = (len(col_constraints) *
                          (weights[0] * len(row_constraints) +
                           weights[1] * ceil(len(row_constraints) / 2)) +
                          (len(row_constraints) *
                           (weights[0] * len(col_constraints) +
                            weights[1] * ceil(len(col_constraints) / 2))))

    population.sort(key=lambda individual: individual.fitness)
    draw_population(population, 'pics/' + run_name + '/gen_0/population/',
                    'nono')
    population_metrics(population, 0)

    for i in range(0, GEN_ITERATIONS):
        # path = 'pics/gen_' + str(i) + '/'
        # draw_population(population, path + 'fit_population/', 'fit_nono')

        # Create new chromosomes until reaching POPUlATION_SIZE
        next_gen = []
        # print("GEN", i)
        while len(next_gen) < len(population):
            if random() > CROSSOVER_RATE:
                next_gen.append(mate(population[:], mode, worst_possible_fit))

        next_gen.sort(key=lambda individual: individual.fitness)
        population_metrics(next_gen, i + 1)
        population = next_gen

    t1 = time.time()
    file = open('nonogram.log', 'a')
    file.write("Running time: " + str(t1 - t0) + "\n")
    file.close()

    draw_population(next_gen, 'pics/' + run_name + '/last_gen/population/',
                    'nono')
    return population[:int(EXPERT_THRESHOLD * (len(population)))]

def ga_mutate(population, rate):
    """ Mutation function. Has a chance to mutate each row or column of each board in the population
    depending on the mode being used"""
    for i in range(0, len(population)):
        # means we are in ROW_MODE
        if population[i].mode == ROW_MODE:
            mutate_row(population[i], rate)

        # means we are in COL_MODE
        elif population[i].mode == COL_MODE:
            mutate_col(population[i], rate)

        # means we are in RAND_MODE
        else:
            mutate_row(population[i], rate)



def mutate_col(board, rate):
    """Function to mutate a board while in COLUMN_MODE"""
    for i in range(0, len(board.col_constraints)):
        chance = float(randint(0, 1000))
        chance /= 1000
        if (chance <= rate):
            # mutate a column
            # encodes into a smaller format
            encode = []
            groupFlag = False
            for j in range(0, len(board.row_constraints)):
                if groupFlag == False:
                    if board.grid[j][i] == 1:
                        groupFlag = True
                        encode.append(1)
                    elif board.grid[j][i] == 0:
                        encode.append(0)
                else:
                    if board.grid[j][i] == 0:
                        groupFlag = False
                        encode.append(0)
            # select random position to swap
            pos1 = randint(0, len(encode))
            pos2 = randint(0, len(encode))
            while pos1 == pos2:
                pos2 = randint(0, len(encode))

            # swap values
            temp = encode[pos1]
            encode[pos1] = encode[pos2]
            encode[pos2] = temp

            # check to make sure no rules are violated
            for k in encode:
                groupFlag2 = False
                if groupFlag2 == False:
                    if k == 1:
                        groupFlag2 == True

                else:
                    if k == 1:
                        # violation has occurred, two groups without separating white square
                        return
                    else:
                        groupFlag2 = False

            # decode
            board_column = []
            current_group = 0
            for l in encode:
                if l == 0:
                    board_column.append(0)
                else:
                    for i in range(0, board.col_constraints[i][current_group]):
                        board_column.append(1)
                        current_group += 1

            for m in range(0, len(board.row_constraints)):
                board.grid[m][i] = board_column[m]

            # recalculate fitness
            board.calc_fitness(board.grid, board.row_constraints, board.col_constraints, board.mode, board.weights)



def mutate_row(board, rate):
    """Function to mutate a board while in ROW_MODE, Also used for RAND_MODE"""
    for i in range(0, len(board.row_constraints)):
        chance = float(randint(0, 1000))
        chance /= 1000
        if (chance <= rate):
            # mutate a column
            # encodes into a smaller format
            encode = []
            groupFlag = False
            for j in range(0, len(board.col_constraints)):
                if groupFlag == False:
                    if board.grid[i][j] == 1:
                        groupFlag = True
                        encode.append(1)
                    elif board.grid[i][j] == 0:
                        encode.append(0)
                else:
                    if board.grid[i][j] == 0:
                        groupFlag = False
                        encode.append(0)
            # select random position to swap
            pos1 = randint(0, len(encode))
            pos2 = randint(0, len(encode))
            while pos1 == pos2:
                pos2 = randint(0, len(encode))

            # swap values
            temp = encode[pos1]
            encode[pos1] = encode[pos2]
            encode[pos2] = temp

            # check to make sure no rules are violated
            for k in encode:
                groupFlag2 = False
                if groupFlag2 == False:
                    if k == 1:
                        groupFlag2 == True

                else:
                    if k == 1:
                        # violation has occurred, two groups without separating white square
                        return
                    else:
                        groupFlag2 = False

            # decode
            board_row = []
            current_group = 0
            for l in encode:
                if l == 0:
                    board_row.append(0)
                else:
                    for i in range(0, board.row_constraints[i][current_group]):
                        board_row.append(1)
                        current_group += 1

            for m in range(0, len(board.col_constraints)):
                board.grid[i][m] = board_row[m]

            # recalculate fitness
            board.calc_fitness(board.grid, board.row_constraints, board.col_constraints, board.mode, board.weights)

def woac_aggregate(crowd):
    """Returns grid filled with percentages of squares crowd aggrees or not
    agrees on"""
    row_constraints = crowd[0].row_constraints
    col_constraints = crowd[0].col_constraints
    arr = np.array([
        list(map(list, zip(*solution.grid)))
        if solution.mode is COL_MODE else solution.grid for solution in crowd
    ])
    adj_grid = np.mean(arr, axis=0)
    print(adj_grid)
    new_solution = [[
        FILLED if x > WOAC_UPPER_THRESHOLD else EMPTY
        if x < WOAC_LOWER_THRESHOLD else UNDECIDED for x in row
    ] for row in adj_grid]
    return Nonogram(
        row_constraints,
        col_constraints,
        RAND_MODE,
        EVAL_MODE,
        grid=new_solution)


def ga_woac(population_size, row_constraints, col_constraints):
    """ Algorithm that combines Genetic Algorithm and Wisdom of Crowds to find
    best solution for a nonogram puzzle

    Calls ga_algorithm() n times to create n different solutions then calls woc
    to aggreate solutions.

    Warning: this method alters GROUP_PENALTY and SQUARE_PENALTY global
    variables"""

    # Start timer to measure performance
    t0 = time.time()

    solutions = []
    for i in range(5):

        solutions.extend(
            ga_algorithm(population_size, row_constraints, col_constraints,
                         ROW_MODE,
                         str(i) + '_ga_row_group', GROUP_MODE))
        solutions.extend(
            ga_algorithm(population_size, row_constraints, col_constraints,
                         ROW_MODE,
                         str(i) + '_ga_row_scater', SCATER_MODE))
        solutions.extend(
            ga_algorithm(population_size, row_constraints, col_constraints,
                         COL_MODE,
                         str(i) + '_ga_col_group', GROUP_MODE))
        solutions.extend(
            ga_algorithm(population_size, row_constraints, col_constraints,
                         COL_MODE,
                         str(i) + '_ga_col_scater', SCATER_MODE))
        # solutions.extend(
        #     ga_algorithm(population_size, row_constraints, col_constraints,
        #                  RAND_MODE,
        #                  str(i) + '_ga_rand_group', GROUP_MODE))
        # solutions.extend(
        #     ga_algorithm(population_size, row_constraints, col_constraints,
        #                  RAND_MODE,
        #                  str(i) + '_ga_rand_scater', SCATER_MODE))

    file = open('nonogram.log', 'a')
    for i, board in enumerate(solutions):
        board.fitnes = Nonogram.calc_fitness(
            board.grid, row_constraints, col_constraints, board.mode, (1, 1))
        file.write("BOARD " + str(i) + " " + mode_dict_g[board.mode] + " " +
                   str(board.fitnes) + '\n')

    adj_nonogram = woac_aggregate(solutions)
    adj_nonogram.draw_nonogram().save("solution.png")
    print("ADJ nonogram fitness: ", adj_nonogram.fitness)

    t1 = time.time()
    file.write("TOTAL Running time: " + str(t1 - t0) + "\n")
    file.write("POPULATION_SIZE " + str(POPULATION_SIZE) + "\n")
    file.write("CROSSOVER_RATE " + str(CROSSOVER_RATE) + "\n")
    file.write("GEN_ITERATIONS " + str(GEN_ITERATIONS) + "\n")
    file.write("EXPERT_THRESHOLD " + str(EXPERT_THRESHOLD) + "\n")
    file.write("MUTATION_RATE " + str(MUTATION_RATE) + "\n")
    file.write("WOAC_UPPER_TRHESHOLD " + str(WOAC_UPPER_THRESHOLD) + "\n")
    file.write("WOAC_LOWER_TRHESHOLD " + str(WOAC_LOWER_THRESHOLD) + "\n")
    file.write("<----------END RUN---------->\n\n")
    file.close()


ga_woac(POPULATION_SIZE, ROW_CONSTRAINTS, COL_CONSTRAINTS)
