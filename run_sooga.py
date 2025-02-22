from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import IntegerSBXCrossover, IntegerPolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file

from jmetal.core.problem import Problem
from jmetal.util.evaluator import Evaluator, S, List
from jmetal.util.ranking import FastNonDominatedRanking

import logging, csv

from utils.utils import get_initial_population, get_sims, get_street_length, get_travel_time, heal, GraphDataset, get_graph_data, get_emb_graph_data, GNN, get_model

from torch.utils.data import random_split

from karateclub import LDP
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import random_split
import numpy as np

log_file = "output/sooga.logfile"
logging.basicConfig(filename=log_file,
                    filemode='a',
                    # format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=logging.DEBUG)

def pareto_frontier(points):
    # Sort by first dimension (x), then by second dimension (y)
    points = sorted(points, key=lambda x: (x[0], x[1]))

    pareto_optimal = []
    
    for point in points:
        # A point is Pareto optimal if no other point dominates it
        if not any(p[0] <= point[0] and p[1] <= point[1] and p != point for p in points):
            pareto_optimal.append(point)
    
    return pareto_optimal

def log_pareto_front(generation, pareto_front):
    logging.info("Pareto front (Generation " + str(generation) + ")")
    log_full_infos = []
    log_infos = []
    for s in pareto_front:
        print("\t", s)
        logging.info(s)
        log_full_infos.append(s)
        log_infos.append(s)
    log_full_infos.append([])

    with open("output/soo_pareto_fronts_with_graphs.csv", 'a') as file:
        writer = csv.writer(file)
        writer.writerows(log_full_infos)
    
    with open("output/soo_pareto_fronts", 'a') as file:
        writer = csv.writer(file)
        writer.writerow(["Generation " + str(generation)])
        writer.writerows(log_infos) 

class Emb_LEGCS_Evaluator(Evaluator[S]):
    def __init__(self):
        super().__init__()
        self.generation = 0
        self.train_emb = []
        self.train_sim = []
        self.emb_model = LDP()
        self.reg_model = RandomForestRegressor()
    
    def get_embedding(self, genomes):
        graphs = []
        for g in genomes:
            graphs.append(get_emb_graph_data(g))
        self.emb_model.fit(graphs)
        emb = self.emb_model.get_embedding()
        emb_list = []
        for e in emb:
            array = np.array(e)
            emb_list.append(tuple(array))
        return emb_list
        
    
    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        genomes = []
        for s in solution_list:
            genomes.append(s.variables)
        embedding = self.get_embedding(genomes)
       
        if self.generation == 0:
            for s in solution_list:
                self.train_sim.append(s.travel_time)
            self.train_emb = embedding
        else:
            self.reg_model.fit(self.train_emb, self.train_sim)
            predictions = self.reg_model.predict(embedding)            
            genome_counter = 0            
            for solution in solution_list:
                solution.variables = heal(solution.variables)
                solution.simID = "gen" + str(self.generation) + "genome" + str(genome_counter)
                solution.objectives[0] = predictions[genome_counter] + get_street_length(solution.variables)
                genome_counter += 1
                
            ranking = FastNonDominatedRanking()
            ranking.compute_ranking(solution_list)

            sim_list = []
            i = 0
            j = 0
                        
            while len(sim_list) <= 50:
                if len(ranking.ranked_sublists[i]) > j:
                    sim_list.append(ranking.ranked_sublists[i][j].simID)
                    j += 1
                else:
                    i += 1
                    j = 0
            
            genome_counter = 0
            for solution in solution_list:
                if solution.simID in sim_list:
                    street_length = get_street_length(solution.variables)
                    travel_time = get_travel_time(solution.simID, solution.variables)
                    solution.objectives[0] = travel_time + street_length    
                    self.train_emb.append(embedding[genome_counter])
                    self.train_sim.append(solution.objectives[0])
                    sep_solutions.append((travel_time, street_length))
                    logging.info(solution.simID + ": fitness = " + str(solution.objectives))
                genome_counter += 1
        
            pareto_front = pareto_frontier(sep_solutions)
            log_pareto_front(self.generation, pareto_front)

        self.generation += 1
        return solution_list            

class LEGCS_Evaluator(Evaluator[S]):
    def __init__(self):
        super().__init__()
        self.generation = 0
        self.dataset = GraphDataset([], [])
        self.model = None

    def train_gnn(self):       
        gnn_options = []
        for _ in range(3):
            train_set, val_set = random_split(self.dataset, [0.8, 0.2])
            model, val_mae, val_r2 = get_model(GNN(), train_set, val_set)
            gnn_options.append((model, val_mae, val_r2))
        
        #self.model = model
        max_r2 = 0
        for (model, mae, r2) in gnn_options:
            if r2 > max_r2:
                self.model = model
                max_r2 = r2
                val_mae = mae
                val_r2 = r2
        
        logging.info("new GNN model trained, mae = " + str(val_mae) + ", r2 = " + str(val_r2))

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        if self.generation % 10 == 0 and self.generation > 0:
            self.train_gnn()
       
        if self.generation == 0:
            for solution in solution_list:
                self.dataset.append(get_graph_data(solution.variables), solution.travel_time)
            self.train_gnn()
        else:
            genome_counter = 0
            for solution in solution_list:
                solution.variables = heal(solution.variables)
                solution.simID = "gen" + str(self.generation) + "genome" + str(genome_counter)
                self.model.eval()
                travel_time = self.model(get_graph_data(solution.variables)).item()
                street_length = get_street_length(solution.variables)
                solution.objectives[0] = travel_time + street_length
                genome_counter += 1
                
            ranking = FastNonDominatedRanking()
            ranking.compute_ranking(solution_list)

            sim_list = []
            i = 0
            j = 0
                        
            while len(sim_list) <= 50:
                if len(ranking.ranked_sublists[i]) > j:
                    sim_list.append(ranking.ranked_sublists[i][j].simID)
                    j += 1
                else:
                    i += 1
                    j = 0
            
            sep_solutions = []
            for solution in solution_list:
                if solution.simID in sim_list:
                    street_length = get_street_length(solution.variables)
                    travel_time = get_travel_time(solution.simID, solution.variables)
                    solution.objectives[0] = travel_time + street_length    
                    self.dataset.append(get_graph_data(solution.variables), travel_time)
                    sep_solutions.append((travel_time, street_length))
                    logging.info(solution.simID + ": fitness = " + str(solution.objectives))
        
            pareto_front = pareto_frontier(sep_solutions)
            log_pareto_front(self.generation, pareto_front)

        self.generation += 1
        return solution_list


class SUMO_Evaluator(Evaluator[S]):
    def __init__(self):
        super().__init__()
        self.generation = 0

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:        
        if self.generation > 0:
            logging.info("starting generation " + str(self.generation))        
            genome_counter = 0
            
            sep_solutions = []
            for solution in solution_list:
                solution.variables = heal(solution.variables)
                genome = solution.variables
                solution.simID = "gen" + str(self.generation) + "genome" + str(genome_counter)
                travel_time = get_travel_time(solution.simID, genome)
                street_length = get_street_length(genome)
                solution.objectives[0] = travel_time + street_length
                sep_solutions.append((travel_time, street_length))
                genome_counter += 1
                logging.info(solution.simID + ": fitness = " + str(solution.objectives))
                
            pareto_front = pareto_frontier(sep_solutions)
            log_pareto_front(self.generation, pareto_front)

        self.generation += 1
        return solution_list

# Define a Custom Multi-Objective Problem
class SOO_Traffic_Network_Problem(IntegerProblem):
    def __init__(self):
        super().__init__()
        self.number_of_variables = 32**2  
        self.number_of_objectives = 1  # Two objectives to minimize
        self.number_of_constraints = 0  

        self.lower_bound = [0] * self.number_of_variables # Lower bounds (Integers)
        self.upper_bound = [3] * self.number_of_variables # Upper bounds (Integers)

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        pass # using Evaluator instead

    def create_solution(self) -> IntegerSolution:
        pass # using create_initial_solutions instead

    def name(self):
        return "SOO_Traffic_Network_Problem"

    def number_of_objectives(self):
        return self.number_of_objectives

    def number_of_constraints(self):
        return self.number_of_constraints

class TNP_GA(GeneticAlgorithm):
    def create_initial_solutions(self):
        
        initial_population = get_initial_population(range(offset, offset+population_size))
        sims = get_sims()[offset:offset+population_size]
        
        solutions = []
        for i in range(population_size):
            new_solution = IntegerSolution([0] * num_variables, [5] * num_variables, 1, 0)
            new_solution.variables = initial_population[i]
            travel_time = sims[i]
            street_length = get_street_length(initial_population[i])
            new_solution.objectives[0] = travel_time + street_length
            new_solution.travel_time = travel_time
            new_solution.simID = "gen0genome" + str(i)
            solutions.append(new_solution)
        return solutions
            
offset = 2500
population_size = 500
num_towns = 32
num_variables = num_towns**2

patience = 0
iterator = 0
runs = 3

evaluator = SUMO_Evaluator()
eval_counter = 5500
#evaluator = LEGCS_Evaluator()
#eval_counter = 50500


while iterator < runs:

    try:

        # Instantiate the problem
        problem = SOO_Traffic_Network_Problem()
        

        # Configure NSGA-II Algorithm
        algorithm = TNP_GA(
            problem=problem,
            population_size=population_size,
            offspring_population_size=population_size,
            mutation=IntegerPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            crossover=IntegerSBXCrossover(probability=0.9),  # Single Point Crossover
            termination_criterion=StoppingByEvaluations(max_evaluations=eval_counter),
            population_evaluator=SUMO_Evaluator(),
        )

        logging.info("starting TNP_GA with " + type(evaluator).__name__ + " with population size = " + str(population_size) + " and offset = " + str(offset))

        # Run the Algorithm
        algorithm.run()

        # Get and Print Results
        solution = algorithm.result()

        print("Best Solution:")
        for solution in non_dominated_solutions:
            print(sum(solution.variables), "->", solution.objectives[0])

        # Save Results to File
        #print_function_values_to_file(non_dominated_solutions, "pareto_front.txt")
    
        offset += population_size
        iterator += 1
        patience = 0

    except Exception as e: 
        print("EXCEPTION! ", e)
        patience += 1
       
        if patience == 3:
            offset += population_size
            iterator += 1
            patience = 0
