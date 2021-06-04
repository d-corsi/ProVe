import random, numpy, time, math, gc
from support.tools import *

class ProVe:
    """
    ProVe:
    ----------
    A tool for the safety evaluation of ANNs, evaluating different safety properties,
	
    """

    def __init__(self, input_area, neural_network, condition_a, condition_b, 
                    property_mode = 0,
                    analysis_mode = None, 
                    semi_formal_prec = None,
                    heuristic = None, 
                    round_eps = None, 
                    cpu_memory_pool = None, 
                    gpu_memory_pool = None, 
                    input_means = None, 
                    input_ranges = None,
                    basic_iteration = None
                ):

        # Input Variables
        self.node_number = len(input_area)
        self.heu = heuristic
        self.eps = round_eps
        self.net = neural_network
        self.cpu_mem_pool = cpu_memory_pool
        self.gpu_mem_pool = gpu_memory_pool
        self.analysis_mode = analysis_mode
        self.semi_formal_prec = semi_formal_prec
        self.condition_a = condition_a
        self.condition_b = condition_b
        self.property_mode = property_mode
        self.input_means = input_means
        self.input_ranges = input_ranges
        self.basic_iteration = basic_iteration

        # Fix "None" Values
        if self.cpu_mem_pool == None: self.cpu_mem_pool = 3.2e+10
        if self.gpu_mem_pool == None: self.gpu_mem_pool = 4.0e+9
        if self.input_means == None: self.input_means = [0.5 for _ in range(self.node_number)]
        if self.input_ranges == None: self.input_ranges = [1 for _ in range(self.node_number)]
        if self.eps == None: self.eps = [1 for _ in range(self.node_number)]
        if self.heu == None: self.heu = 'biggest_first'
        if self.analysis_mode == None: self.analysis_mode = 'formal'
        if self.semi_formal_prec == None: self.semi_formal_prec = 10
        if self.basic_iteration == None: self.basic_iteration = 0


        # Private Variables
        self.areas_matrix = numpy.array([sum(input_area, [])], dtype="float32")
        self.mul_matrix = generate_hypermatrix(len(input_area)) 
        self.current_area_size = 100

        # Round of the basic matrix (initial)
        #for i in range(0, self.node_number*2, 2):
        #    self.areas_matrix[:, i] = numpy.trunc(self.areas_matrix[:, i]*10**self.eps[int(i/2)])/(10**self.eps[int(i/2)])
        #    self.areas_matrix[:, i+1] = numpy.trunc(self.areas_matrix[:, i+1]*10**self.eps[int(i/2)])/(10**self.eps[int(i/2)])

        # Normalization
        self.__normalize_input_area()

        # Round of the basic matrix (normalized)
        for i in range(0, self.node_number*2, 2):
            self.areas_matrix[:, i] = numpy.round(self.areas_matrix[:, i], self.eps[int(i/2)])
            self.areas_matrix[:, i+1] = numpy.round(self.areas_matrix[:, i+1], self.eps[int(i/2)])

        # Splitting variables
        if(self.analysis_mode == 'formal'): self.influence = ProVe.get_influence(self.net)
        
    ########################
    ###### MAIN LOOP #######
    ########################

    def main_loop(self, verbose=0):
        
        time_start = time.time()
        total_violation_rate = 0
        total_safe_rate = 0

        cycle_counter = 0

        for _ in range(self.basic_iteration): self.split()

        while not self.memory_empty():
            _ = self.split()
            output_bounds, _ = self.generate_output_bound()
            safe_indexes, violation_indexes, _, _ = self.verify_property(output_bounds)

            total_violation_rate += self.get_area_size(violation_indexes)
            total_safe_rate += self.get_area_size(safe_indexes)
            _ = self.remove_indexes(safe_indexes + violation_indexes)

            if(verbose > 0):
                print("Cycle: +", cycle_counter)
                print("Violation Rate:", total_violation_rate)
                print("Safe Rate:", total_safe_rate)
                print("Analized Area:", 100 - self.current_area_size)
                print()
                cycle_counter += 1

        return total_violation_rate, (time.time() - time_start)

    ########################
    ### PUBLIC METHODS #####
    ########################

    def split(self):
        time_start = time.time()
        node_2_cut = self.choose_node()

        res = (self.areas_matrix.dot(self.mul_matrix[node_2_cut]))
        self.areas_matrix = res.reshape((len(res) * 2, self.node_number * 2)) ;del res; gc.collect()
        for i in range(0, self.node_number*2, 2):
            self.areas_matrix[:, i] = numpy.round(self.areas_matrix[:, i], self.eps[int(i/2)]) ;gc.collect()
            self.areas_matrix[:, i+1] = numpy.round(self.areas_matrix[:, i+1], self.eps[int(i/2)]) ;gc.collect()
        self.areas_matrix = numpy.unique(self.areas_matrix, axis=0) ;gc.collect()

        return (time.time() - time_start)


    def remove_indexes(self, indexes):
        self.current_area_size -= (self.current_area_size / len(self.areas_matrix) * len(indexes)) # Fix Area Size

        time_start = time.time()
        self.areas_matrix = numpy.delete(self.areas_matrix, indexes, axis=0) ;gc.collect() 

        return (time.time() - time_start)

    def generate_output_bound(self):
        time_start = time.time()

        if self.analysis_mode == 'formal': 
            final_bound_gpu, _ = get_output_bound_propagation(self.get_matrix_propagation(), self.net, self.gpu_mem_pool, verbose=0)
        elif self.analysis_mode == 'semiformal':
            final_bound_gpu = get_output_bound_estimation(self.get_matrix_propagation(), self.net, precision=self.semi_formal_prec)
        elif self.analysis_mode == "informal":
            raise NotImplementedError
        else:
            raise SyntaxError("Invalid analysis mode: '{}' (options: [formal, semiformal, informal]).".format(self.analysis_mode))
        
        return final_bound_gpu, (time.time() - time_start)

    def verify_property(self, output_bounds):
        time_start = time.time()
        safe_indexes, violation_indexes, unkn_indexes = verify_property_sequential(output_bounds, self.condition_a, self.condition_b, self.property_mode)
        return safe_indexes, violation_indexes, unkn_indexes, (time.time() - time_start)

    def memory_empty(self):
        return (self.areas_matrix.shape[0] == 0)

    ########################
    ### PRIVATE METHODS ####
    ########################

    def __normalize_input_area(self):
        input_ranges = sum([[a, a] for a in self.input_ranges], [])
        input_means = sum([[a, a] for a in self.input_means], [])

        self.areas_matrix = numpy.add(numpy.subtract(self.areas_matrix, input_means) / input_ranges, 0.5) #questo 0.5 è perchè la normalizza tra -0.5 e 0.5, per riprotare tra [0 1] risommo 0.5

    ########################
    ##### GET METHODS ######
    ########################

    def get_matrix_propagation(self):
        input_ranges = sum([[a, a] for a in self.input_ranges], [])
        input_means = sum([[a, a] for a in self.input_means], [])
        
        matrix_to_return = (numpy.add(numpy.subtract(self.areas_matrix, 0.5) * input_ranges, input_means))
        return matrix_to_return.reshape((matrix_to_return.shape[0], self.node_number, 2))           

    def get_total_entries(self):
        return self.areas_matrix.shape[0]

    def get_area_size(self, indexes):
        return (self.current_area_size / len(self.areas_matrix) * len(indexes))
        
    ########################
    ### SPLITTING METHODS ##
    ########################

    # Matrix is only the first row
    def choose_node(self):
        first_row = self.areas_matrix[0]

        distances = []
        closed = []
        for index, el in enumerate(first_row.reshape(self.node_number, 2)):
            distance = el[1] - el[0]
            if(distance == 0): closed.append(index)
            distances.append(el[1] - el[0])

        if(self.heu == "random"): return self.random_split(closed)
        if(self.heu == "biggest_first"): return self.biggest_first_split(distances)
        if(self.heu == "best_first"): return self.best_first_split(closed)
        if(self.heu == "hybrid"): return self.hybrid_split(distances, closed)
        else: raise ValueError("Invalid heuristic : '{}' (options: [random, biggest_first, best_first, hybrid]).".format(self.heu))

    def random_split(self, closed):
        array = [item for item in range(0, self.node_number) if item not in closed]
        return random.sample(array, 1)[0]
    
    def biggest_first_split(self, distances):  
        return distances.index(max(distances))

    def best_first_split(self, closed):
        self.influence[closed] = -1
        return numpy.argmax(self.influence)

    def hybrid_split(self, distance, closed):
        mod_a = 0.7 #multiplier for distances
        mod_b = 0.3 #multiplier for influence
        assert ((mod_a + mod_b) == 1)

        norm_distance = numpy.array(distance) / sum(distance)
        summed = numpy.sum([norm_distance * mod_a, self.influence * mod_b], axis=0) / (mod_a + mod_b)
        summed[closed] = 0
        probability = (summed / sum(summed)) # normalized

        return numpy.random.choice(self.node_number, p=probability)

    @staticmethod
    def get_influence(net):
        weigths = net.layers[0].weights[0].numpy()
        weigths_absolute = numpy.absolute(weigths)
        weigths_summed = numpy.sum(weigths_absolute, axis=1)
        weigths_normalized = weigths_summed / sum(weigths_summed)

        return (numpy.around(weigths_normalized, 2))
