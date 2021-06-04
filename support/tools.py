import numpy as np
from support.cuda_eval import get_output_bound_gpu

def get_output_bound_propagation(unverified_minibound, net, mem_pool, verbose=0):

    final_bound_gpu = np.array([], dtype="float32")
    total_time = 0

    memory_estimation_bit = (len(unverified_minibound) * 2 * net.input.shape[1] * 32)
    split_number = int((memory_estimation_bit / mem_pool)) + 1
    split_size = int(len(unverified_minibound) / split_number)
    split_rest = len(unverified_minibound) - (split_size * split_number)
    iteration_number = split_number if split_rest == 0 else (split_number + 1)

    if(verbose > 1):
        print("\tMemory Estimation: {} bit, {} split needed, {} iteration needed".format(memory_estimation_bit, split_number, iteration_number))

    for i in range(iteration_number):
        current_size = split_size if i < split_number else split_rest
        split_start = i * split_size
        split_end = (split_start + current_size)

        partial_bound_gpu, time = get_output_bound_gpu(net, unverified_minibound[split_start:split_end])

        final_bound_gpu = np.append(final_bound_gpu, partial_bound_gpu)
        total_time += time

        if(verbose > 2):
            print("\tCycle ({}) completed, time: {:0.2f}s". format(i+1, time))

    return final_bound_gpu.reshape(len(unverified_minibound), net.layers[-1].output_shape[1], 2), total_time


def verify_property_sequential(output_bounds, a, b, mode):
    unkn_indexes = []
    violation_indexes = []
    safe_indexes = []

    for index, bound in enumerate(output_bounds):
        truth_table = get_truth_table(bound, a, b)
        
        if(mode == 0):
            true_table = [node.count(True) > 0 for node in truth_table]
            false_table = [node.count(False) == len(b) for node in truth_table]

            if(true_table.count(True) == len(a)): safe_indexes.append(index)
            elif(false_table.count(True) > 0): violation_indexes.append(index)
            else: unkn_indexes.append(index)

        else:
            true_table = [node.count(True) == len(b) for node in truth_table]
            false_table = [node.count(False) > 0 for node in truth_table]

            if(true_table.count(True) > 0): safe_indexes.append(index)
            elif(false_table.count(True) == len(a)): violation_indexes.append(index)
            else: unkn_indexes.append(index)

    return safe_indexes, violation_indexes, unkn_indexes


def get_truth_table(bound, a, b):
    truth_table = []
    for left in a:
        truth_table.append([])
        for right in b:
            if bound[left][1] < bound[right][0]: truth_table[-1].append(True)
            elif bound[left][0] >= bound[right][1]: truth_table[-1].append(False)
            else: truth_table[-1].append(None)
    return truth_table
       

def generate_hypermatrix(node_number):
    n = (node_number * 2)

    mul_matrix = []
    for i in range(node_number):
        mul_matrix_base = np.concatenate((np.identity(n, dtype="float32"), np.identity(n, dtype="float32")), 1)

        node_2_cut = i

        mul_matrix_base[(node_2_cut * 2) + 0][(node_2_cut * 2) + 1] = 0.5
        mul_matrix_base[(node_2_cut * 2) + 1][(node_2_cut * 2) + 1] = (0.5 - 0.00001) # numero di cifre decimale almeno una cifra superiore di eps, seve una cosa un po' minore di eps

        mul_matrix_base[(node_2_cut * 2) + 0][n + (node_2_cut * 2)] = 0.5 
        mul_matrix_base[(node_2_cut * 2) + 1][n + (node_2_cut * 2)] = (0.5 + 0.00001)

        mul_matrix.append(mul_matrix_base)

    return mul_matrix


def get_output_bound_estimation(areas_matrix, network, precision=100):
    input_nodes = len(areas_matrix[0])

    # generate ranges, mins
    ranges = []; mins = []
    for area in areas_matrix:
        ranges.append([]); mins.append([])
        for node_bound in area: 
            ranges[-1].append(node_bound[1] - node_bound[0])
            mins[-1].append(node_bound[0])

    network_input = np.random.rand(precision * len(areas_matrix), input_nodes)

    for indx, i in enumerate(range(0, network_input.shape[0], precision)): 
        network_input[i:i+precision] = np.add(network_input[i:i+precision] * ranges[indx], mins[indx])
    
    network_output = network.predict(network_input)
    network_output = network_output.reshape(len(areas_matrix), precision, -1)

    output_bounds = []
    for output in network_output:
        bound = np.dstack((output.min(axis=0), output.max(axis=0)))[0]
        output_bounds.append(bound.tolist())

    return np.array(output_bounds)
