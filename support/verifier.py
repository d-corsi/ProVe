import time, numpy, tqdm, random
from support.ProVe import ProVe

def recursive_verifier(loaded_model, input_area, basic_iteration, round_eps, heuristic, condition_a, condition_b, property_mode, gpu_memory_pool, cpu_memory_pool, input_means, input_ranges, analysis_mode, semi_formal_prec, profiler_active, job_name, model_name):
    time_start = time.time()
    
    PROVE = ProVe(  input_area, loaded_model, condition_a, condition_b, 
                    property_mode = property_mode, 
                    analysis_mode = analysis_mode, 
                    semi_formal_prec = semi_formal_prec,
                    heuristic = heuristic, 
                    round_eps = round_eps, 
                    cpu_memory_pool = cpu_memory_pool, 
                    gpu_memory_pool = gpu_memory_pool, 
                    input_means = input_means, 
                    input_ranges = input_ranges
                )    
    
    worst_case_iteration = get_worst_case(PROVE.areas_matrix[0], PROVE.eps)
    print("This input area require in the worst case {} iterations".format(worst_case_iteration))

    if basic_iteration == None: basic_iteration = 1
    for _ in tqdm.tqdm(range(basic_iteration)): PROVE.split()
    print("Time initial splitting: {}, Matrix size {}".format((time.time() - time_start), PROVE.get_total_entries()))
     
    
    stats = [0, PROVE.get_total_entries(), basic_iteration, 0, [], []] #Format: [Time, Memory, Iterations, Violation-Rate, Checked-Depth, Matrix-Sizes]
    stats = recursive_verifier_cycle(PROVE, basic_iteration, stats)
    stats[0] = (time.time() - time_start)

    if (profiler_active): 
        job_info = [model_name, condition_a, condition_b, input_area, PROVE.eps, PROVE.heu, PROVE.gpu_mem_pool, PROVE.cpu_mem_pool, PROVE.analysis_mode]
        save_stats_file(job_name, job_info, stats)

    print()
    return stats[3]

def recursive_verifier_cycle(PROVE, depth, stats):
    print("\nRecursive Depth {}, Max Range ***, Matrix Size {}".format(depth, PROVE.get_total_entries()))

    output_bounds, time_propagation = PROVE.generate_output_bound()
    print("Time bound calculation: {}".format(time_propagation))

    safe_indexes, violation_indexes, unkn_indexes, time_property_ver = PROVE.verify_property(output_bounds)
    current_violation_rate = PROVE.get_area_size(violation_indexes) 
    current_verified_rate = PROVE.get_area_size(safe_indexes) 
    current_checked_rate = current_violation_rate + current_verified_rate
    print("Time property verification: {}".format(time_property_ver))
    print("\tVerified: {} ({:0.3f}% of the former area)".format(len(safe_indexes), current_verified_rate))
    print("\tFailed: {} ({:0.3f}% of the former area)".format(len(violation_indexes), current_violation_rate))
    print("\tUnknown: {}".format(len(unkn_indexes)))
    
    stats[1] = max(stats[1], PROVE.get_total_entries())
    stats[2] = depth
    stats[3] += current_violation_rate
    stats[4].append(current_checked_rate)
    stats[5].append(PROVE.get_total_entries())

    print("Total Checked: {:0.3f}(%)".format(sum(stats[4])))

    PROVE.remove_indexes((safe_indexes + violation_indexes))
    if(PROVE.memory_empty()): return stats

    time_split = PROVE.split()
    print("Time next split: {}".format(time_split))

    return recursive_verifier_cycle(PROVE, depth+1, stats)


def get_worst_case(starting_area, rounding):
    tot = 0
    for i in range(0, len(starting_area), 2):
        div = 1 / (10**rounding[int(i/2)])
        a = (starting_area[i+1]-starting_area[i]) / div 
        tot += numpy.ceil(numpy.log2(a + 1))

    return int(tot)

def save_stats_file(file_name, job_info, stats, write_type="a"):
    f = open("profilation_file/job_{}.txt".format(file_name), write_type)
    f.write("================\nJob Name: {}\n================".format(file_name))
    f.write("\n\nModel Name: {}\nProperty: {} < {}\nInput Area: {}\nEPS: {}\nAnalysis Mode: {}\nHeuristic: {}".format(job_info[0], job_info[1], job_info[2], job_info[3], job_info[4], job_info[8], job_info[5]))
    f.write("\n\nGPU memory pool: {}\nCPU memory pool: {}".format(job_info[6], job_info[7]))
    f.write("\n\nTime: {}\nMemory: {}\nIterations: {}\nViolation Rate: {}%".format(stats[0], stats[1], stats[2], stats[3]))
    f.write("\n\nChecked Depth: {}\nMatrix Size Depth: {}\n\n".format(stats[4], stats[5]))
    f.close()