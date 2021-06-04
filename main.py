import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import yaml, tensorflow.keras.models, sys
from support.ProVe import ProVe
from support.main_acas import main as acas_main
from support.main_navigation import navigation_main
from support.verifier import recursive_verifier

def main():

    ymlfile = open("config/example.yml", 'r')
    job_config = yaml.safe_load(ymlfile)

    profiler_active = job_config['profiler']

    job_name = job_config['job_name']
    model_name = job_config['model_name']

    loaded_model = tensorflow.keras.models.load_model("trained_model/{}.h5".format(job_config['model_name'])) # if loadable with standard value

    input_means = job_config['input_means']
    input_ranges = job_config['input_ranges']

    condition_a = job_config['condition_a']
    condition_b = job_config['condition_b'] 
    property_mode = job_config['property_mode'] 

    input_area = job_config['input_area']

    basic_iteration = job_config['basic_iteration'] 
    round_eps = job_config['round'] 

    gpu_memory_pool = job_config['gpu_memory_pool'] 
    cpu_memory_pool = job_config['cpu_memory_pool'] 

    heuristic = job_config['heuristic'] 

    analysis_mode = job_config['analysis_mode'] 
    semi_formal_prec = job_config['semi_formal_precision'] 

    recursive_verifier(
            loaded_model, input_area, basic_iteration, round_eps, heuristic, condition_a, condition_b, 
            property_mode, gpu_memory_pool, cpu_memory_pool, input_means, input_ranges, analysis_mode, semi_formal_prec,
            profiler_active, job_name, model_name
        )

if __name__ == "__main__":
    if(sys.argv[1] == "-custom"): main()
    elif(sys.argv[1] == "-ACAS"): acas_main(int(sys.argv[2]))
    elif(sys.argv[1] == "-water"): navigation_main()
    else: raise ValueError(f"Invalid command: '{sys.argv[1]}' (options: [-ACAS *n*, -water, -custom])")
