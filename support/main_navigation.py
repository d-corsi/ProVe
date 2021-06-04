import yaml, glob, numpy
from support.ProVe import ProVe

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization
from tensorflow.keras.models import Model


def my_load_model(model_name):
    input_shape = (17,)
    action_size = 7 
    h_layers = 2
    h_size = 64
    dueling = True
    batch_norm = False

    state_input = Input(shape=input_shape, name='input_layer')
    if batch_norm:
        h = BatchNormalization()(state_input)
    else:
        h = state_input
    for i in range(h_layers):
        h = Dense(h_size, activation='relu', name='hidden_' + str(i))(h)
        if batch_norm and i != h_layers - 1:
            h = BatchNormalization()(h)
    if (dueling):
        y = Dense(action_size + 1, activation='linear', name='dueling_layer')(h)
        y = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(action_size,), name='output_layer')(y)
    else:
        y = Dense(action_size, activation='linear', name='output_layer')(h)

    model = Model(inputs=state_input, outputs=y)

    model.load_weights(model_name)

    return model

def navigation_test(model_path, property_name):
    
    ymlfile = open(f"config/{property_name}.yml", 'r')
    job_config = yaml.safe_load(ymlfile)

    job_name = f"{property_name}_{model_path[23:-85]}"
    model_name = model_path[23:]

    profiler_active = job_config['profiler']

    loaded_model = my_load_model(model_path)

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

    PROVE = ProVe(  input_area, 
                    loaded_model, 
                    condition_a, 
                    condition_b, 
                    property_mode = property_mode, 
                    input_means = input_means,
                    input_ranges = input_ranges, 
                    analysis_mode = analysis_mode,
                    semi_formal_prec = semi_formal_prec,
                    basic_iteration = basic_iteration,
                    heuristic = heuristic,
                    round_eps = round_eps,
                    cpu_memory_pool = cpu_memory_pool, 
                    gpu_memory_pool = gpu_memory_pool, 
                ) 

    violation_rate, _ = PROVE.main_loop( verbose=0 )
    return violation_rate

def navigation_main(): 
    min_rate = 97
    print(f"Getting Models Over {min_rate}% succes rate...")

    grainbow_candidates_seed_1 = []
    grainbow_candidates_seed_2 = []
    grainbow_candidates_seed_3 = []

    success_test = [f'success{i}' for i in range(min_rate, 101)]    

    for file_name in glob.glob("trained_model/navigation_models/seed_1/*.h5"): 
        for test in success_test:
            if (test in file_name): grainbow_candidates_seed_1.append(file_name)

    for file_name in glob.glob("trained_model/navigation_models/seed_2/*.h5"): 
        for test in success_test:
            if (test in file_name): grainbow_candidates_seed_2.append(file_name)

    for file_name in glob.glob("trained_model/navigation_models/seed_3/*.h5"): 
        for test in success_test:
            if (test in file_name): grainbow_candidates_seed_3.append(file_name)
    print("\tGRainbow Candidates (seed 1):", len(grainbow_candidates_seed_1))
    print("\tGRainbow Candidates (seed 2):", len(grainbow_candidates_seed_2))
    print("\tGRainbow Candidates (seed 3):", len(grainbow_candidates_seed_3))
    print()
    
    total_violation_rate = []
    for candidate in grainbow_candidates_seed_1:
        print(f"Candidate: {candidate}, (seed 1)")
        
        total_violation_rate.append( navigation_test(candidate, "navigation_prp_1") )
        print("\tProperty 1 Violation Rate", total_violation_rate[-1])

        total_violation_rate.append( navigation_test(candidate, "navigation_prp_2") )
        print("\tProperty 2 Violation Rate", total_violation_rate[-1])

        total_violation_rate.append( navigation_test(candidate, "navigation_prp_3") )
        print("\tProperty 3 Violation Rate", total_violation_rate[-1])

        print("\tMean Violation Rate", numpy.mean(total_violation_rate))
        print()


    total_violation_rate = []
    for candidate in grainbow_candidates_seed_3:
        print(f"Candidate: {candidate}, (seed 3)")
        
        total_violation_rate.append( navigation_test(candidate, "navigation_prp_1") )
        print("\tProperty 1 Violation Rate", total_violation_rate[-1])

        total_violation_rate.append( navigation_test(candidate, "navigation_prp_2") )
        print("\tProperty 2 Violation Rate", total_violation_rate[-1])

        total_violation_rate.append( navigation_test(candidate, "navigation_prp_3") )
        print("\tProperty 3 Violation Rate", total_violation_rate[-1])

        print("\tMean Violation Rate", numpy.mean(total_violation_rate))
        print()

        