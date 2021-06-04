import tensorflow.keras.models, tqdm
from support.ProVe import ProVe

def verify_prp_1(loaded_models, area):
    import time
    time_start = time.time()
    rates = []
    for loaded_model in tqdm.tqdm(loaded_models):
        PROVE = ProVe(area, loaded_model, [], [], input_means = [0, 0, 0, 0, 0], input_ranges = [2, 1, 1, 1, 1], analysis_mode='formal') 
        
        # starting loop
        total_violation_rate = 0
        while not PROVE.memory_empty():
            PROVE.split() 
            safe_indexes = []
            violation_indexes = []
            for idx, el in enumerate(PROVE.generate_output_bound()[0]):
                if(el[0][1] < 0.5011): safe_indexes.append(idx)
                if(el[0][0] > 0.5011): violation_indexes.append(idx)

            total_violation_rate += PROVE.get_area_size(violation_indexes)
            PROVE.remove_indexes(safe_indexes + violation_indexes)

        rates.append(total_violation_rate)

    print("Violation Rates", rates)
    print("Total Time", (time.time() - time_start))
    print()

def set_prop_conf(prp):

    if (prp == 1):
        upper = [0.679858, 0.500000, 0.500000, 0.500000, -0.450000]
        lower = [0.600000, -0.500000, -0.500000, 0.450000, -0.500000]
        condition_a = [0]
        condition_b = [0.5011]
        property_mode = None
        models = ['1_1', '1_2', '1_3', '1_4', '1_5', '1_6', '1_7', '1_8', '1_9', '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7', '2_8', '2_9', '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', '3_7', '3_8', '3_9', '4_1', '4_2', '4_3', '4_4', '4_5', '4_6', '4_7', '4_8', '4_9', '5_1', '5_2', '5_3', '5_4', '5_5', '5_6', '5_7', '5_8', '5_9']        

    if (prp == 2):
        upper = [0.679858, 0.500000, 0.500000, 0.500000, -0.450000]
        lower = [0.600000, -0.500000, -0.500000, 0.450000, -0.500000]
        condition_a = [0]
        condition_b = [1, 2, 3, 4]
        property_mode = 0
        models = ['2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7', '2_8', '2_9', '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', '3_7', '3_8', '3_9', '4_1', '4_3', '4_4', '4_5', '4_6', '4_7', '4_8', '4_9', '5_1', '5_2', '5_4', '5_5', '5_6', '5_7', '5_8', '5_9']        

    if (prp == 3):
        upper = [-0.298553, 0.009549, 0.500000, 0.500000, 0.500000]
        lower = [-0.303531, -0.009549, 0.493380, 0.300000, 0.300000]
        condition_a = [1, 2, 3, 4]
        condition_b = [0]
        property_mode = 1
        models = ['1_1', '1_2', '1_3', '1_4', '1_5', '1_6', '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7', '2_8', '2_9', '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', '3_7', '3_8', '3_9', '4_1', '4_2', '4_3', '4_4', '4_5', '4_6', '4_7', '4_8', '4_9', '5_1', '5_2', '5_3', '5_4', '5_5', '5_6', '5_7', '5_8', '5_9']        

    if (prp == 4):
        upper = [-0.298553, 0.009549, 0.000000, 0.500000, 0.166667]
        lower = [-0.303531, -0.009549, 0.000000, 0.318182, 0.083333]
        condition_a = [1, 2, 3, 4]
        condition_b = [0]
        property_mode = 1
        models = ['1_1', '1_2', '1_3', '1_4', '1_5', '1_6', '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7', '2_8', '2_9', '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', '3_7', '3_8', '3_9', '4_1', '4_2', '4_3', '4_4', '4_5', '4_6', '4_7', '4_8', '4_9', '5_1', '5_2', '5_3', '5_4', '5_5', '5_6', '5_7', '5_8', '5_9']        

    if (prp == 5):
        upper = [-0.321785, 0.063662, -0.499204, -0.227273, -0.166667]
        lower = [-0.324274, 0.031831, -0.500000, -0.500000, -0.500000]
        condition_a = [4]
        condition_b = [0, 1, 2, 3]
        property_mode = 1
        models = ['1_1']

    if (prp == 6):
        upper = [0.679858, -0.111408, -0.499204, -0.409091, 0.500000]
        lower = [-0.129289, -0.500000, -0.500000, -0.500000, -0.500000]
        condition_a = [0]
        condition_b = [1, 2, 3, 4]
        property_mode = 1
        models = ['1_1']

    if (prp == 7):
        upper = [0.679858, 0.500000, 0.500000, 0.500000, 0.500000]
        lower = [-0.328423, -0.500000, -0.500000, -0.500000, -0.500000]
        condition_a = [0, 1, 2]
        condition_b = [3, 4]
        property_mode = 1
        models = ['1_9']

    if (prp == 8):
        upper = [0.679858, -0.375000, 0.015915, 0.500000, 0.500000]
        lower = [-0.328423, -0.500000, -0.015915, -0.045455, 0.000000]
        condition_a = [0, 1]
        condition_b = [2, 3, 4]
        property_mode = 1
        models = ['2_9']

    if (prp == 9):
        upper = [-0.212262, -0.022282, -0.498408, -0.454545, -0.375000]
        lower = [-0.295234, -0.063662, -0.500000, -0.500000, -0.500000]
        condition_a = [3]
        condition_b = [0, 1, 2, 4]
        property_mode = 1
        models = ['3_3']

    if (prp == 10):
        upper = [0.679858, 0.500000, -0.498408, 0.500000, 0.500000]
        lower = [0.268978, 0.111408, -0.500000, 0.227273, 0.000000]
        condition_a = [0]
        condition_b = [1, 2, 3, 4]
        property_mode = 1
        models = ['4_5']

    if (prp == 11):
        upper = [-0.321785, 0.063662, -0.499204, -0.227273, -0.166667]
        lower = [-0.324274, 0.031831, -0.500000, -0.500000, -0.500000]
        condition_a = [0]
        condition_b = [4]
        property_mode = 1
        models = ['1_5']

    if (prp == 12):
        upper = [0.679858, 0.500000, 0.500000, 0.500000, -0.450000]
        lower = [0.600000, -0.500000, -0.500000, 0.450000, -0.500000]
        condition_a = [0]
        condition_b = [1, 2, 3, 4]
        property_mode = 1
        models = ['3_3']


    if (prp == 13):
        upper = [0.679858, 0.500000, 0.500000, -0.263636, -0.200000]
        lower = [0.667246, -0.500000, -0.500000, -0.500000, -0.500000]
        condition_a = [0]
        condition_b = [1, 2, 3, 4]
        property_mode = 1
        models = ['1_1']

    if (prp == 14):
        upper = [-0.321785, 0.063662, -0.499204, -0.227273, -0.166667]
        lower = [-0.324274, 0.031831, -0.500000, -0.500000, -0.500000]
        condition_a = [4]
        condition_b = [0, 1, 2, 3]
        property_mode = 1
        models = ['4_1', '5_1']

    if (prp == 15):
        upper = [-0.321785, -0.031831, -0.499204, -0.227273, -0.166667]
        lower = [-0.324274, -0.063662, -0.500000, -0.500000, -0.500000]
        condition_a = [4]
        condition_b = [0, 1, 2, 3]
        property_mode = 1
        models = ['4_1', '5_1']

    input_area = []
    input_area.append([lower[0], upper[0]])
    input_area.append([lower[1], upper[1]])
    input_area.append([lower[2], upper[2]])
    input_area.append([lower[3], upper[3]])
    input_area.append([lower[4], upper[4]])

    return models, input_area, property_mode, condition_a, condition_b


def acas_test(prp = -1):

    models, area, mode, a, b = set_prop_conf( prp )
    
    loaded_models = []
    for model in models: 
        loaded_models.append( tensorflow.keras.models.load_model("trained_model/ACAS_models/ACASXU_run2a_{}_batch_2000.h5".format(model), compile=False) )

    if prp == 1: verify_prp_1(loaded_models, area); return

    total_time = 0
    rates = []
    for loaded_model in tqdm.tqdm(loaded_models):
        PROVE = ProVe(  area, loaded_model, a, b, 
                        property_mode = mode, 
                        input_means = [0, 0, 0, 0, 0], 
                        input_ranges = [2, 1, 1, 1, 1], 
                        analysis_mode = 'formal'
                    ) 
        violation_rate, required_time = PROVE.main_loop( verbose=0 )  
        total_time += required_time 
        rates.append( violation_rate )
    
    print("Violation Rates", rates)
    print("Total Time", total_time)
    print()

def main( prp ): acas_test(prp)