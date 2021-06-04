import cupy as cp
import numpy as np
import time, tensorflow.keras.activations

my_kernel = cp.RawKernel(
    '''
    extern "C" __global__ void my_kernel(float* sub_areas, int sub_areas_n, int* layer_sizes, int layer_number, float* full_weights, float* full_biases, float* results_cuda, int max_layer_size) {

        // Calculate all the bounds, node by node, for each layer. 'new_layer_values' is the current working layer, old layer is the prevoius (first step old layer is the input layer)

        int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
        if (thread_id >= sub_areas_n) return;

        int area_start = thread_id * layer_sizes[0] * 2;
        
        float* old_layer_values = new float[max_layer_size * 2]();
        float* new_layer_values = new float[max_layer_size * 2]();

        // Step 1: copy inputs in 'old_layer_values' ('new_layer_values' is the first hidden layer)
        for (int i = 0; i < (2 * layer_sizes[0]); i++) old_layer_values[i] = sub_areas[area_start + i];
        
        // Step 2: starting the propagation cycle
        int bias_index = 0;
        int weights_index = 0;

        for (int layer_idx = 0; layer_idx < layer_number - 1; layer_idx ++){
            int old_layer_size = layer_sizes[layer_idx];
            int new_layer_size = layer_sizes[layer_idx + 1];
            
            for (int new_node_idx = 0; new_node_idx < new_layer_size*2; new_node_idx += 2){
                for (int old_node_idx = 0; old_node_idx < old_layer_size*2; old_node_idx += 2){

                    if(full_weights[weights_index] > 0) {
                        new_layer_values[new_node_idx] += (old_layer_values[old_node_idx] * full_weights[weights_index]); //lower bound
                        new_layer_values[new_node_idx + 1] += (old_layer_values[old_node_idx + 1] * full_weights[weights_index]); //upper bound
                    } else {
                        new_layer_values[new_node_idx] += (old_layer_values[old_node_idx + 1] * full_weights[weights_index]); //lower bound
                        new_layer_values[new_node_idx + 1] += (old_layer_values[old_node_idx] * full_weights[weights_index]); //upper bound
                    }
                    weights_index += 1;
                }
                // Adding bias for each layer (including the output)
                new_layer_values[new_node_idx] += full_biases[bias_index];
                new_layer_values[new_node_idx+1] += full_biases[bias_index];  
                bias_index += 1;

                // Relu application, except for the output layer that use linear
                if (layer_idx < layer_number - 2){
                    if (new_layer_values[new_node_idx] < 0)  new_layer_values[new_node_idx] = 0; // Apply ReLu
                    if (new_layer_values[new_node_idx+1] < 0)  new_layer_values[new_node_idx+1] = 0; // Apply ReLu
                }
            }

            for (int i = 0; i < max_layer_size * 2; i++) old_layer_values[i] = new_layer_values[i];
            for (int i = 0; i < max_layer_size * 2; i++) new_layer_values[i] = 0;
        }

        // Step 3: copy the local output layer in the global 'results_cuda' array
        int results_start = thread_id * layer_sizes[layer_number - 1] * 2;
        for (int i=0; i < layer_sizes[layer_number - 1] * 2; i++) results_cuda[results_start + i] = old_layer_values[i];

        // Free memory
        delete[] old_layer_values;
        delete[] new_layer_values;        
    }
    '''
    , 'my_kernel')

def get_output_bound_gpu(input_network, sub_areas, thread_number=32):
    time_start = time.time()
    
    layer_sizes = [input_network.layers[0].input_shape[1]]
    full_weights = np.array([])
    full_biases = np.array([])

    for idx, layer in enumerate(input_network.layers):
        layer_sizes.append(layer.output_shape[1])
        weight, bias = layer.get_weights()
        full_weights = np.concatenate((full_weights, weight.T.reshape(-1)))
        full_biases = np.concatenate((full_biases, bias.reshape(-1)))

        try:
            if(idx < len(input_network.layers) - 1): assert (layer.activation == tensorflow.keras.activations.relu)
            else: assert(layer.activation == tensorflow.keras.activations.linear)
        except:
            raise AssertionError("Invalid activation functions. Formal version of ProVe requires ReLU for the hidden layers and Linear for the output layer.")
    
    max_layer_size = max(layer_sizes)
    results_cuda = cp.zeros(layer_sizes[-1] * 2 * len(sub_areas), dtype=cp.float32)
    layer_sizes = cp.array(layer_sizes, dtype=cp.int32)
    sub_areas = cp.array(sub_areas, dtype=cp.float32)
    full_weights = cp.array(full_weights, dtype=cp.float32)
    full_biases = cp.array(full_biases, dtype=cp.float32)
    
    block_number = int(len(sub_areas) / thread_number) + 1

    kernel_input = (sub_areas, len(sub_areas), layer_sizes, len(layer_sizes), full_weights, full_biases, results_cuda, max_layer_size)
    my_kernel((block_number, ), (thread_number, ), kernel_input)
    cp.cuda.Stream.null.synchronize()

    results_reshaped = cp.asnumpy(results_cuda).reshape((len(sub_areas), input_network.layers[-1].output_shape[1], 2))

    return results_reshaped, (time.time() - time_start)
