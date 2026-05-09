inline float sigmoid(float x) {
    return tanh(x);
}

__kernel void forward_pass(
    __global const float* input,
    __global const float* weights,
    const int             num_layers,
    const int             layer_size,
    const int             input_size,
    __global float*       output
) {
    int creature = get_global_id(0);
    
    int brain_number = (input_size + 4) * layer_size + (num_layers - 1) * layer_size * layer_size;
    int brain_offset = creature * brain_number;

    float layer_output[1024];


    for(int i = 0; i < input_size; i++) {
        layer_output[i] = input[creature * input_size + i];
    }

    float next_layer[1024];
    int w_offset = 0;

    for(int l = 0; l < num_layers; l++) {
        int in_size  = (l == 0)              ? input_size : layer_size;
        int out_size = (l == num_layers - 1) ? 4          : layer_size;

        for(int n = 0; n < out_size; n++) {
            float sum = 0.0f;
            for(int j = 0; j < in_size; j++) {
                sum += weights[brain_offset + w_offset + n * in_size + j] * layer_output[j];
            }
            next_layer[n] = sigmoid(sum);
        }

        for(int n = 0; n < out_size; n++) {
            layer_output[n] = next_layer[n];
        }

        w_offset += in_size * out_size;
    }

    for(int i = 0; i < 4; i++) {
        output[creature * 4 + i] = layer_output[i];
    }
}
