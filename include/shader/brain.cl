inline float sigmoid(float x) {
    return tanh(x);
}

inline float sigmoid_deriv(float y) {
    return 1.0f - y * y;
}

__kernel void forward_pass(
    __global const float* input,
    __global       float* weights,
    const int             num_layers,
    const int             layer_size,
    const int             input_size,
    const int             output_size,
    const int             creatures_size,
    __global const float* cost_functions,
    __global float*       output
) {
    float learning_rate = 0.03f;
    int creature = get_global_id(0);
    if (creature >= creatures_size) return;

    int brain_number = (input_size + output_size) * layer_size
                     + (num_layers - 1) * layer_size * layer_size;
    int brain_offset = creature * brain_number;
    int cost_offset  = creature * input_size * output_size;

    float activations[9][1024];
    for (int i = 0; i < input_size; i++) {
        activations[0][i] = input[creature * input_size + i];
    }

    int w_offset = 0;
    for (int l = 0; l < num_layers; l++) {
        int in_size  = (l == 0)              ? input_size : layer_size;
        int out_size = (l == num_layers - 1) ? output_size : layer_size;
        for (int n = 0; n < out_size; n++) {
            float sum = 0.0f;
            for (int j = 0; j < in_size; j++) {
                sum += weights[brain_offset + w_offset + n * in_size + j]
                     * activations[l][j];
            }
            activations[l + 1][n] = sigmoid(sum);
        }
        w_offset += in_size * out_size;
    }

    for (int i = 0; i < output_size; i++) {
        output[creature * output_size + i] = activations[num_layers][i];
    }

    float delta[1024];
    for (int i = 0; i < output_size; i++) {
        float linear_comb = 0.0f;
        for (int j = 0; j < input_size; j++) {
            linear_comb += input[creature * input_size + j]
                         * cost_functions[cost_offset + i * input_size + j];
        }
        float a  = activations[num_layers][i];
        delta[i] = 2.0f * (a - linear_comb) * sigmoid_deriv(a);
    }

    float delta_prev[1024];
    for (int l = num_layers - 1; l >= 0; l--) {
        int in_size  = (l == 0)              ? input_size : layer_size;
        int out_size = (l == num_layers - 1) ? output_size : layer_size;
        w_offset -= in_size * out_size;

        if (l > 0) {
            for (int j = 0; j < in_size; j++) {
                float sum = 0.0f;
                for (int n = 0; n < out_size; n++) {
                    sum += weights[brain_offset + w_offset + n * in_size + j]
                         * delta[n];
                }
                delta_prev[j] = sum * sigmoid_deriv(activations[l][j]);
            }
        }

        for (int n = 0; n < out_size; n++) {
            for (int j = 0; j < in_size; j++) {
                int widx = brain_offset + w_offset + n * in_size + j;
                weights[widx] -= learning_rate * delta[n] * activations[l][j];
            }
        }

        if (l > 0) {
            for (int j = 0; j < in_size; j++) {
                delta[j] = delta_prev[j];
            }
        }
    }
}
