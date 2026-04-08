inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

__kernel void forward_pass(
    __global const float* input,      
    __global const float* weights,   
    __global const float* biases,   
    __global const int* layer_offsets, //size of weights
    __global const int* bias_offsets,  // size of bias
    __global const int* layer_sizes, //size of each layer(probably all the same except input and output) 
    const int num_layers,
    __global float* output          
) {
 
    int gid = get_global_id(0);


    __local float layer_output[1024];

 
    if (gid < layer_sizes[0]) {
        layer_output[gid] = input[gid];
    }
//memory fence prevents the guys from getting a head start
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int l = 0; l < num_layers; l++) {
        int in_size = layer_sizes[2*l];      
        int out_size = layer_sizes[2*l + 1]; 

        // Only threads within the output size compute this layer
        if (gid < out_size) {
            float sum = 0.0f;
            int w_start = layer_offsets[l];
            int b_start = bias_offsets[l];

            for (int j = 0; j < in_size; j++) {
                sum += weights[w_start + gid * in_size + j] * layer_output[j];
            }

            sum += biases[b_start + gid];
            layer_output[gid] = sigmoid(sum);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

   
    int last_layer_out_size = layer_sizes[2*(num_layers-1) + 1];
    if (gid < last_layer_out_size) {
        output[gid] = layer_output[gid];
    }
}
