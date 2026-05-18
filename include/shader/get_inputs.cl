__kernel void generate_color_data(
__global const float* input,  
__global const float* lattice,
const int creatures_size,    
const int lattice_length,
__global float* output      
) {
int gid = get_global_id(0);
if (gid >= creatures_size) return;
int idx = gid * 2;
int x = (int)floor(input[idx]);
int y = (int)floor(input[idx+1]);

x = ((x % lattice_length) + lattice_length) % lattice_length;
y = ((y % lattice_length) + lattice_length) % lattice_length;

float col = lattice[x + (y * lattice_length)];
int out_idx = idx*2;
if (col == -1.0f) {
    output[out_idx ] = -1.0;
    output[out_idx + 1] = 0.0f;
} else {
    output[out_idx] = col;
    output[out_idx + 1] = 0.0f;
}
}
