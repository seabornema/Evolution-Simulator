__kernel void generate_color_data(
__global const float* input,  
__global const float* lattice,
const int creatures_size,    
const int lattice_length,
__global float* output      
) {
int gid = get_global_id(0);
if (gid >= creatures_size) return;
int idx = gid * 4;
int x = (int)floor(input[idx+2]);
int y = (int)floor(input[idx+3]);

// Periodic boundaries
x = ((x % lattice_length) + lattice_length) % lattice_length;
y = ((y % lattice_length) + lattice_length) % lattice_length;

float col = lattice[x + (y * lattice_length)];
int out_idx = idx*4;
if (col == -1.0f) {
    output[out_idx ] = 0.0f;
    output[out_idx + 1] = 0.0f;
    output[out_idx + 2] = 1.0f;
    output[out_idx + 3] = 0.0f;
} else {
    output[out_idx] = 0.0f;
    output[out_idx + 1] = col;
    output[out_idx + 2] = 0.0f;
    output[out_idx + 3] = 0.0f;
}
}
