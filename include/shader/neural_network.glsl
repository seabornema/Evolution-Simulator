#version 460

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputVector {
    float v_in[];
};

layout(std430, binding = 1) readonly buffer Matrices {
    float matrices[];
};

layout(std430, binding = 2) writeonly buffer OutputVector {
    float v_out[];
};

uniform int N;
uniform int M;

shared float temp[1024];
shared float temp2[1024];

void main() {
    uint i = gl_GlobalInvocationID.x;

    if (i >= uint(N)) return;

    temp[i] = v_in[i];
    barrier();

    for (int mat = 0; mat < M; mat++) {

        float sum = 0.0;

        for (int j = 0; j < N; j++) {
            float a = matrices[mat * N * N + i * N + j]; 
            float b = temp[j];
            sum += a * b;
        }

        temp2[i] = sum;

        barrier();

        temp[i] = temp2[i];

        barrier();
    }

    v_out[i] = temp[i];
}
