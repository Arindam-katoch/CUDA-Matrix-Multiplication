#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// __global__ means this function runs on GPU, called from CPU
__global__ void vectorAdd(float* a, float* b, float* c, int N) {
    // each thread computes one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // guard: last block may have threads beyond array size
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int N = 1000000;  // 1 million elements
    size_t bytes = N * sizeof(float);

    // --- CPU (host) memory ---
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // fill with test data
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // --- GPU (device) memory ---
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // copy input data from CPU to GPU
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // launch kernel
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;  // ceil(N/blockSize)
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // copy result back to CPU
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // verify
    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            printf("FAIL at index %d\n", i);
            pass = 0;
            break;
        }
    }
    if (pass) printf("PASS\n");

    // free memory — always
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return 0;
}