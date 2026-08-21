#include <stdio.h>
#include <stdlib.h>

#define N 16

__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N)
    {
        C[i][j] = A[i][j] + B[i][j];
    }
}

int main()
{
    float h_A[N][N], h_B[N][N], h_C[N][N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i][j] = 1.0f;
            h_B[i][j] = 2.0f;
        }
    }

    float (*d_A)[N], (*d_B)[N], (*d_C)[N];
    size_t bytes = N * N * sizeof(float);

    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    int correct = 1;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (h_C[i][j] != 3.0f) correct = 0;

    printf(correct ? "PASS: matrix addition correct on GPU\n" : "FAIL: mismatch found\n");
    printf("Sample value C[0][0] = %.2f\n", h_C[0][0]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
