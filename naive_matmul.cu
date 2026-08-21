#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2048   // matrix size N x N — kept modest so CPU verification doesn't take forever

// ---------------------------------------------------------------------
// NAIVE KERNEL: one thread computes exactly ONE output element C[row][col]
// Every thread reads a full row of A and a full column of B directly from
// global memory — no reuse, no shared memory. This is the baseline.
// ---------------------------------------------------------------------
__global__ void matMulNaive(float *A, float *B, float *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // which output row this thread owns
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // which output column this thread owns

    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
        {
            // A is row-major: element (row, k) is at index row*n + k
            // B is row-major: element (k, col) is at index k*n + col
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// ---------------------------------------------------------------------
// CPU reference implementation — used only to verify correctness
// ---------------------------------------------------------------------
void matMulCPU(float *A, float *B, float *C, int n)
{
    for (int row = 0; row < n; row++)
        for (int col = 0; col < n; col++)
        {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
                sum += A[row * n + k] * B[k * n + col];
            C[row * n + col] = sum;
        }
}

int main()
{
    size_t bytes = N * N * sizeof(float);

    // ---- Host allocation ----
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);       // result copied back from GPU
    float *h_C_ref = (float*)malloc(bytes);   // CPU reference result

    // ---- Initialize with simple values ----
    srand(42);
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = (float)(rand() % 5);
        h_B[i] = (float)(rand() % 5);
    }

    // ---- Device allocation ----
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    // ---- Copy inputs to device ----
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // ---- Configure grid/block ----
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // ---- Time the kernel with cudaEvent_t ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulNaive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // ---- Copy result back ----
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // ---- Verify against CPU ----
    matMulCPU(h_A, h_B, h_C_ref, N);

    int correct = 1;
    for (int i = 0; i < N * N; i++)
    {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-3)
        {
            correct = 0;
            break;
        }
    }

    printf(correct ? "PASS: naive matmul matches CPU reference\n"
                    : "FAIL: mismatch with CPU reference\n");
    printf("Naive kernel time: %.4f ms\n", milliseconds);

    // ---- Cleanup ----
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
