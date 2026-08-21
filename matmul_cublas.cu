#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>

#define N 512

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

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_C_ref = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = (float)(rand() % 5);
        h_B[i] = (float)(rand() % 5);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // ---- Set up cuBLAS ----
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);

    float alpha = 1.0f, beta = 0.0f;
// Warm-up call — absorbs one-time initialization cost, not measured
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
cudaDeviceSynchronize();

// NOW start the real timer
cudaEventRecord(start);
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
cudaEventRecord(stop);
    // ---- Time it ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // IMPORTANT: cuBLAS assumes COLUMN-major storage, but our arrays are
    // ROW-major (standard C layout). The trick: C = A*B in row-major is
    // mathematically identical to C^T = B^T * A^T in column-major, which
    // is exactly what this call computes if we just swap A and B's order
    // and swap which matrix is "m" vs "n". This is a well-known trick,
    // not a mistake — memorize it, it always trips people up the first time.
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,
                d_A, N,
                &beta,
                d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    matMulCPU(h_A, h_B, h_C_ref, N);

    int correct = 1;
    for (int i = 0; i < N * N; i++)
    {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-1)   // slightly looser tolerance — cuBLAS uses different summation order
        {
            correct = 0;
            break;
        }
    }

    printf(correct ? "PASS: cuBLAS matmul matches CPU reference\n"
                    : "FAIL: mismatch with CPU reference\n");
    printf("cuBLAS time: %.4f ms\n", milliseconds);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
