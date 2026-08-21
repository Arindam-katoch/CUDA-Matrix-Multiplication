#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2048   // match the size used for naive/tiled/cuBLAS GPU runs

// ---------------------------------------------------------------------
// Plain single-threaded CPU matrix multiply — the "what if you didn't
// have a GPU at all" baseline. No CUDA, no tricks, just three nested loops.
// This is deliberately the kind of code someone with zero GPU knowledge
// would write on day one.
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
    size_t bytes = (size_t)N * N * sizeof(float);

    float *A = (float*)malloc(bytes);
    float *B = (float*)malloc(bytes);
    float *C = (float*)malloc(bytes);

    srand(42);   // same seed as GPU versions, so it's the same problem
    for (int i = 0; i < N * N; i++)
    {
        A[i] = (float)(rand() % 5);
        B[i] = (float)(rand() % 5);
    }

    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);

    matMulCPU(A, B, C, N);

    clock_gettime(CLOCK_MONOTONIC, &stop);

    double elapsed_ms = (stop.tv_sec - start.tv_sec) * 1000.0 +
                         (stop.tv_nsec - start.tv_nsec) / 1e6;

    printf("CPU (single-threaded, naive) time: %.2f ms\n", elapsed_ms);
    printf("Sample value C[0][0] = %.2f (sanity check only)\n", C[0]);

    free(A); free(B); free(C);
    return 0;
}