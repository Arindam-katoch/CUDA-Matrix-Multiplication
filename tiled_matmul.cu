#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 2048
#define TILE_WIDTH 16   // must match threadsPerBlock dimensions below

// ---------------------------------------------------------------------
// TILED KERNEL: threads in a block cooperate to load a TILE_WIDTH x TILE_WIDTH
// chunk of A and B into fast on-chip shared memory ONCE, then every thread in
// the block reuses that shared data many times instead of re-reading global
// memory for every multiply. This is what cuts global memory traffic drastically.
// ---------------------------------------------------------------------
__global__ void matMulTiled(float *A, float *B, float *C, int n)
{
    // Shared memory tiles — visible to all threads in this block only
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;   // global output row this thread owns
    int col = blockIdx.x * TILE_WIDTH + tx;   // global output column this thread owns

    float sum = 0.0f;

    // Number of tiles needed to sweep across the full row/column of length n
    int numTiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++)
    {
        // ---- Load one element of A and one element of B into shared memory ----
        // Each thread loads exactly one element per tile — collectively the
        // whole block fills both tiles.
        int aCol = t * TILE_WIDTH + tx;   // column index into A for this tile
        int bRow = t * TILE_WIDTH + ty;   // row index into B for this tile

        if (row < n && aCol < n)
            tileA[ty][tx] = A[row * n + aCol];
        else
            tileA[ty][tx] = 0.0f;   // out-of-bounds padding

        if (col < n && bRow < n)
            tileB[ty][tx] = B[bRow * n + col];
        else
            tileB[ty][tx] = 0.0f;

        __syncthreads();   // wait until ALL threads finish loading before computing

        // ---- Compute partial sum using the tile currently in shared memory ----
        for (int k = 0; k < TILE_WIDTH; k++)
            sum += tileA[ty][k] * tileB[k][tx];

        __syncthreads();   // wait until ALL threads finish computing before next tile overwrites shared memory
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
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

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                   (N + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulTiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

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

    printf(correct ? "PASS: tiled matmul matches CPU reference\n"
                    : "FAIL: mismatch with CPU reference\n");
    printf("Tiled kernel time: %.4f ms\n", milliseconds);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
