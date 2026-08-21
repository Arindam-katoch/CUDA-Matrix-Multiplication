#include <stdio.h>
__global__ void hello() {
    printf("GPU thread %d is alive\n", threadIdx.x);
}
int main() {
    hello<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
