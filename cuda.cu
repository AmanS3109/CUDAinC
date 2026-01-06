#include <stdio.h>
#include <cuda_runtime.h>

// creating a kernel
__global__ void gpuHelloWorld() {
    printf("Hello World!, thread = %d\n", threadIdx.x);
}

int main() {
    gpuHelloWorld<<<2, 5>>>();    // calling kernel
    cudaDeviceSynchronize();
    return 0;
}
