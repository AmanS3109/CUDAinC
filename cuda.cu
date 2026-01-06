#include <stdio.h>
#include <cuda_runtime.h>

// creating a 1D kernel
__global__ void gpuHelloWorld() {
    printf("Hello World!, thread = %d\n", threadIdx.x);
}

// creating 2D kernel
__global__ void gpu2DKernel() {
    printf("Col: %d, Row: %d\n", threadIdx.x, threadIdx.y);
}

//creating 3D kernel
__global__ void gpu3DKernel() {
    printf("Col: %d, Row: %d, Plane: %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

// flattening 2D into 1D
__global__ void flattenKernel() {
    // 1. Get the width of the grid (number of columns)
    int width = blockDim.x; 

    // 2. Get coordinates
    int x = threadIdx.x; // Column
    int y = threadIdx.y; // Row

    // 3. Apply the formula
    int uniqueId = (y * width) + x;

    printf("Col: %d, Row: %d maps to 1D Index: %d\n", x, y, uniqueId);
}

// print global ID
__global__ void printGlobalID(int N) {
    // 1. Calculate Global ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Boundary Check (Safety First!)
    if (i < N) {
        printf("Global ID: %d processing data element %d\n", i, i);
    }
}

int main() {
    int N = 100; // Total elements
    int threadsPerBlock = 32;
    
    // Calculate how many blocks we need. 
    // Usually: (N + threadsPerBlock - 1) / threadsPerBlock
    // This ensures we have enough blocks even if N doesn't divide perfectly.
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printGlobalID<<<blocksPerGrid, threadsPerBlock>>>(N);
    cudaDeviceSynchronize();
    return 0;

    gpuHelloWorld<<<1, 1>>>();    // calling 1D kernel

    dim3 blockSize(3, 3);
    gpu2DKernel<<<1, blockSize>>>();    // calling 2D kernel

    // A 3x3x3 cube of threads
    dim3 blockSize3D(3, 3, 3); 
    // Launch 1 block with 27 threads total (3*3*3)
    gpu3DKernel<<<1, blockSize3D>>>();
    
    flattenKernel<<<1, blockSize>>>();
 
    cudaDeviceSynchronize();

    return 0;
}
