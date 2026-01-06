#include <stdio.h>
#include <cuda_runtime.h>

// 1. THE KERNEL
// This runs on the GPU. Each thread handles one element.
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    // Calculate global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: prevent accessing memory outside the array
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000; // Number of elements
    size_t size = n * sizeof(int);

    // --- HOST (CPU) MEMORY ALLOCATION ---
    // Allocate memory for A, B, and C on the CPU
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);

    // Initialize input vectors with some values
    for(int i = 0; i < n; i++) {
        h_a[i] = i;      // 0, 1, 2...
        h_b[i] = i * 2;  // 0, 2, 4...
    }

    // --- DEVICE (GPU) MEMORY ALLOCATION ---
    // Allocate memory for A, B, and C on the GPU
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // --- DATA TRANSFER (Host -> Device) ---
    // Copy data from CPU arrays to GPU arrays
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // --- KERNEL LAUNCH ---
    int threadsPerBlock = 256;
    // Calculate grid size to cover all elements
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching kernel with %d blocks and %d threads per block\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // --- DATA TRANSFER (Device -> Host) ---
    // Copy the result back from GPU to CPU
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify result (just checking the first few)
    for(int i = 0; i < 5; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // --- CLEANUP ---
    // Free GPU memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    // Free CPU memory
    free(h_a); free(h_b); free(h_c);

    return 0;
}