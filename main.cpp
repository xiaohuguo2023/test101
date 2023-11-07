#include <iostream>
#include <hip/hip_runtime.h>

__global__ void helloWorldKernel() {
    printf("Hello, World from GPU thread %d\n", hipThreadIdx_x);
}

int main() {
    // Define the number of blocks and threads per block
    int numBlocks = 2;
    int threadsPerBlock = 128;

    // Launch the HIP kernel
    hipLaunchKernelGGL(helloWorldKernel, dim3(numBlocks), dim3(threadsPerBlock), 0, 0);

    // Wait for the kernel to finish
    hipDeviceSynchronize();

    std::cout << "Hello, World from the CPU!" << std::endl;

    return 0;
}

