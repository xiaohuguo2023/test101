#include <iostream>
#include <hip/hip_runtime.h>

__global__ void helloWorld() {
    printf("Hello, World from GPU thread %d\n", threadIdx.x);
}

int main() {
    int deviceId = 0;  // Use the first available GPU device
    hipSetDevice(deviceId);

    // Launch the kernel on the GPU
    helloWorld<<<1, 256>>>();
    hipDeviceSynchronize();  // Wait for the GPU to finish

    // Check for any errors during the GPU execution
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Hello, World from CPU!" << std::endl;

    return 0;
}

