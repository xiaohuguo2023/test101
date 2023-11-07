#include <iostream>
#include <hip/hip_runtime.h>
#include <vector>

__global__ void addVectors(int* a, int *b, int* result, int n){
	int tid=threadIdx.x+ blockIdx.x * blockDim.x;
	if(tid <n){
	   result(tid)= a[tid] + b[tid];
	}
}

bool check_result(const std::vector<int>& result, const std::vector<int>& A, std::vector<int>&, B){
	for(int i=0; i<result.size(); i++){
	   if(result[i] != A[i]+B[i]){
	      return false;
	   }
	}
        
	return true;
}

int main() {

    const int n=10000;
    // Define the number of blocks and threads per block
    int numBlocks = (n+256-1)/256;
    int threadsPerBlock = 256;

    // host array A, B and results array C
    std::vector<int> hostA(n);
    std::vector<int> hostB(n);
    std::vector<int> hostC(n);

    // host array init
    for(int i=0; i<n; i++){
	    hostA[i]=i;
	    hostB[i]=i;
    }

    // Allocate device arrays
    int* deviceA;
    int* deviceB;
    int* deviceC;

    HIP_CHECK(hipMalloc(&deviceA, n*sizeof(int)));
    HIP_CHECK(hipMalloc(&deviceB, n*sizeof(int)));
    HIP_CHECK(hipMalloc(&deviceC, n*sizeof(int)));

    HIP_CHECK(hipMemcpy(deviceA, hostA.data(), n*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(deviceB, hostB.data(), n*sizeof(int), hipMemcpyHostToDevice));

    // Launch the HIP kernel
    hipLaunchKernelGGL(addVectors, dim3(numBlocks), dim3(threadsPerBlock), 0, 0, deviceA, deviceB, deviceC, n);
    addVectors<<< dim3(numBlocks), dim3(threadsPerBlock)>>>(deviceA, deviceB, deviceC, n);
    // Wait for the kernel to finish
    hipDeviceSynchronize();


    HIP_CHECK(hipMemcpy(hostC.data(), deviceC, n*sizeof(n), hipMemcpyDeviceToHost));


    if(checkResults(hostC, hostA, hostB)){
      std::cout << "result match between host and device";
    }
    else{
      std::cout << "result do not match between host and device";
    }


    hipFree(deviceA);
    hipFree(deviceB);
    hipFree(deviceC);
    return 0;
}

