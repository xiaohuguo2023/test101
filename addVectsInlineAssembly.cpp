#include <iostream>
#include <hip/hip_runtime.h>
#include <vector>

#define HIP_CHECK(command){  \
	hipError_t status = command ; \
	if(status!=hipSuccess){       \
	 std::cerr << "Error: HIP reports " << hipGetErrorString(status) << std::endl; \
	 std::abort();}}

__global__ void addVectors(int* a, int* b, int* result, int n){

	int tid=threadIdx.x+ blockIdx.x * blockDim.x;

	if(tid <n){
	   int a_val, b_val, result_val;
	   asm volatile("buffer_load_dword %0, %1_offset(0);" : "=v" (a_val) : "v" (a + tid));
	   asm volatile("buffer_load_dword %0, %1_offset(0);" : "=v" (b_val) : "v" (b + tid));

	   asm volatile("v_add_i32_e32 %0, %1, %2, 0;"
			: "=v" (result_val) : "v" (a_val), "v" (b_val));
	   asm volatile ("buffer_store_dword %0, %1_offset(0);"
			:
			: "v" (result_val), "v" (result + tid));
	}
}

bool check_result(const std::vector<int>& result, const std::vector<int>& A, std::vector<int>& B){

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


    if(check_result(hostC, hostA, hostB)){
      std::cout << "result match between host and device" << std::endl;
    }
    else{
      std::cout << "result do not match between host and device"<< std::endl;
    }


    hipFree(deviceA);
    hipFree(deviceB);
    hipFree(deviceC);
    return 0;
}

