#include <iostream>
#include <hip/hip_runtime.h>
#include <vector>
#include <chrono>

#define HIP_CHECK(command){  \
	hipError_t status = command ; \
	if(status!=hipSuccess){       \
	 std::cerr << "Error: HIP reports " << hipGetErrorString(status) << std::endl; \
	 std::abort();}}

__global__ void addVectors(int* a, int* b, int* result, int n) {
           int tid = blockIdx.x * blockDim.x + threadIdx.x;
           int a_val, b_val, result_val;

           if (tid < n) {
	      // Inline assembly using AMD GCN syntax
	      asm volatile (
                  "s_load_dwordx2 %0, %1;"
		  "s_load_dwordx2 %2, %3;"
		  "v_add_i32 %4, %0, %2;"
		  "s_store_dwordx2 %5, %4;"
		  : "=v"(a_val), "=s"(a[tid]), "=v"(b_val), "=s"(b[tid]), "=v"(result_val), "=s"(result[tid])
	      );			 
	      
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
    auto start_time = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(addVectors, dim3(numBlocks), dim3(threadsPerBlock), 0, 0, deviceA, deviceB, deviceC, n);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto start_time1 = std::chrono::high_resolution_clock::now();
    addVectors<<< dim3(numBlocks), dim3(threadsPerBlock)>>>(deviceA, deviceB, deviceC, n);
    auto end_time1 = std::chrono::high_resolution_clock::now();
    // Wait for the kernel to finish
    hipDeviceSynchronize();


    HIP_CHECK(hipMemcpy(hostC.data(), deviceC, n*sizeof(n), hipMemcpyDeviceToHost));


    if(check_result(hostC, hostA, hostB)){
      std::cout << "result match between host and device" << std::endl;
    }
    else{
      std::cout << "result do not match between host and device"<< std::endl;
    }

    std::chrono::duration<double> elapsed = end_time - start_time;
    std::chrono::duration<double> elapsed1 = end_time1 - start_time1;

    std::cout << "Kernel execution time: " << elapsed.count() * 1000 << " ms" << std::endl;
    std::cout << "Kernel1 execution time: " << elapsed1.count() * 1000 << " ms" << std::endl;

    hipFree(deviceA);
    hipFree(deviceB);
    hipFree(deviceC);
    return 0;
}

