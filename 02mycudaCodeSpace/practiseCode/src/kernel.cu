#include <stdio.h>
#include <cuda_runtime.h>

__global__ void test_print_kernel(const float* pdata, int ndata){
    /*
    gridDim.z= 1       blockIdx.z 0
    gridDim.y 1       blockIdx.y 0
    gridDim.x  1     blockIdx.x  0
    blockDim.z 1     threadIdx.z 0
    blockDim.y 1     threadIdx.y 0
    blockDim.x 10     threadIdx.x= 0-9
    */

   int position = threadIdx.x;
   printf("%d\n", position);
   printf("pdata[%d] = %f \tthreadIdx.x=%d, blockIdx.x = %d, blockDim.x=%d\n", position, pdata[position], threadIdx.x, blockIdx.x, blockDim.x);
}

void test_print(const float* pdata, int ndata){
    printf("print_info 成功启动了\n");

    test_print_kernel<<<1, ndata, 0, nullptr>>>(pdata, ndata);
    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess)
    {
        const char* error_name = cudaGetErrorName(code);
        const char* error_message = cudaGetErrorString(code);
        printf("runtime error  code = %s, message=%s\n", error_name, error_message);
    }
}

