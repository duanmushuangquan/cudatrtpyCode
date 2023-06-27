#include <cuda_runtime.h>
#include <stdio.h>

//演示如何定义静态共享内存
const size_t static_shared_memory_size = 6 * 1024;
__shared__ char static_shared_memory[static_shared_memory_size];
__shared__ char static_shared_memory2[2];

__global__ void demo1_kernel(){
    //演示如何定义动态共享内存
    extern __shared__ char dynamic_shared_memory[];
    extern __shared__ char dynamic_shared_memory2[];

    printf("static_shared_memory的地址=%p\n", static_shared_memory);
    printf("static_shared_memory2的地址=%p\n", static_shared_memory2);
    printf("dynamic_shared_memory的地址=%p\n", dynamic_shared_memory);
    printf("dynamic_shared_memory2的地址=%p\n", dynamic_shared_memory2);

    if(blockIdx.x == 0 && threadIdx.x == 0){
        
        printf("==========blockidx.x == 0 && threadIdx.x == 0,run kernel.cu==========\n");
    }

}

//==============如何赋值=============
//定义静态共享内存
__shared__ int static_shared_memory10;
__global__ void demo2_kernel(){
    //第二次定义静态共享内存
    __shared__ int static_shared_memory11;
    if (threadIdx.x == 0)
    {
        if (blockIdx.x == 0)
        {
            static_shared_memory10 = 333;
            static_shared_memory11 = 444;
        }
        else{
            static_shared_memory10 = 77777;
            static_shared_memory11 = 88888;
        }
        
    }
    //等待所有线程执行到这一步
    __syncthreads();

    printf("blockIdx.x=%d, threadIdx.x=%d, static_shared_memory10=%d, 地址为:%p, static_shared_memory11=%d, 地址为:%p\n"
    , blockIdx.x, threadIdx.x, static_shared_memory10, 
    &static_shared_memory10, 
    static_shared_memory11, &static_shared_memory11);
}


void launch(){
    demo1_kernel<<<1, 1, 12, nullptr>>>();
    demo2_kernel<<<2, 5, 0, nullptr>>>();

    cudaError_t code = cudaPeekAtLastError();
    if(code != cudaSuccess){
        const char* error_name = cudaGetErrorName(code);
        const char* error_string = cudaGetErrorString(code);
        printf("出现错误，错误名称=%s, 错误内容=%s\n", error_name, error_string);
    }
}