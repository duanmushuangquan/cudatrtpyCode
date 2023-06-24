
#include <cuda_runtime.h>
#include <stdio.h>

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

void launch();

int main(){
    //1.0 获取设备信息。单独打印sharedMem的大小
    //越近越快，越近越贵
    cudaDeviceProp prop;
    checkRuntime(cudaGetDeviceProperties(&prop, 0));
    printf("prop.sharedMemPerBlock = %.2f KB\n", prop.sharedMemPerBlock / 1024.0f); //只有48kb

    launch(); //写在shared-memory.cu中
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaDeviceSynchronize());
    printf("done\n");
    return 0;
}