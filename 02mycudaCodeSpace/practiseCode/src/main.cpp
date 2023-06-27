#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <chrono>
#include <thread>

using namespace std;

#define checkRuntime(op) __check_cuda_runtime_error_code((op), #op, __FILE__, __LINE__)
#define checkCudaDriver(op) __check_cuda_driver_error_code((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime_error_code(cudaError_t code, const char* op, const char* file_name, int line_num){
    if(code != cudaSuccess){ //
        const char* error_name = cudaGetErrorName(code);
        const char* error_message = cudaGetErrorString(code);
        printf("runtime error%s:%d %s failed. \n  code = %s, message=%s\n", file_name, line_num, op, error_name, error_message);
        return false;
    }
    return true;
}

bool __check_cuda_driver_error_code(CUresult code, const char* op, const char* file_name, int line_num){
    if(code != CUresult::CUDA_SUCCESS){
        const char* err_name = nullptr;
        const char* err_string = nullptr;
        cuGetErrorName(code, &err_name);
        cuGetErrorString(code, &err_string);
        printf("runtime error%s:%d %s failed. \n  code = %s, message=%s\n", file_name, line_num, op, err_name, err_string);
        return false;
    }
    return true;
}

//下方代码使用了cudaRuntime以及 Driver API的代码。cuda开头的方法调用，同时对cu开头的方法初始化了
// int main(){
//     //下面的代码中，我们发现
//     int version = 0;
//     checkCudaDriver(cuDriverGetVersion(&version));
//     printf("当前cudaDriver版本： \t:= %d\n", version);

//     int device_count2 = 0;
//     checkCudaDriver(cuDeviceGetCount(&device_count2));

//     char device_name[100];
//     for(int i = 0; i < device_count2; ++i){
//         CUdevice device = i;
//         checkCudaDriver(cuDeviceGetName(device_name, sizeof(device_name), device));
//         printf("设备名字%s\n", device_name);
//     }
    
//     return 0;
// }


//================================上下文===============================
// int main(){
//     cout << "=============以下为main函数结果==============" <<endl;
//     //1.演示没有通过cudaSetDevice，此时上下文的情形。
//         //因为cu开头的方法前方没有使用cuda开头的方法
//         //所以要手动初始化
//     checkCudaDriver(cuInit(0));
//     CUcontext context = nullptr; // CUcontext 其实是 struct CUctx_st*（是一个指向结构体CUctx_st的指针）
//     checkCudaDriver(cuCtxGetCurrent(&context));
//     printf("第%d行,cuCtxGetCurrent得到的前上下文地址:= %p\n", __LINE__, context);

//     //2.演示通过cudaSetDevice设置上下文
//     int count = 0;
//     checkRuntime(cudaGetDeviceCount(&count));
//     printf("当前一共%d个设备\n", count);

//     int device_id = 0;
//     checkRuntime(cudaSetDevice(device_id));
//     checkCudaDriver(cuCtxGetCurrent(&context));
//     printf("第%d行,cuCtxGetCurrent得到的前上下文地址:= %p\n", __LINE__, context);

//     int current_id = 0;
//     checkRuntime(cudaGetDevice(&current_id));
//     printf("当前设备为%d\n", current_id);
//     return 0;
// }


//================================内存===============================

// int main(){
//     int device_id = 0;
//     checkRuntime(cudaSetDevice(device_id)); //初始化、初始化上下文

//     CUcontext context = nullptr;
//     checkCudaDriver(cuCtxGetCurrent(&context));
//     printf("%p\n", context);

//     int device_id2 = 1;
//     checkRuntime(cudaSetDevice(device_id2)); //初始化、初始化上下文
//     checkCudaDriver(cuCtxGetCurrent(&context));
//     printf("%p\n", context);


//     float* pmemory_device = nullptr;
//     checkRuntime(cudaMalloc(&pmemory_device, 100*sizeof(float)));

//     float* float_array = new float[100];
//     float_array[2] = 520.25;
//     checkRuntime(cudaMemcpy(pmemory_device, float_array, sizeof(float) * 100, cudaMemcpyHostToDevice));

//     float* memory_page_locked = nullptr;
//     checkRuntime(cudaMallocHost(&memory_page_locked, 100*sizeof(float)));
//     checkRuntime(cudaMemcpy(memory_page_locked, pmemory_device, sizeof(float) * 100, cudaMemcpyDeviceToHost));

//     printf("当前memory_page_locked[2]的数据=%f\n", memory_page_locked[2]);
//     checkRuntime(cudaFreeHost(memory_page_locked));
//     delete [] float_array;
//     checkRuntime(cudaFree(pmemory_device));

//     return 0;
// }


//================================流===============================

// int main(){
//     int device_id = 0;
//     checkRuntime(cudaSetDevice(device_id));

//     cudaStream_t stream = nullptr;
//     checkRuntime(cudaStreamCreate(&stream));

//     float* memory_device = nullptr;
//     size_t size = 100 * sizeof(float);
//     checkRuntime(cudaMalloc(&memory_device, size));

//     float* memory_host_pageable = new float[100];
//     memory_host_pageable[2] = 520.22222222;
//     checkRuntime(cudaMemcpyAsync(memory_device, memory_host_pageable, size, cudaMemcpyHostToDevice, stream));

//     float* page_memory_locked = nullptr;
//     checkRuntime(cudaMallocHost(&page_memory_locked, size));
//     checkRuntime(cudaMemcpyAsync(page_memory_locked, memory_device, size, cudaMemcpyDeviceToHost, stream));

//     printf("当前page_memory_locked[2]的数据为：=%f\n", page_memory_locked[2]);
//     checkRuntime(cudaStreamSynchronize(stream));
//     printf("当前page_memory_locked[2]的数据为：=%f\n", page_memory_locked[2]);

//     checkRuntime(cudaFreeHost(page_memory_locked));
//     delete [] memory_host_pageable;
//     checkRuntime(cudaFree(memory_device));
//     checkRuntime(cudaStreamDestroy(stream));
//     return 0;
// }


// ================================核函数===========================
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <stdio.h>
// #include <iostream>
// using namespace std;

// void test_print(const float* pdata, int ndata);

// int main(){
//     int device_id = 0;
//     checkRuntime(cudaSetDevice(device_id));

//     //cpu的pageable上创建内存，赋值
//     int num_data = 10;
//     size_t data_bytes = num_data * sizeof(float);
//     float* memory_pageable = new float[num_data];
//     for(int i = 0; i < num_data; ++i){
//         memory_pageable[i] = (float)i;
//     }
    

//     //数据copy到GPU上
//     float* memory_device = nullptr;
//     checkRuntime(cudaMalloc(&memory_device, data_bytes));
//     checkRuntime(cudaMemcpy(memory_device, memory_pageable, data_bytes, cudaMemcpyHostToDevice));

//     //调用函数
//     test_print(memory_device, num_data);
//     checkRuntime(cudaDeviceSynchronize()); //注意前面的是cudaStreamSynchronize

//     checkRuntime(cudaFree(memory_device));
//     delete[] memory_pageable;

//     return 0;
// }


// ================================共享内存===========================

// void launch(); //定义cuda里面执行核函数的Host函数

// int main(){
//     int device_id = 0;
//     checkRuntime(cudaSetDevice(device_id));
//     cudaDeviceProp deviceProp;
//     checkRuntime(cudaGetDeviceProperties(&deviceProp, device_id));

//     printf("deviceProp.sharedMemPerBlock,也就是共享内存大小:=%f  KB\n", deviceProp.sharedMemPerBlock / 1024.0f);
    
//     launch();
//     checkRuntime(cudaDeviceSynchronize());
//     printf("结束");
//     return 0;
// }

//==========================================
int main(){
    printf("chrono::system_clock::now()的结果为：=%d\n", chrono::system_clock::now());
    printf("chrono::system_clock::now().time_since_epoch()/ 1000.0的结果为：=%d\n", chrono::system_clock::now().time_since_epoch());
    printf("chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0：=%f\n", chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0);
    auto tic = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    auto toc = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    printf("chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0：=%f\n", chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0);
    printf("%f\n",toc - tic);
    printf("main函数启动了\n");
    return 0;
    /*
    这行代码使用 C++ 的 `<chrono>` 库来获取当前时间并计算以毫秒为单位的时间戳。

    具体解释如下：

    1. `chrono::system_clock::now()`：获取当前的系统时钟时间点。

    2. `.time_since_epoch()`：获取当前时间点与纪元（epoch）时间点之间的时间间隔。

    3. `chrono::duration_cast<chrono::microseconds>(...)`：将时间间隔转换为微秒（microseconds）的类型。

    4. `.count()`：获取时间间隔的计数值。

    5. `/ 1000.0`：将计数值除以 1000，将微秒转换为毫秒。

    最后，将得到的时间戳保存在 `tic` 变量中。

    请注意，这行代码的作用是获取当前时间的毫秒级时间戳，可能用于性能计时或时间测量等应用场景。
    
    */
}
