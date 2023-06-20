
// // CUDA运行时头文件
// #include <cuda_runtime.h>

// // CUDA驱动头文件
// #include <cuda.h>
// #include <stdio.h>
// #include <string.h>

// #define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

// bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
//     if(code != cudaSuccess){    
//         const char* err_name = cudaGetErrorName(code);    
//         const char* err_message = cudaGetErrorString(code);  
//         printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
//         return false;
//     }
//     return true;
// }

// int main(){

//     CUcontext context = nullptr;
//     cuCtxGetCurrent(&context);
//     printf("Current context = %p，当前无context\n", context);

//     // cuda runtime是以cuda为基准开发的运行时库
//     // cuda runtime所使用的CUcontext是基于cuDevicePrimaryCtxRetain函数获取的
//     // 即，cuDevicePrimaryCtxRetain会为每个设备关联一个context，通过cuDevicePrimaryCtxRetain函数可以获取到
//     // 而context初始化的时机是懒加载模式，即当你调用一个runtime api时，会触发创建动作
//     // 也因此，避免了cu驱动级别的init和destroy操作。使得api的调用更加容易
//     int device_count = 0;
//     //杨秀勇：第一次调用的API，自动触发了初始化 cuInit（）
//     checkRuntime(cudaGetDeviceCount(&device_count));
//     printf("device_count = %d\n", device_count);

//     // 取而代之，是使用setdevice来控制当前上下文，当你要使用不同设备时
//     // 使用不同的device id
//     // 注意，context是线程内作用的，其他线程不相关的, 一个线程一个context stack
//     int device_id = 0;
//     printf("set current device to : %d，这个API依赖CUcontext，触发创建并设置\n", device_id);
//     //杨秀勇：第一次关联设备，自动出发了cuDevicePrimaryCtxRetain（）
//     checkRuntime(cudaSetDevice(device_id));

//     // 注意，是由于set device函数是“第一个执行的需要context的函数”，所以他会执行cuDevicePrimaryCtxRetain
//     // 并设置当前context，这一切都是默认执行的。注意：cudaGetDeviceCount是一个不需要context的函数
//     // 你可以认为绝大部分runtime api都是需要context的，所以第一个执行的cuda runtime函数，会创建context并设置上下文
//     cuCtxGetCurrent(&context);
//     printf("SetDevice after, Current context = %p，获取当前context\n", context);

//     int current_device = 0;
//     checkRuntime(cudaGetDevice(&current_device));
//     printf("current_device = %d\n", current_device);
//     return 0;
// }


//=============================================

// // CUDA运行时头文件
// #include <cuda_runtime.h>

// #include <stdio.h>
// #include <string.h>

// #define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

// bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
//     if(code != cudaSuccess){    
//         const char* err_name = cudaGetErrorName(code);    
//         const char* err_message = cudaGetErrorString(code);  
//         printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
//         return false;
//     }
//     return true;
// }

// int main(){

//     int device_id = 0;
//     checkRuntime(cudaSetDevice(device_id));

//     //globle memory   显卡上芯片周围的那一圈
//     float* memory_device = nullptr;
//     checkRuntime(cudaMalloc(&memory_device, 100 * sizeof(float))); // pointer to device

//     //pageable memory 内存上的可分页内存(一般读取这里的数据不一定准确)
//     float* memory_host = new float[100];
//     memory_host[2] = 520.25;
//     checkRuntime(cudaMemcpy(memory_device, memory_host, sizeof(float) * 100, cudaMemcpyHostToDevice)); // 返回的地址是开辟的device地址，存放在memory_device

//     float* memory_page_locked = nullptr;
//     checkRuntime(cudaMallocHost(&memory_page_locked, 100 * sizeof(float))); // 返回的地址是被开辟的pin memory的地址，存放在memory_page_locked
//     checkRuntime(cudaMemcpy(memory_page_locked, memory_device, sizeof(float) * 100, cudaMemcpyDeviceToHost)); // 

//     //上面数据在pageable memory上创建   ---》globle memory ----》pin memory----》打印
//     printf("%f\n", memory_page_locked[2]);
//     checkRuntime(cudaFreeHost(memory_page_locked));
//     delete [] memory_host;
//     checkRuntime(cudaFree(memory_device)); 

//     return 0;
// }


//==========================================================================================================================



// // CUDA运行时头文件
// #include <cuda_runtime.h>

// #include <stdio.h>
// #include <string.h>

// #define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

// bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
//     if(code != cudaSuccess){    
//         const char* err_name = cudaGetErrorName(code);    
//         const char* err_message = cudaGetErrorString(code);  
//         printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
//         return false;
//     }
//     return true;
// }

// int main(){

//     int device_id = 0;
//     checkRuntime(cudaSetDevice(device_id)); //设置cuda0设备。内部自动调用cuInint函数

//     cudaStream_t stream = nullptr; // 新建cuda流指针
//     checkRuntime(cudaStreamCreate(&stream)); // 创建流

//     // 在GPU上开辟空间
//     float* memory_device = nullptr; 
//     checkRuntime(cudaMalloc(&memory_device, 100 * sizeof(float))); //在global memory上开辟内存

//     // 在CPU上开辟空间并且放数据进去，将数据复制到GPU
//     float* memory_host = new float[100]; //pageable memory上开辟 100个float大小的字节数。  32位是= float为4字节   64位   8字节 
//     memory_host[2] = 520.25;
//     checkRuntime(cudaMemcpyAsync(memory_device, memory_host, sizeof(float) * 100, cudaMemcpyHostToDevice, stream)); // 异步复制操作，主线程不需要等待复制结束才继续
//     //注意上方拷贝内存函数   增加了Async   同理 参数中 也增加了流
//     //执行了cudaMemcpyAsync，立即返回结果。即使没有拷贝完成，也立即返回结果

//     // 在CPU上开辟pin memory,并将GPU上的数据复制回来 
//     float* memory_page_locked = nullptr;
//     checkRuntime(cudaMallocHost(&memory_page_locked, 100 * sizeof(float)));
//     checkRuntime(cudaMemcpyAsync(memory_page_locked, memory_device, sizeof(float) * 100, cudaMemcpyDeviceToHost, stream)); // 异步复制操作，主线程不需要等待复制结束才继续
//     printf("%f\n", memory_page_locked[2]); //这行说明在没有执行下一行同步的时候。值可能不是最终值（异步没有结束）
//     checkRuntime(cudaStreamSynchronize(stream));//!!!!!通过stream同步的方式，统一等待所有操作结束。主要耗时任务就在这里。
//     printf("%f\n", memory_page_locked[2]);
    
//     // 释放内存
//     checkRuntime(cudaFreeHost(memory_page_locked));
//     checkRuntime(cudaFree(memory_device));
//     checkRuntime(cudaStreamDestroy(stream));
//     delete [] memory_host;
//     return 0;
// }

//==========================================================================================================================



// #include <cuda_runtime.h>
// #include <stdio.h>

// #define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

// bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
//     if(code != cudaSuccess){    
//         const char* err_name = cudaGetErrorName(code);    
//         const char* err_message = cudaGetErrorString(code);  
//         printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
//         return false;
//     }
//     return true;
// }

// void test_print(const float* pdata, int ndata);

// int main(){
//     float* parray_host = nullptr;
//     float* parray_device = nullptr;
//     int narray = 10;
//     int array_bytes = sizeof(float) * narray;

//     parray_host = new float[narray];
//     checkRuntime(cudaMalloc(&parray_device, array_bytes));

//     for(int i = 0; i < narray; ++i)
//         parray_host[i] = i;
    
//     checkRuntime(cudaMemcpy(parray_device, parray_host, array_bytes, cudaMemcpyHostToDevice));
//     test_print(parray_device, narray);
//     checkRuntime(cudaDeviceSynchronize());

//     checkRuntime(cudaFree(parray_device));
//     delete[] parray_host;
//     return 0;
// }



//==========================================================================================================================
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void test_print_kernel(const float* pdata, int ndata){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    /*    dims                 indexs
        gridDim.z            blockIdx.z
        gridDim.y            blockIdx.y
        gridDim.x            blockIdx.x
        blockDim.z           threadIdx.z
        blockDim.y           threadIdx.y
        blockDim.x           threadIdx.x

        Pseudo code:
        position = 0
        for i in 6:
            position *= dims[i]
            position += indexs[i]
    */
    printf("Element[%d] = %f, threadIdx.x=%d, blockIdx.x=%d, blockDim.x=%d\n", idx, pdata[idx], threadIdx.x, blockIdx.x, blockDim.x);
}

void test_print(const float* pdata, int ndata){

    // <<<gridDim, blockDim, bytes_of_shared_memory, stream>>>
    test_print_kernel<<<1, ndata, 0, nullptr>>>(pdata, ndata);

    // 在核函数执行结束后，通过cudaPeekAtLastError获取得到的代码，来知道是否出现错误
    // cudaPeekAtLastError和cudaGetLastError都可以获取得到错误代码
    // cudaGetLastError是获取错误代码并清除掉，也就是再一次执行cudaGetLastError获取的会是success
    // 而cudaPeekAtLastError是获取当前错误，但是再一次执行 cudaPeekAtLastError 或者 cudaGetLastError 拿到的还是那个错
    // cuda的错误会传递，如果这里出错了，不移除。那么后续的任意api的返回值都会是这个错误，都会失败
    cudaError_t code = cudaPeekAtLastError();
    if(code != cudaSuccess){    
        const char* err_name    = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("kernel error %s:%d  test_print_kernel failed. \n  code = %s, message = %s\n", __FILE__, __LINE__, err_name, err_message);   
    }
}