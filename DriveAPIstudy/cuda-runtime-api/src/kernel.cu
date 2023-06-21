#include <stdio.h>
#include <cuda_runtime.h>
__device__  __host__ void show_info(){
    printf("我是show_info函数嘿嘿嘿\n");
}

__global__ void test_print_kernel(const float* pdata, int ndata){
    //此处为核函数内的作用域
    //我想在这里调用一个我自定义的show_info函数。会报错
        //src/kernel.cu(10): error: calling a __host__ function("show_info") 
        //from a __global__ function("test_print_kernel") is not allowed
    //所以，自定义的show_info函数被归为主机函数。
    //如果想在和函数内调用，必须在函数前 加 __device__
    //同样的，show_info函数会被调用线程数次；
    show_info();

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
    //此区域被认为是Host函数的作用域。
    show_info();  //对于没有__device__修饰的show_info函数来说，这里是可以被调用的。但是此时，核函数中不能调用
    //对于有__device__修饰的show_info函数来说，这里是不可以被调用的。但是此时，核函数中能调用
    //如果想show_info被两方同时调用，需要用两个前缀修饰 __device__  __host__ show_info()



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