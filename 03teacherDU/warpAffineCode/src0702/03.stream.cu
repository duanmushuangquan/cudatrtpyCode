// #include <cuda_runtime.h>
// #include <stdio.h>
// #include <chrono>
// #include <thread>

// __global__ void test_kernel(float* array, int edge){
//     /*
//     gridDim.z       blockIdx.z
//     gridDim.y       blockIdx.y
//     gridDim.x   x     blockIdx.x
//     blockDim.z      threadIdx.z
//     blockDim.y      threadIdx.y
//     blockDim.x  512    threadIdx.x
//     */
//     int position = blockIdx.x * blockDim.x + threadIdx.x;
//     if (position >= edge) return;
//     array[position] *= 0.5f; 
//     // ！！！这里一定要小心小心再小心，使用f，否则0.5默认double类型
//     // 这个案例很难演示
//     // printf("message form Device"); //线程过多就不要打印了
// }

// void test(cudaStream_t stream, float* array, int num){
//     int threads= 512;
//     int blocks = ceil(num / (float)threads);
//     printf("test()函数被调用了\n");
//     test_kernel<<<blocks, threads, 0, stream>>>(array, num);
//     //会启动32个线程，但是只有1个是激活的
//     //test_kernel是异步执行的。cuda核的执行都是异步的。
// }