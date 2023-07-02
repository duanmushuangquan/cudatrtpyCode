// // 流相当于cuda上的多线程

// #include <cuda_runtime.h>
// #include <stdio.h>
// #define check_runtime(op)  __check_cudaRuntime_func((op), #op, __FILE__, __LINE__)

// bool __check_cudaRuntime_func(cudaError_t code, const char* op, const char* file_name, int code_num){
//     if(code != cudaSuccess){
//         const char* error_name = cudaGetErrorName(code);
//         const char* error_string = cudaGetErrorString(code);
//         printf("错误文件%s, 错误代码行：%d行,错误代码%s,错误名称%s,错误内容%s\n", file_name, code_num, op, error_name, error_string);
//         return false;
//     }
//     return true;
// }

// void test(cudaStream_t stream, float* array, int num);

// int main(){
//     cudaStream_t stream;
//     cudaEvent_t start, stop;
//     check_runtime(cudaEventCreate(&start));
//     check_runtime(cudaEventCreate(&stop));
//     //cudaEvent，是事件，通常用来观察流中，队列的执行情况
//     //比如统计执行时间等操作。

//     check_runtime(cudaStreamCreate(&stream));
//     //流的创建是一个重操作，不要随便创建太多，他是消耗资源的。
//     //GPU计算的基本原则，是尽可能的使计算密集，对应流就是尽可能使用流，使得任务在这个流中串行。提高GPU使用率
//     //GPU有个使用率，GPU使用率越高，越好。具体看  nvidia-smi 看 GPU-Util 即GPU单元使用率

//     //准备统计时间=========
//     float ms = 0; //毫秒
//     check_runtime(cudaEventRecord(start, stream)); 
//         //以下是需要统计时间的程序，统计设备上运行的时间
//     int num = 10000;
//     float* a = new float[num];
//     for(int i=0; i < num; ++i){
//         a[i] = i;
//     }

//     float* a_device = nullptr;
//     size_t a_bytes = sizeof(float) * num;

//     check_runtime(cudaMalloc(&a_device, a_bytes));
//     check_runtime(cudaMemcpyAsync(a_device, a, a_bytes, cudaMemcpyHostToDevice, stream));
//     //这里没有用cudaMemcpy。因为cudaMemcpy属于同步操作
//     //异步函数依赖的指针，必须在完成之前，一直存在
//     //并且异步执行时，对指针数据修改，也需要合理的理解(就是说可以修改，但需要正确的方法)
//         //例如我在cudaMemcpyAsync下面写一个for循环，修改a的数据。 
//             //for(int i=0; i < 100 ; ++i){
//             //    a[i] += 100;
//             //}
//         //虽然代码写在cudaMemcpyAsync下面，但因为cudaMemcpyAsync是异步执行。所以执行之后立马有返回值，数据仍在GPU中跑
//         //此时修改数据，结果会是未知的。

//     test(stream, a_device, num);
//     check_runtime(cudaMemcpyAsync(a, a_device, a_bytes, cudaMemcpyDeviceToHost, stream));
//     check_runtime(cudaEventRecord(stop, stream)); 
//     check_runtime(cudaEventSynchronize(stop)); //这个同步十分重要否则报错错误名称cudaErrorNotReady,错误内容device not ready

//     check_runtime(cudaEventElapsedTime(&ms, start, stop));
//     printf("cuda核的执行时间为 %.8f ms\n", ms); 
//     //结束统计时间=========
//     for(int i = 0; i < 10 ; ++i){
//         printf(i == 0 ? "%.2f \t" : "%.2f\t", a[i]);
//     }
//     printf("/n");

//     //check_runtime(cudaStreamSynchronize(stream)); 
//     //流同步。如果写了cudaEventSynchronize(stop)，就不需要上面这一行了
//     //如果写了cudaEventSynchronize并且写了cudaStreamSynchronize，这个小案例执行时间会多2ms左右

//     // check_runtime(cudaDeviceSynchronize()); //整个gpu设备的同步等待，等待任务完成
//     printf("Message from Host \n");
//     /*
//     先调用的test(), 后打印。直觉上应该打印如下：
//         printf("test()函数被调用了\n");
//         Message from Device 
//         Message from Host
//     然而实际上是(加getchar(); 才能显示全)
//         printf("test()函数被调用了\n");
//         Message from Host
//         Message from Device 

//     如果想实现这个效果，需要在test()下加同步cudaDeviceSynchronize();加上后结果如下
//         test()函数被调用了
//         Message from Device 
//         Message from Host

//     同理，加上cudaStreamSynchronize(nullptr)。也可以打到效果。这个是这一期的重点。  
//     */

//    /*
//    1）流的概念，stream，类型全称是cudaStream_t
//    1.认为流是一个线程，任务级别的线程
//    2.认为流是一个任务队列
//    3.把异步执行的任务管理起来，在需要的时候等待或者做更多的处理
//    4.默认流，指nullptr，如果给定为nullptr，就会使用默认流。

//    cuda核的执行都是异步的，通过流来实现需要的同步

//     2）cudaMemcpy，属于同步版本的内存拷贝.等价于执行cudaMemcpyAsync + cudaDeviceSynchronize
//         等价于干了   -》 发送指令(任务队列中增加一个任务)，我要复制了的消息,相当于发送cudaMemcpyAsync，其中流为默认流
//                     -》 等待复制完成，cudaDeviceSynchronize

//     3）队列特性：
//         先进先出，后进后出                

//     4）流的优化理解：
//     在warpAffine.cu中 第182  183行。两次执行cudaMemcpy。相当于执行了两次同步。
//     这种重复的同步增加了耗时。所以流的设置，是优化程序的重要方式之一
//    */

//   /*
//   1.为何，cuda核的函数默认都是异步？
//   相对于cpu来说，GPU是一个外来者。正常情况下，代码执行到GPU的函数，需要等GPU完成才能继续执行
//   这样执行就会很慢。所以核的执行都是异步的
//   */

//     // getchar(); 
//     //增加一个等待用户输入的命令，主要是为了让终端程序卡着，这样核函数中的printf信息才能打印出来
//     //没有这句话，Device上的信息打印不出来。

//     check_runtime(cudaFree(a_device));
//     delete[] a;
//     check_runtime(cudaEventDestroy(stop));
//     check_runtime(cudaEventDestroy(start));
//     check_runtime(cudaStreamDestroy(stream));
//     return 0;
// }


// /*
// 1.流stream相当于一个队列，把任务高效的串联在一起
// 2.  cudaDeviceSynchronize()要等待所有GPu完成任务，可理解为所有流完成任务同步。    
//     cudaStreamSynchronize只需要等待一个流完成任务。同步

// 3.流只有destory的时候才会消失

// */