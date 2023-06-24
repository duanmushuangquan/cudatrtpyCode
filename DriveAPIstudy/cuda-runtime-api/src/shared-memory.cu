#include <cuda_runtime.h>
#include <stdio.h>

//////////////////////demo1 //////////////////////////
/* 
demo1 主要为了展示查看静态和动态共享变量的地址
 */
const size_t static_shared_memory_num_element = 6 * 1024; // 6KB  杨秀勇：这里定义了6KB的静态的共享内存的 大小。内存大小类型为长整型。
__shared__ char static_shared_memory[static_shared_memory_num_element]; //杨秀勇：静态共享内存定义方式：前面加__shared__后面和正常数组定义一样
__shared__ char static_shared_memory2[2]; //杨秀勇：这里定义了第二个静态的共享内存数组，数组大小为2字节

__global__ void demo1_kernel(){ //杨秀勇：核函数。 下面extern告诉dynamic_shared_memory[]是在核函数外定义的。目前看，dynamic_shared_memory[]仍然是在和函数内定义
    extern __shared__ char dynamic_shared_memory[];      // 静态共享变量和动态共享变量在kernel函数内/外定义都行，没有限制
    extern __shared__ char dynamic_shared_memory2[];    //杨秀勇： 这两行定义的是动态共享内存。区别于静态共享内存，动态共享内存不能定义时直接分配大小。
                                                                //而是核函数启动时，第三个参数分配大小。例如demo1_kernel<<<1, 1, 12, nullptr>>>();中那个12，就是动态共享内存的大小。
    printf("static_shared_memory = %p\n",   static_shared_memory);   // 静态共享变量，定义几个地址随之叠加
    printf("static_shared_memory2 = %p\n",  static_shared_memory2); 
    printf("dynamic_shared_memory = %p\n",  dynamic_shared_memory);  // 动态共享变量，无论定义多少个，地址都一样
    printf("dynamic_shared_memory2 = %p\n", dynamic_shared_memory2); //杨秀勇：打印结果显示两个同台共享内存地址一样。说明你无论定义多少次，都是指向同一个东西，同一块内存。都是核函数启动时，第三个参数指定的大小。

    if(blockIdx.x == 0 && threadIdx.x == 0) // 第一个thread
        printf("Run kernel.\n");
}

/////////////////////demo2//////////////////////////////////
/* 
demo2 主要是为了演示的是如何给 共享变量进行赋值
 */
// 定义共享变量，但是不能给初始值，必须由线程或者其他方式赋值
__shared__ int shared_value1; //杨秀勇：定义了静态的shared_memroy是一个整数值

__global__ void demo2_kernel(){
    
    __shared__ int shared_value2; //杨秀勇：定义了静态的shared_memroy是一个整数值，为啥不是动态共享内存呢，因为没写extern
    //杨秀勇：符合threadIdx.x==0的只有两次  即 blockIdx.x==0时 threadIdx.x == 0一次。blockIdx.x==1时 threadIdx.x == 0一次
    //杨秀勇：blockIdx.x取值范围0-2   threadIdx.x取值范围0-5
    if(threadIdx.x == 0){

        // 在线程索引为0的时候，为shared value赋初始值
        if(blockIdx.x == 0){ // 
            shared_value1 = 123;
            shared_value2 = 55;
        }else{
            shared_value1 = 331;
            shared_value2 = 8;
        }
    }

    // 等待block内的所有线程执行到这一步
    __syncthreads(); //杨秀勇：线程thiradIdx.x == 1,2,3,4时，会在这一步等待thiradIdx.x==0的从if else语句中走出来。
    
    printf("%d.%d. shared_value1 = %d[%p], shared_value2 = %d[%p]\n", 
        blockIdx.x, threadIdx.x,
        shared_value1, &shared_value1, 
        shared_value2, &shared_value2
    );
}

void launch(){
    
    demo1_kernel<<<1, 1, 12, nullptr>>>(); //gridDim=1   blockDim=1  shared_memory=12字节， nullptr指默认的流
    demo2_kernel<<<2, 5, 0, nullptr>>>();//
}