#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
using namespace cv;

//这里实现使用2个维度
//gridDim 的类型是dim3
// gridDim = 123  指的是 gridDim.x = 123
// 同理blockDim = 123  值得是 blockDim.x = 123
// 
// 步骤一：确定线程数量：----使用图像大小个线程。例如图像大小是100*100则启动10000个线程
// 步骤二：确定任务是否适合GPU计算，是否具备空间独立
    // 每次核函数只需要处理一个像素的计算
    // 像素之间的输出在计算上是独立的。即1个像素经过变换和其他像素没有关系。只要按照矩阵变换就好。像素之间互相不干扰
// 步骤三：确定blockDim、blockDim
    //已知：blockDim最好整除32  二维图像一般去512或256(nvidia推荐值)  2080ti中block最大值为1024,一般不取最大值
            //blockDim Max dimension size of a thread block (x,y,z): (1024, 1024, 64)    # 即 block的限制
    //已知：size.widt    size.height    需考虑如何设置 gridDim     
    //  
    //gridDim ： 因为blockDim已经确定，所以gridDim = ceil(size.width * size.height / 512)
    //blockDim ：根据本次案例是二维图像，取512
// 步骤四：对构建的核函数进行思考。在当前需要构建的核函数warp_affine_gpu_impl中。思考是不是执行次数就一定是线程数。
    // - 首先会发现旋转操作
    // - 需要控制边界

//(一个SM由两个SMP 一个SMP一般是32个Core  要通过deviceQuery查询)
__global__ void warp_affine_gpu_impl(
    unsigned char* src, //输入图片
    unsigned char* dst,//输出图片
    float* M,   //变换矩阵
    int src_width, int src_height,
    int dst_width, int dst_height,
    int edge //边界
){
    //确认position，此案例中
    //shape
    // gridDim.z   =   1;
    // gridDim.y   =   1;
    // gridDim.x   =   A    = ceil(jobs / (float)threads);
    // blockDim.z   =   1;
    // blockDim.y   =   1;
    // blockDim.x   =   B   = threads;

    //index
    // blockIdx.z   =   0;  因为gridDim.z=1,索引只能取gridDim.z - 1，所以为0
    // blockIdx.y   =   0;
    // blockIdx.x   =   U;  
    // threadIdx.z   =   0;
    // threadIdx.y   =   0;
    // threadIdx.x   =   V;
    //position = (((((0 * 1) + 0) * A + U) * 1 + 0) * 1 + 0) * B + V= U * B + V =blockIdx.x * blockDim.x + threadIdx.x
    //思考！！！！。<<<gridDim  blockDim>>到最后计算position的过程。有点像是把有维度的数据，编个号，拍扁成1维。而且使得内存连续？？？
    int position = blockIdx.x * blockDim.x + threadIdx.x;
    const int channels = 3;
    //1.边界判断 ： 线程号，会存在比jobs大的情况，大于等于jobs时，咱们不需要做任何事情，直接返回
    if(position >= edge) return;

    //2.计算dst x dst y  即已知position，求这个position在目标图像(输出图像)上的像素值索引x  y值
    //position 取值是  0-目标图图像面积。 有点像求position的反向过程
    int dx = position % dst_width;
    int dy = position / dst_width;

    //3.求ix  iy  
    //即求  输入的图像的x值y值。注意M让Xiyi.jpg变成了affine.jpg    M_inv让affine.jpg  变成affine_gpu.jpg
    //此时传入的M是M_inv
    float ix = dx * M[0] + dy * M[1] + M[2];
    float iy = dx * M[3] + dy * M[4] + M[5]; //这里涉及M的内存时连续的。M是两行三列，因为内存连续。用索引3取值就相当于取第二行第一列的数

    //4计算插值点的权重
    //主要是计算后是浮点数，如何变成整数----插值法
    int lx = floor(ix); //lx表示low x  向下取整。
    int ly = floor(iy);
    int hx = lx + 1;  //表示height x  即向上取整。
    int hy = ly + 1;

    unsigned char constant_color[3] = {0, 255, 255};

    //增加边界控制
    if(lx >= src_width || ly >= src_height || hx < 0 || hy < 0){ //上方的四种情况是极端情况。即4个像素计算后肯定超边界。而且是4个像素都超了。给他们统一赋值一个颜色。
        //把dst对应目标的像素值设置为 常量像素(constant color), border value
        unsigned char* pdst = dst + position * channels; 
        //这里计算在dst上，索引为position对应的指针的计算。即dst首地址(还是dst) + position * channels.
        //之所以乘channels，就类似贾志刚opencv中用指针循环遍历图像的思路。一个dim=3的图像。指针走三次才到下一个像素
        for(int i = 0; i < channels; ++i){
            //
            pdst[i] = constant_color[i]; // 给计算出来的像素点中的3个通道赋值
        }
        return;
    }

    //画个图 a点最左上角点。d右下角点   p插值点。 即计算出来的ix iy并不是整数。
    // low = high - p   所有的低位 = 高位 - p值
    // high = p - low   所有的高位 = p值 - 低位
    /*
            。a(lx, ly)             。b(hx, ly)
    
                    。p(ix, iy)

            。c(lx, hy)             。d(hx, hy)
    */
    float a_weight = (hx - ix) * (hy - iy); //a点的lx, ly都属于低位 使用公式high - p 。这里的high指的 hx 相比 lx是高位   hy相比ly是高位 
    float b_weight = (ix - lx) * (hy - iy); //b点的hx属于高位 使用公式p - low。 即(ix - lx) b点的ly属于低位 使用公式high - p。 即(hy - iy)
    float c_weight = (hx - ix) * (iy - ly); //c点的lx属于低位 使用公式high - p。 即(hx - ix) c点的hy属于高位
    float d_weight = (ix - lx) * (iy - ly);


    //5计算插值点所处的指针
    unsigned char* a_ptr = constant_color;
    unsigned char* b_ptr = constant_color;
    unsigned char* c_ptr = constant_color;
    unsigned char* d_ptr = constant_color;
    const int line_bytes = src_width * channels * sizeof(uchar); //sizeof(uchar) == 1

    if(lx >= 0 && lx < src_width && ly >= 0 && ly <= src_height){ //a点没越界的条件,越界的就是默认值。所以a_ptr不能用nullptr初始化
        a_ptr = src + ly * line_bytes + lx * channels; 
    }
    if(hx >= 0 && hx < src_width && ly >= 0 && ly <= src_height){ //b点没越界的条件
        b_ptr = src + ly * line_bytes + hx * channels;
    }
    if(lx >= 0 && lx < src_width && hy >= 0 && hy <= src_height){ //c点没越界的条件
        c_ptr = src + hy * line_bytes + lx * channels;
    }
    if(hx >= 0 && hx < src_width && hy >= 0 && hy <= src_height){ //d点没越界的条件
        d_ptr = src + hy * line_bytes + hx * channels;
    }


    //dst 在position位置上的指针,循环给里面的三个通道数赋值
    unsigned char* dst_ptr = dst + position * channels;
    for(int i=0; i < channels; ++i){ //i++ 会发生拷贝  i++会取值  赋值  加1三个步骤   而++i不会产生拷贝
        dst_ptr[i] = a_ptr[i] * a_weight + b_ptr[i] * b_weight + 
        c_ptr[i] * c_weight + d_ptr[i] * d_weight + 0.5f; 
        //加0.5f是为了四舍五入。opencv代码也这么做的。0.5后面一定要加f
        //所有cuda核中的小数都要加f！！！！！！！！！！！！！！
        // auto a = 0.5  //c++中自动退到为double   double的数据在CUDA中会使用DPUnit  具体看显卡介绍那个结构，DPUnit核太少。
        // auto b=0.5f 自动推导为float              float的数据在CUDA中会使用CUDA Core
    }
}

Mat warp_affine_gpu(const Mat& image, const Mat& M, const Size& size){
    //目的，将被cpp旋转后的图片 affine.jpg进行还原。

    //1. 构建输出的Mat
    Mat output = Mat(size.height, size.width, image.type());

    //2. 构建逆变换矩阵
    //传入的M是cpp文件中获得的，将  原始文件Xiyi.jpg 转换为  affine.jpg的矩阵M
    //所以还原需要一个根M变换矩阵正好相反的矩阵  M_inv
    Mat M_inv;
    cv::invertAffineTransform(M, M_inv);
    //这里求逆矩阵。是为了核函数中，利用已知的dst输出图像的index进行逆运算得到src输出图像的index
    //具体见核函数中65行左右的程序。

    //数据类型转换。M_inv变换矩阵的类型是double类型。而我们的核函数定义的类型是float*
    //数据类型转换
    M_inv.convertTo(M_inv, CV_32F);

    //3. 准备device指针
    unsigned char* src_device = nullptr;
    unsigned char* dst_device = nullptr;
    float* M_inv_device = nullptr;

    //4.分配内存
    //首先要认清图片的数据类型。图片的数据类型。此时图片为unsigned char 对应opencv的uchar 使用 sizeof(uchar） 可以查看字节数
        //unsigned char 为 1字节  8位  CV_8U
        //所以字节数 image bytes = image1.widht * image1.height  * channels * sizeof(uchar）

    //在 OpenCV 中，image.step 是一个表示图像每行存储的字节数的属性。
            // image.step.p[0] * image.size[0]是一个万能写法  只要你想获取Mat中数据所占字符数，这句话就能满足你
 
    //size_t 无符号长整型
    size_t src_bytes = image.step.p[0] * image.size[0];
    size_t dst_bytes = output.step.p[0] * output.size[0];
    size_t M_inv_bytes = M_inv.step.p[0] * M_inv.size[0];

    cudaMalloc(&src_device, src_bytes);
    cudaMalloc(&dst_device, dst_bytes);//没有初始化，一般是黑色
    cudaMalloc(&M_inv_device, M_inv_bytes);

    //5.数据拷贝到device
    //image.ptr<uchar>(0) 是一个指向图像数据第一行的指针，用于访问第一行像素值。注意返回值要加const uchar* src_ptr = image.ptr<uchar>(0)
    //image.data 就是行号为0的地址
    cudaMemcpy(src_device, image.data, src_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(M_inv_device, M_inv.data, M_inv_bytes,cudaMemcpyHostToDevice);

    //6.执行阶段
    //定义总的线程数---对于二维图片计算
    int jobs = size.area(); // size.height * size.width
    int blockDimNum = 512; //具体见上方解释
    int gridDimNum = ceil(jobs / (float)blockDimNum); //除法时，要转换成单精度float

    //<<<gridDim  blockDim  memory  stream>>>
    warp_affine_gpu_impl<<<dim3(gridDimNum, 1, 1), dim3(blockDimNum, 1, 1)>>>(
        src_device, dst_device, M_inv_device, 
        image.cols, image.rows, 
        size.width, size.height, jobs);
    
    //7.计算结果复制回来
    cudaMemcpy(output.data, dst_device, dst_bytes, cudaMemcpyDeviceToHost);

    //8.清除内存
    cudaFree(src_device);
    cudaFree(dst_device);
    cudaFree(M_inv_device);

    return output;
}