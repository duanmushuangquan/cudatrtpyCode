// #include <opencv2/opencv.hpp>
// using namespace cv;

// //知识点。cpp文件如何直接调用cu文件中的函数？
//     // 1) 如果引入的是头文件。在
//     //
//     // 如果有默认参数，应该加在声明里还是实现里？  加在声明里。
//         // 编译的时候，默认函数就已经确认了。 在链接的时候传递给实现。实现在连接时才发生关系
//         //
// //gpu实现的函数声明
// Mat warp_affine_gpu(const Mat& image, const Mat& M, const Size& size);

// int main(){
//     Mat image1 = cv::imread("/home/shenlan09/YXY/trtpystudy/warpAffineCode/img/Xiyi.jpg");
//     Mat M = cv::getRotationMatrix2D(Point2f(image1.cols * 0.5, image1.rows * 0.5), 30, 0.85);
    
//     Mat res_affine;
//     cv::warpAffine(image1, res_affine, M, image1.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 0, 0));
//     cv::imwrite("affine.jpg", res_affine);

//     /*
//     实现了warp_affine_gpu，很多功能都可以实现。
//     例如我们想对image图像缩小一下，只需要把M这个矩阵换一下
//     */
//     Mat res_scale;
//     float scale = 0.25;
//     float scaleM_data[]={
//         scale, 0, 0,
//         0, scale, 0
//     };
//     Mat scaleM = Mat(2, 3, CV_32F, scaleM_data);
//     res_scale = warp_affine_gpu(image1, scaleM, Size(image1.cols * scale, image1.rows * scale));
//     cv::imwrite("resize.jpg", res_scale);


//     res_affine = warp_affine_gpu(image1, M, image1.size());
//     cv::imwrite("affine_gpu.jpg", res_affine);

//     return 0;
// }

//    /*
//     warp_affine_gpu这个函数的思路如下：
//     1.我们想在cpu中用一个变换矩阵。将image1变成res。
//         - 问题：
//             - 把啥放到cuda中？开辟多大空间？
//                 - 把输入图像、变换矩阵数据传入cuda，并开辟空间。输出图像只在cuda中开辟空间
//                 - 开辟根图像所占字节数大小一致的空间。即 image.width * image.height * sizeof(uchar) * channel 
//                 - 但是可以用万能的image.step.p[0] * image.size[0]算出字节数;
//             - 需要多少个线程。cuda函数gridDim  blockDim咋取
//                 - 线程数等于图片像素个数个。即jobs = image.area()
//                 - 先确定blockDim，从deviceQuery中知道blockDim第一个维度的限制一般是1024，cuda推荐取512，且是32的整数倍
//                 - 再确定gridDim    ceil(jobs / blockDim)
//             - gridDim  blockDim确定后，position能计算出来。position能干啥用？
//                 - gridDim  blockDim确定后 一个虚拟的网格就画好了。
//                 - position 就可以根据 gridDim  blockDim  blockIdx threadIdx计算出来
//                 - 数据按照网格从头到尾，由position编号
//                 - position编好号的图像，变换后，position号码不变。可以用来计算像素的索引
//                     -     int dx = position % dst_width;
//                     -     int dy = position / dst_width;
//             - 为啥要加算一个逆矩阵？逆矩阵计算后为啥要转成CV_32F类型
//                 - cv::invertAffineTransform(M, M_inv);转换好的是double类型，cuda我们一般用CUDACore
//                 - 首先要知道，旋转后的输出图像是原来的输入图像的大小。所以上面的公式计算出来的dx dy结果是固定的。
//                     - 那如何计算iamge上的像素点  经过M变换后  像素点跑哪里去了呢？
//                     - 于是有了选择。
//                         - 1）在image上计算出x、y坐标。然后用M矩阵，计算出这个像素点变换后，在dst上的x y坐标
//                         - 2）在dst上计算出x、y坐标。然后用M矩阵的逆矩阵，计算出这个像素点变换前，在dst上的x y坐标
//                     - 代码中使用了第二种思路
//             - 为啥要进行插值？插值咋算的？
//                 - 插值是为了让像素值更精准。毕竟计算x y 坐标计算出来是小数
//                 - 熟悉高位、低位概念 
//             - 一个三通道的像素点，地址在cuda中咋算？
//                 - 这个图像的首地址 + position * channels  这是因为前提是cuda保证了内存是连续的
//                 - unsigned char* pdst = dst + position * channels; 
//             - 为啥cuda中用小数一定要加f
//                 - 加不加f直接影响cuda是使用CUDA Core单精度执行还是DPUnit 双精度执行
//                 - 加f就是单精度。CUDA Core数量多，速度快。
//             - opencv利用指针给3通道像素点赋值的特性？
//     */