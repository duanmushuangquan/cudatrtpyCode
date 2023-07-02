// #include <NvInfer.h>
// #include <NvInferRuntime.h>
// #include <stdio.h>
// #include <onnx-tensorrt-8.0-GA/NvOnnxParser.h>
// #include <fstream>
// #include <vector>
// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include <opencv2/core/core.hpp>
// #include <cuda.h>
// #include <cuda_runtime.h>

// #include <opencv2/dnn/dnn.hpp>

// using namespace std;
// using namespace cv;

// inline const char* severity_string(nvinfer1::ILogger::Severity t){
//     switch(t){
//         case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
//         case nvinfer1::ILogger::Severity::kERROR:   return "error";
//         case nvinfer1::ILogger::Severity::kWARNING: return "warning";
//         case nvinfer1::ILogger::Severity::kINFO:    return "info";
//         case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
//         default: return "unknow";
//     }
// }

// class TRTLogger : public nvinfer1::ILogger{
// public:
//     virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
//         if(severity <= Severity::kINFO){
//             // 打印带颜色的字符，格式如下：
//             // printf("\033[47;33m打印的文本\033[0m");
//             // 其中 \033[ 是起始标记
//             //      47    是背景颜色
//             //      ;     分隔符
//             //      33    文字颜色
//             //      m     开始标记结束
//             //      \033[0m 是终止标记
//             // 其中背景颜色或者文字颜色可不写
//             // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
//             if(severity == Severity::kWARNING){
//                 printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
//             }
//             else if(severity <= Severity::kERROR){
//                 printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
//             }
//             else{
//                 printf("%s: %s\n", severity_string(severity), msg);
//             }
//         }
//     }
// } logger;

// bool save_engine_data_binary_file(const char* file_name, const void* data, size_t size){
//     ofstream ofs(file_name, ios::out | ios::binary);
//     if (!ofs)
//     {
//         printf("\033[31mOpen %s Faliure\033[0m\n", file_name);
//         return false;
//     }

//     ofs.write((const char*)data, size);
//     ofs.close();
//     return true;
// }

// vector<char> load_engine_data_binary_file(const char* file_name){
//     ifstream ifs(file_name, ios::in | ios::binary);
//     if (!ifs)
//     {
//         printf("\033[31mOpen %s Faliure\033[0m\n", file_name);
//         return {};
//     }
//     std::vector<char> data;

//     ifs.seekg(0, ios::end);
//     size_t size = ifs.tellg();
//     ifs.seekg(0, ios::beg);

//     data.resize(size);

//     ifs.read(data.data(), size);
//     if (!ifs)
//     {
//         printf("\033[31m Read %s Faliure in Line[%d]\033[0m\n", file_name, __LINE__);
//         ifs.close();
//         return {};
//     }
//     ifs.close();
//     return data;  
// }

// void load_onnx_and_parser(){
//     TRTLogger mylogger;
//     nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(mylogger);
//     nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);
//     nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
//     builder->setMaxBatchSize(1);
//     config->setMaxWorkspaceSize( 1 << 30);
//     // config->setFlag(nvinfer1::BuilderFlag::kFP16);  //是否使用FP16

//     const char* onnx_file_name = "resnet18.onnx";
//     nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, mylogger);

//     if (!parser->parseFromFile(onnx_file_name, 1))
//     {
//         printf("\033[31m] Failed to parseFromFile \033[0m\n]");
//         parser->destroy();
//         config->destroy();
//         network->destroy();
//         builder->destroy();
//         return;
//     }

//     printf("\033[33m] Sucess \033[0m\n]");

//     nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
//     if (engine == nullptr)
//     {
//         printf("\033[31m] No data in engine_data \033[0m\n]");
//     }
//     nvinfer1::IHostMemory* engine_data = engine->serialize();

//     save_engine_data_binary_file("04.renetonnx.trtmodel.data", (char*)engine_data->data(), engine_data->size());

//     engine_data->destroy();
//     engine->destroy();
//     parser->destroy();
//     config->destroy();
//     network->destroy();
//     builder->destroy();
//     printf("Done");
// }

// string dims_print(const nvinfer1::Dims& dims){
//     char buffer[100] = {0};
//     char *p = buffer;
//     for (int i = 0; i < dims.nbDims; ++i)
//     {
//         p += sprintf(p, i == 0 ? "%d" : " x %d", dims.d[i]);
//     }
//     return buffer;
// }

// void model_inference(){
//     TRTLogger mylogger;
//     const char* file_name = "04.renetonnx.trtmodel.data";
//     auto engine_vector = load_engine_data_binary_file(file_name);
//     if (engine_vector.empty())
//     {
//         printf("\033[31m Load %s Failure in Line[%d] \033[0m\n", file_name, __LINE__);
//         return;
//     }

//     nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(mylogger);
//     nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_vector.data(), engine_vector.size());
//     nvinfer1::IExecutionContext* context = engine->createExecutionContext();

//     int a = engine->getNbBindings();
//     for (int i = 0; i < a; ++i)
//     {
//         auto name = engine->getBindingName(i);
//         auto type = engine->getBindingDataType(i);
//         auto dims = engine->getBindingDimensions(i);
//         string s = dims_print(dims);
//         printf("\033[33mgetBindingName(%d)\tname=%s\t dims=%s\033[0m\n", i, name, s.c_str());
//     }

//     Mat image = cv::imread("kj.jpg");
//     cv::resize(image, image, Size(224, 224));
//     cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    
//     //均值标准差
//     Scalar mean(0.485, 0.456, 0.406);
//     Scalar std(0.229, 0.224, 0.225);

//     // 转换Scalar到Mat
//     cv::Mat mean_mat = cv::Mat(224, 224, CV_32FC3, mean);
//     cv::Mat std_mat = cv::Mat(224, 224, CV_32FC3, std);

//     //uint8转换iamge到浮点数，归一化
//     Mat image_float;
//     image.convertTo(image_float, CV_32F, 1 / 255.0f);//此操作没有影响通道数
//     image_float = (image_float - mean_mat) / std_mat;
//     //老师的课程里   image_float = (image_float - mean) / std; 是可以的。我这里不知道为啥不行。

//     //image的像素排列此刻是rgbrgbrgb
//     //但是需要的网络输入像素排列格式是? 
//         // batch  channel  height  width --> [N, 3, W, H]
//         // 假设有4个像素，channel=3，此时网络要求的像素格式是  rrrrggggbbbb?
//         // 
//     // cv::split()
//     // cout << mean_mat.at<float>(0, 0) << endl; //0.485
//     // cout << image_float.at<float>(0, 0) << endl; //-0.936296
//     vector<float> input_host_image_vector(image.cols * image.rows * image.channels()); 

//     //让下面3个通道，分别引用input_host_image_vector的地址。
//     Mat input_image_by_changed_to_rrrbbbggg_array[3];//经过rbgrbg变成rrrbbbggg的图片Mat数组
//     float* input_host_image_vector_ptr = input_host_image_vector.data();
//     for (int i = 0; i < 3; ++i)
//     {
//         /*
//         这个是新建一个Mat，分配内存大小是image1.rows * image1.cols
//         Mat r_reference = Mat(image1.rows, image1.cols, CV_32F);

//         这个是新建一个Mat，不新分配内存。而是引用用户自定义的  地址，以这个地址为首地址。
//         他的值的地址范围为  [image_vector.data() + r_channel_start_pointer_position,  image_vector.data() + r_channel_start_pointer_position + area]
//         Mat r_reference = Mat(image1.rows, image1.cols, CV_32F，image_vector.data() + r_channel_start_pointer_position);
//         */
//         input_image_by_changed_to_rrrbbbggg_array[i] = Mat(image_float.rows, image_float.cols, CV_32F, input_host_image_vector_ptr);
//         input_host_image_vector_ptr += image_float.rows * image_float.cols;//每次指针移动一个通道像素个数个位置
//     }
//     /*
//     在创建一个新的input_host_image_vector向量之后，你使用一个指针input_host_image_vector_ptr引用它的数据。
//     然后，你创建了一个图像数组input_image_by_changed_to_rrrbbbggg，
//     每个元素都是一个指向input_host_image_vector中特定位置的Mat对象。
//     这样做的目的是将原始的RGBRGBRGB格式的图像转换为RRRGGGBBB格式，以符合网络输入的要求。
//     */

//     cv::split(image_float, input_image_by_changed_to_rrrbbbggg_array); //修改input_image_by_changed_to_rrrbbbggg_array的同时。input_host_image_vector也被改了
//     //split必须保证image_float 与 input_image_by_changed_to_rrrbbbggg_array类型、大小通道完全一致
//     //否则会导致input_image_by_changed_to_rrrbbbggg的内存重新分配。也就是断开了引用关系，导致结果与预期不符
//     /*使用的是这个重载。将image_float，按照通道，分开，到input_image_by_changed_to_rrrbbbggg的首地址
//     void cv::split(const cv::Mat &src, cv::Mat *mvbegin)
//     +3 overloads
//     ! copies each plane of a multi-channel array to a dedicated array

//     在你的代码中，cv::split(image_float, input_image_by_changed_to_rrrbbbggg); 
//     这一行的作用就是将浮点型图像image_float的通道分离开来，并将结果存储在input_image_by_changed_to_rrrbbbggg中。
//     注意，cv::split()函数会根据image_float的通道数来分割，也就是说，如果image_float是一个三通道的图像（RGB图像），
//     那么input_image_by_changed_to_rrrbbbggg就会是一个包含了三个单通道图像（R、G、B）的数组。每一个元素对应原图像的一个通道，
//     例如input_image_by_changed_to_rrrbbbggg[0]对应的就是image_float的第一个通道。
//     这样做的目的是将原始的RGBRGBRGB格式的图像转换为RRRGGGBBB格式，以符合网络输入的要求。

//     个人理解不知道对不对：想象卷积的图。输入的图片RGB是分开的三个chennl。而3通道Mat的数据是Scalar(255, 255, 255)
//         用at取值是Vec3f。这些都我认为是一个整体的数据类型。就类似只有一个通道，而通道的像素值是Scalar或者说Vec3f
//         深度学习模型是读不懂的。一定要拆成三个通道。每个通道的像素值明确的就是float。才能输入网络计算。
//     */
//     //断言，如果image_float 与 input_image_by_changed_to_rrrbbbggg_array类型、大小通道完全一致。不执行
//     //否则特出错误
//     CV_Assert((void*)input_image_by_changed_to_rrrbbbggg_array[0].data == (void*)input_host_image_vector.data()); 

//     //cuda的设置设备和初始化
//     int device_num = 0;
//     cudaSetDevice(device_num);
    
//     //设置流
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);

//     //设置input_device数据
//     float* input_device_image = nullptr;
//     size_t input_device_bytes = sizeof(float) * input_host_image_vector.size();
//     cudaMalloc(&input_device_image, input_device_bytes);
//     cudaMemcpyAsync(input_device_image, input_host_image_vector.data(), input_device_bytes, cudaMemcpyHostToDevice, stream);

//     //设置output_device数据
//     float* output_device_image = nullptr;
//     size_t output_bytes = sizeof(float) * 1000;
//     cudaMalloc(&output_device_image, output_bytes);//输出数据大小，看数据集。这个数据集就是1000个输出

//     //入队、执行网络推理
//     void* bindings[] = {input_device_image, output_device_image};
//     context->enqueueV2(bindings, stream, nullptr); ////???
//     // context->enqueue(1, bindings, stream, nullptr); 

//     //异步 赋值数据
//     vector<float> output_host_predict(1000);
//     cudaMemcpyAsync(output_host_predict.data(), output_device_image, output_bytes, cudaMemcpyDeviceToHost, stream);

//     //同步流，查看有没有错误
//     cudaStreamSynchronize(stream);
//     cudaError_t code = cudaPeekAtLastError();
//     if(!code == cudaSuccess){
//         printf("\033[31m Failed inferrence or wrong output in cuda LINE[%d] \033[0m\n", __LINE__);
//     }

//     //softmax转换为概率，这里提到了sigmoid上下溢的问题
//     float sum;
//     for(float& item : output_host_predict){
//         sum += exp(item);
//     }

//     for (float& item : output_host_predict)
//     {
//         item = exp(item) / sum;
//     }

//     int label = std::max_element(output_host_predict.begin(), output_host_predict.end()) - output_host_predict.begin();
//     float confidence = output_host_predict[label];
//     printf("\033[33m result:= label=%d,  confidence=%f \033[0m\n", label, confidence);

//     cudaFree(output_device_image);
//     cudaFree(input_device_image);
//     cudaStreamDestroy(stream);
//     context->destroy();
//     engine->destroy();
//     runtime->destroy();
// }

// int main()
// {
//     load_onnx_and_parser();
//     model_inference();

//     return 0;
// }