// /*
// 实现一个TensorRTCNN推理，模型师手动构建方式
// 预期，网络具有4个节点
// input节点，为输入
// conv1节点，输入为input
// relu1为激活，输入为conv1的输出
// relu1的输出，为输出节点output
// */

// #include <cuda_runtime.h>
// // 8版本开始，似乎不用这个了
// #include <NvInfer.h>   
// #include <stdio.h>
// #include <string.h>
// #include <iostream>
// #include <vector>
// using namespace std;
// //第一步::定义Logger类
//     //类名随便取，关键是继承自nvinfer1下的ILogger
//     //I 是interface 接口。 该设计模式属于接口模式
//     //
//     //TensorRT内部出现任何消息，都会通过Logger类打印出来
//     //根据消息等级区分问题，有助于排查bug

// // 纯虚函数：纯虚函数语法：virtual 返回值类型 函数名 （参数列表）= 0 ;
//     // 1.具有纯虚函数的类，不能够被实例化
//     // 2.纯虚函数是只有声明，没有实现的虚函数。
//     // 3.作用是，神宫韩淑，实现交给其子类完成
//     // 4.如果子类继承的基类中存在纯虚函数，则必须实现纯虚函数
// class JLogger : public nvinfer1::ILogger{
// public:
//     virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override 
//     //注意此虚函数重写一定AsciiChar前要增加nvinfer1::
//     //severity就是消息级别
//     {
//             printf("==LOG[%d]==:%s\n", static_cast<int>(severity), msg);
//     }
// };

// bool save_to_file(const string& file, const void* data, size_t size){
//     FILE* handle = fopen(file.c_str(), "wb");
//     if(handle == nullptr){
//         return false;
//     }
//     fwrite(data, 1, size, handle);
//     fclose(handle);
//     printf("写入成功");
//     return true;
// }

// //读取序列化后的文件，数据因为是vector传的。所以要用到vector
// vector<unsigned char> load_from_file(const string& file){
//     FILE* handle = fopen(file.c_str(), "rb");
//     if(handle == nullptr){
//         return {};
//     }

//     //1.获取文件大小，并分配内存
//     //SEEK_END， 移动文件指针到相对末尾的位置。
//     fseek(handle, 0, SEEK_END);  //第二个参数0表示：指针移动到相对于末尾0的位置。

//     //2.获取当前文件指针位置，得到大小
//     long size = ftell(handle);

//     //3.恢复文件指针到文件开头
//     //直接设置文件指针到特定位置，这里是0，就是头文件了
//     fseek(handle, 0, SEEK_SET);

//     vector<unsigned char> output;

//     //4.如果文件长度大于0，才有必要读取和分配
//     if(size > 0){
//         output.resize(size);
//         fread(output.data(), 1, size, handle);
//     }

//     //5.关闭文件句柄，返回结果
//     fclose(handle);
//     return output;
// }

// string format_dim(const nvinfer1::Dims& dims){
//     char buffer[100] = {0};
//     char *p = buffer;
//     for (int i = 0; i < dims.nbDims; ++i)
//     {
//         p += sprintf(p, i == 0 ? "%d" : " x %d", dims.d[i]);
//         //sprintf(buffer, format, __args__) 第三个参数是变参-》返回字符串buffer的长度
//         /*
//         前提提要：
//         1）p在这里指指针、或者说数组首地址。p如果++ 或者 P+=1 表示指针移动到下一个位置
//         2）sprintf返回值-1，表示写入错误。    写入正确时，返回正确写入的字符数
//         3）sprintf的主要功能是把数据格式化后(把%d这类的替换好)，写入p中
//         4) sprintf每次默认从头写入字符串。p += sprintf就是相当于把写入模式，改为追加模式。

//         原理:
//             1-1,当i = 0 时，假设第0维度数dims.d[0] = 1， 则把字符串  "1" 写入 p中
//                 此时sprintf 返回 "1" 的字符数计数  即  int类型 1
//             1-2.指针移动1

//             2-1.当i = 1 时，假设第1维度数dims.d[1] = 3， 则把字符串  " x 3" 写入 p中
//                 此时sprintf 返回 " x 3" 的字符数计数  即  int类型 4
//             2-2.指针继续移动4次

//             不断循环
//         总结就是，指针p始终指向数组 buffer数组中，最后一个有数据地方的地址
//         */
//     }
//     return buffer;
// }

// int model_build(){
//     //第二步::实例化Logger类
//     JLogger logger;

//     //第三步::构建模型编译器
//         //- 创建网络
//         //- 创建编译配置
//     nvinfer1::IBuilder*         builder = nvinfer1::createInferBuilder(logger);
//     nvinfer1::INetworkDefinition*   network = builder -> createNetworkV2(1);
//     nvinfer1::IBuilderConfig*       config  = builder -> createBuilderConfig();

//     //配置网络参数
//     //配置最大的batchsize，移位置推理所制定的batch参数不能超过这个
//     builder->setMaxBatchSize(1);

//     //配置工作空间的大小
//     //每个节点不用自己管理内存空间吗不用自己去Malloc
//     //使得所有节点均把workspace当做内存池，重复使用，是的内存
//     //更加紧凑，高效
//     config->setMaxWorkspaceSize( 1 << 30);  //1 左移 30

//     // builder->platformHasFastFp16(); //这个函数告诉你，当前显卡是否具有FP16的加速能力
//     // builder->platformHasFastInt8();//这个函数告诉你，当前显卡是否具有int8的加速能力
//     // config->setFlag(nvinfer1::BuilderFlag::kFP16); //程序默认使用FP32推理。如果希望使用FP16可以这么设置
//     // ！！！如果使用int8，则需要做精度标定。这个属于模型量化的内容，把权重变为int8格式，计算乘法(相当于两个整数相乘变成一个长一点的整数)。
//         //减少浮点数乘法操作，用整数替代(整数速度快)。
//         //剪枝、蒸馏、量化三大技术

//         //int8的简单介绍。大致就是把数以万计的权重，从浮点数变成int8。即0-255的数。
//             //首先要权重归一化。然后最大值 到 最小值 划分255份。把数据放大到0-255区间
//             //但是精度会损失很大。所以需要标定。即统计原来数据的分布再划分？？
//     printf("Workspace Size = %.2f MB\n", (1 << 30) / 1024.0f / 1024.0f); // Mib


//     //第四步::构建网络结构，并赋值权重
//     //4.0 定义卷积的核权重
//     float kernel_weight[] = {
//         1, 0, 0,
//         0, 1, 0,
//         0, 0, 1
//     };

//     nvinfer1::Weights conv1_weight;
//     nvinfer1::Weights conv1_no_bias;
//     //参数的需要赋值的三个参数  count   type    values
//     conv1_weight.count = sizeof(kernel_weight) / sizeof(kernel_weight[0]);
//     conv1_weight.type = nvinfer1::DataType::kFLOAT;
//     conv1_weight.values = kernel_weight;
//     conv1_no_bias.count = 0;
//     conv1_no_bias.type = nvinfer1::DataType::kFLOAT;
//     conv1_no_bias.values = nullptr;

//     //4.1 确定input
//     nvinfer1::ITensor* input = network -> addInput(
//         "Image",                        //输入节点的名称
//         nvinfer1::DataType::kFLOAT,     //输入节点的数据类型
//         nvinfer1::Dims4(1, 1, 3, 3));      //输入节点的shape大小

//     //4.2 卷积 
//     nvinfer1::IConvolutionLayer* conv1 = network -> addConvolution(
//         *input,                        //输入节点的tensor，需要提供引用
//         1,                              //指定输出通道数
//         nvinfer1::DimsHW(3, 3),         //指定卷积核的大小
//         conv1_weight,                   //指定卷积核的参数
//         conv1_no_bias);                 //偏置的参数，这里没有设置偏置

//     //4.2.1 设置卷积  名字、步长、填充、膨胀系数
//     conv1->setName("Conv1");
//     conv1->setStride(nvinfer1::DimsHW(1, 1));
//     conv1->setPadding(nvinfer1::DimsHW(0, 0));
//     conv1->setDilation(nvinfer1::DimsHW(1, 1));
    
//     //4.3 激活函数
//     nvinfer1::IActivationLayer* relu1 = network -> addActivation(
//         *(conv1->getOutput(0)), 
//         nvinfer1::ActivationType::kRELU);
//         //4.3.1
//     relu1->setName("ReLU1");

//     //4.4 网络输出节点设置
//     nvinfer1::ITensor* output = relu1->getOutput(0);    //获取relu的输出
//     output -> setName("Predict");                       //设置输出的节点的名字
//     network -> markOutput(*output);                     //告诉网络，这个节点是输出节点，值会被保留

//     //第五步::使用构建好的网络编译引擎   builder使用前两个创建的network config创建的第三个对象engine。
//         //什么是引擎：引擎存储的是当前环境下，最佳的运算方案。参数、运行的选择、算法的定义。
//     nvinfer1::ICudaEngine* engine = builder -> buildEngineWithConfig(*network, *config);  
    
//     //第六步::序列化模型为数据，并储存为文件
//     nvinfer1::IHostMemory* host_memory = engine->serialize();
//     printf("准备输出成文件04.cnn.trtmodel， host_memory->data()为：%sm  size()为%d\n", (const char*)host_memory->data(), host_memory->size());
//     save_to_file("04.cnn.trtmodel", host_memory->data(), host_memory->size());

//     //第七步::回收内存，清理内存，打印一个消息，说我们搞定了
//     host_memory->destroy();
//     engine->destroy();
//     config->destroy();
//     network->destroy();
//     builder->destroy();
//     printf("Done\n");
// }

// void model_inference(){
//     JLogger logger;
//     cudaStream_t stream = nullptr;
//     cudaEvent_t start, stop;

//     //第一步:: 设备推理用的设备，创建流
//     cudaSetDevice(0);
//     cudaStreamCreate(&stream);
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start, stream);

//     //第二步::加载模型
//     auto model_data = load_from_file("04.cnn.trtmodel");
//     if(model_data.empty()){
//         printf("Load model failure.\n");
//         return;
//     }
//     //第三步:: 创建运行时引擎对象，并反序列化得到引擎。
//     //通过引擎，创建执行上下文
//     nvinfer1::IRuntime*         runtime = nvinfer1::createInferRuntime(logger); //？？？？
//     nvinfer1::ICudaEngine*          engine  = runtime->deserializeCudaEngine(model_data.data(), model_data.size());
//     nvinfer1::IExecutionContext*    context = engine->createExecutionContext();

//     //第四步::获取绑定的tensor信息，并打印出来
//     //所谓绑定的tensor，就是指输入和输出节点
//     int nbindings = engine->getNbBindings();

//     printf("nbindings = %dd\n", nbindings);
//     for (int i = 0; i < nbindings; ++i)
//     {
//         auto dims = engine->getBindingDimensions(i);
//         auto name = engine->getBindingName(i);
//         auto type = engine->getBindingDataType(i); //???
//         printf("Binding %d [%s], dimension is %s, type is %s\n",
//         i,
//         name,
//         format_dim(dims).c_str(),
//         type);
//     }
    
//     //第五步::准备输入和输出数据内存
//     nvinfer1::Dims output_dims = engine->getBindingDimensions(1); //因为这个案例，binding里面有两个节点。第一个0是input  第二个1是output
//     float* output_device = nullptr;
//     cudaMalloc(&output_device, sizeof(float));

//     float input_data[] = {
//         1, 2, 3, 
//         4, 5, 6, 
//         7, 8, 9
//     };

//     float* input_device = nullptr;
//     cudaMalloc(&input_device, sizeof(input_data));
//     cudaMemcpyAsync(input_device, input_data, sizeof(input_data), cudaMemcpyHostToDevice, stream);

//     //这里的bindings设备指针的顺序，必须与bindings的索引相对应
//     void* bindings_device_pointer[] = {input_device, output_device};

//     //第六步::入队并进行推理,得到结果
//     //6.1 入队
//     bool finished = context->enqueue(1, bindings_device_pointer, stream, nullptr);
//     //最后一个参数表示，如果设备中的input_device用完了，你改它不会影响结果了，给你发个信号，让你可以修改或者做其他操作。
//         //当前是没有设置第四个参数的。只能等到 同步 后，再修改。否则没有同步就修改，会导致结果未知。
//     if(!finished){
//         printf("Enqueue failure.\n");
//     }

//     //6.2收集执行结果
//     float output_data = 0;
//     cudaMemcpyAsync(&output_data, output_device, sizeof(float), cudaMemcpyDeviceToHost, stream);

//     //同步流，等待执行完成
//     cudaEventRecord(stop, stream);
//     cudaEventSynchronize(stop);
//     float ms = 0;
//     cudaEventElapsedTime(&ms, start, stop);
//     // cudaStreamSynchronize(stream);

//     //第七步::打印最后的结果
//     printf("最后的结果 output_value = %f, 执行时间：=%f\n", output_data, ms);

//     /*
//     cuda出来的结果出现在前置后置：
//         例如，   前置主要是图像的预处理，会使用cuda
//                 后置主要指计算结果出来后，筛选置信度等操作使用cuda
//     */

//    //第八部::释放内存
//    cudaFree(input_device);
//    cudaFree(output_device);
//    context->destroy();
//    engine->destroy();
//    runtime->destroy();
//    cudaStreamDestroy(stream);
// }

// int main(){
//     model_build();
//     model_inference();
//     return 0;
// }
