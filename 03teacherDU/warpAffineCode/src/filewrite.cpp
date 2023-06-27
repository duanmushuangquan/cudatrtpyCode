// #include <NvInfer.h>
// #include <iostream>
// #include <stdio.h>
// #include <vector>
// #include <fstream>
// #include <cuda.h>
// #include <cuda_runtime.h>
// using namespace std;

// class Logger : public nvinfer1::ILogger{
// public:
//     virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
//         printf("==Log[%d]==, msg:=%s\n", severity, msg);
//     }
// };

// vector<unsigned char> load_from_file(const string &file_name){
//     ifstream ifs(file_name, ios::in | ios::binary);
//     if (!ifs.is_open())
//     {
//         printf("!!!在第%d行发现错误，文件没能成功读取\n", __LINE__);
//         return {};
//     }
//     vector<unsigned char> vector_data;

//     ifs.seekg(0, ios::end); //指针移动到举例末尾偏移0的位置(就是末尾的位置)
//     size_t length_of_data = ifs.tellg();
//     if (length_of_data > 0)
//     {
//         vector_data.resize(length_of_data);
//         ifs.seekg(0, ios::beg);
//     }

//     ifs.read((char*)&vector_data[0], length_of_data);
//     ifs.close();
//     return vector_data;   
// }

// string format_Dims(nvinfer1::Dims& dims){
//     char buffer[100] = {0};
//     int dims_len = dims.nbDims;
//     char* p = buffer;
//     for (int i = 0; i < dims_len; ++i)
//     {
//         p += sprintf(p, i == 0 ? "%d" : " x %d", dims.d[i]);
//     }
//     return buffer;
// }

// int main(){
//     Logger mylogger;

//     int cuda_device = 0;
//     cudaSetDevice(cuda_device);
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);

//     string file_name = "04.cnn.trtmodel";
//     vector<unsigned char> vector_data = load_from_file(file_name);

//     nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(mylogger);
//     nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(vector_data.data(), vector_data.size());
    
//     nvinfer1::IExecutionContext* context = engine->createExecutionContext();
//     size_t num_bindings = engine->getNbBindings();

//     for (int i = 0; i < num_bindings; i++)
//     {
//         nvinfer1::Dims Dims = engine->getBindingDimensions(i);
//         auto name = engine->getBindingName(i);
//         auto type = engine->getBindingDataType(i);

//         printf("当前为第%d个bingding，维度：=%s, 名字：=%s， 类型为:=%s\n", i, format_Dims(Dims).c_str(), name, type);
//     }

//     float input_host[] = {
//         1, 2, 3,
//         4, 5, 6,
//         7, 8, 9
//     };

//     float* input_device = nullptr;
//     float* output_device = nullptr;
//     cudaMalloc(&input_device, sizeof(input_host));
//     cudaMalloc(&output_device, sizeof(float));
//     cudaMemcpyAsync(input_device, input_host, sizeof(input_host), cudaMemcpyHostToDevice, stream);
//     void* binding_array[] = {input_device, output_device};
//     bool issuccess = context->enqueue(1, binding_array, stream, nullptr);
//     if (!issuccess)
//     {
//         printf("计算失败！！！！！！\n");
//     }

//     float output_host = 0;
//     cudaMemcpyAsync(&output_host, output_device, sizeof(float), cudaMemcpyDeviceToHost, stream);
//     cudaStreamSynchronize(stream);

//     printf("出结果了！！ 结果为 %f \n", output_host);

//     cudaFree(output_device);
//     cudaFree(input_device);
//     context->destroy();
//     engine->destroy();
//     runtime->destroy();
//     cudaStreamDestroy(stream);

//     return 0;
// }

#include <stdio.h>

void foo();

int main() {

	foo();
	return 0;
}