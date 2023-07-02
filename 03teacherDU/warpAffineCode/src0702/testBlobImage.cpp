// #include <opencv2/opencv.hpp>
// #include <opencv2/dnn/dnn.hpp>
// #include <stdio.h>
// #include <vector>

// using namespace std;
// using namespace cv;

// void pixel_visit_demo_pointer(Mat& image) {
// 	int height = image.rows;
// 	int width = image.cols;
// 	int dims = image.channels();

// 	for (int row = 0; row < height; row++)
// 	{
// 		//得到该行的第一个像素店的指针
// 		uchar* row_pointer = image.ptr<uchar>(row);
// 		for (int col = 0; col < width; col++)
// 		{
// 			if (dims == 1)
// 			{
// 				//得到改行的第一个像素点的值
// 				int pixel_value = *row_pointer;

// 				//翻转第一个像素点的值
// 				*(row_pointer++) = 255 - pixel_value;
// 			}

// 			if (dims == 3)
// 			{
// 				*(row_pointer++) = 255 - *row_pointer;
// 				*(row_pointer++) = 255 - *row_pointer;
// 				*(row_pointer++) = 255 - *row_pointer;
// 			}
// 		}
// 	}
// 	cout << "dims=" << dims << endl;
// }

// int main(){

//  	//=================第一种写法============================
// 	////创建一个3通道的图片。
// 	//Mat image1 = Mat(4, 4, CV_32FC3);

// 	////计算面积与总共的像素值
// 	//int area = image1.rows * image1.cols;
// 	//int total = area * image1.channels();

// 	////创建一个vector，也可以是一个float数组。用来开辟空间
// 	////开辟空间的大小是总共像素点的大小。  内存大小是 像素点数量  乘以 sizeof(float)
// 	////这个空间用来给下面的  Mat数组  image_array  用。
// 	//vector<float> image_vector(total);
// 	//float* vector_ptr = image_vector.data(); //拿到这个vector的首地址

// 	//image1 = Scalar(0.1, 0.5, 0.9); //创建三通道的单个像素点的标量

// 	//Mat image_array[3]; //创建Mat数组，主要是给split函数用，用于将Image1 的三个通道分离开。此时该数组没有分配内存

// 	//for (int i = 0; i < 3; i++)
// 	//{
// 	//	image_array[i] = Mat(image1.rows, image1.cols, CV_32F, vector_ptr);
// 	//	//i=0。让image_array[0]的地址是vector_ptr的首地址
// 	//	//i=1。让image_array[1]的地址是vector_ptr的首地址偏移area个单位后的地址
// 	//	//i=2。让image_array[2]的地址是vector_ptr的首地址偏移两个area 个单位后的地址
// 	//	vector_ptr += area;
// 	//}

// 	//cv::split(image1, image_array);
// 	//// 经过循环此时image_array有内存空间了。大小正好可以将image1按照通道拆分后，
// 	////放到image_array[0]  image_array[1]   image_array[2]中

// 	//cout << image_array[0] << endl;
// 	//cout << image_array[1] << endl;
// 	//cout << image_array[2] << endl;
// 	//=================第一种写法============================


// 	//=================第二种写法============================
// 	////创建一个3通道的图片。
// 	//Mat image2 = Mat(4, 4, CV_32FC3);
// 	//image2 = Scalar(0.1, 0.5, 0.9); //创建三通道的单个像素点的标量
// 	////下面验证 blobFromImage
// 	//Mat blob2 = cv::dnn::blobFromImage(image2, 1.0, Size(4, 4), Scalar(0, 0, 0), false, false);
// 	//int num = blob2.size[0];      // batch size
// 	//int channels = blob2.size[1]; // number of channels
// 	//int height = blob2.size[2];   // height of the image
// 	//int width = blob2.size[3];    // width of the image
// 	//cout << blob2.size() << endl;
// 	//for (int i = 0; i < channels; i++)
// 	//{
// 	//	for (int h = 0; h < height; h++)
// 	//	{
// 	//		for (int w = 0; w < width; w++)
// 	//		{
// 	//			cout << blob2.ptr<float>(0, i, h)[w] << " ";
// 	//		}
// 	//		cout << endl;
// 	//	}
// 	//}
// 	//cout << blob2.total() << endl;
// 	//=================第二种写法============================

// 	//=================第三种写法============================
// 	//创建一个3通道的图片。
// 	Mat image1 = Mat(4, 4, CV_32FC3);

// 	//计算面积与总共的像素值
// 	int area = image1.rows * image1.cols;
// 	int total = area * image1.channels();

// 	//新建vector，用来储存所有像素值。但是前area个值是R通道的值，第area 到 2*area个值是B通道的值。。
// 	vector<float> image_vector(total);
// 	image1 = Scalar(0.1, 0.5, 0.9); //创建三通道的单个像素点的标量

// 	//新建一个Mat，使其成为引用关系
// 	/*
// 	这个是新建一个Mat，分配内存大小是image1.rows * image1.cols
// 	Mat r_reference = Mat(image1.rows, image1.cols, CV_32F);

// 	这个是新建一个Mat，不新分配内存。而是引用用户自定义的  地址，以这个地址为首地址。
// 	他的值的地址范围为  [image_vector.data() + r_channel_start_pointer_position,  image_vector.data() + r_channel_start_pointer_position + area]
// 	Mat r_reference = Mat(image1.rows, image1.cols, CV_32F，image_vector.data() + r_channel_start_pointer_position);
// 	*/
// 	int r_channel_start_pointer_position = 0 * image1.rows * image1.cols;
// 	int g_channel_start_pointer_position = 1 * image1.rows * image1.cols;
// 	int b_channel_start_pointer_position = 2 * image1.rows * image1.cols;
// 	Mat r_reference = Mat(image1.rows, image1.cols, CV_32F, image_vector.data() + r_channel_start_pointer_position);
// 	Mat g_reference = Mat(image1.rows, image1.cols, CV_32F, image_vector.data() + g_channel_start_pointer_position);
// 	Mat b_reference = Mat(image1.rows, image1.cols, CV_32F, image_vector.data() + b_channel_start_pointer_position);

// 	//行排列优先，所以一般先循环y  先循环行
// 	for (int y = 0; y < image1.rows; ++y)
// 	{
// 		for (int x = 0; x < image1.cols; ++x)
// 		{
// 			const Vec3f& rgb_pixel_value = image1.at<Vec3f>(y, x);
// 			cout << rgb_pixel_value << endl;
// 			float R = rgb_pixel_value[0];
// 			float G = rgb_pixel_value[1];
// 			float B = rgb_pixel_value[2];
//             /*复杂写法
//             int position = y * image1.cols + x;
// 			int rpos = position + r_channel_start_pointer_position;
// 			int gpos = position + g_channel_start_pointer_position;
// 			int bpos = position + b_channel_start_pointer_position;

// 			image_vector[rpos] = R;
// 			image_vector[gpos] = G;
// 			image_vector[bpos] = B;
//             */
//             r_reference.at<float>(y, x) = R;
//             g_reference.at<float>(y, x) = G;
//             g_reference.at<float>(y, x) = B;
// 		}
// 	}

// 	for (vector<float>::const_iterator cit= image_vector.begin();cit!=image_vector.end();cit++)
// 	{
// 		cout << *cit << " ";
// 	}
// 	//=================第三种写法============================

//     return 0;

// }
