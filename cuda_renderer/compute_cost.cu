#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cuda_runtime.h>

using std::cout;
using std::endl;
#define SQR(x) ((x)*(x))
#define POW2(x) SQR(x)
#define POW3(x) ((x)*(x)*(x))
#define POW4(x) (POW2(x)*POW2(x))
#define POW7(x) (POW3(x)*POW3(x)*(x))
#define DegToRad(x) ((x)*M_PI/180)
#define RadToDeg(x) ((x)/M_PI*180)

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

// __device__ std::vector<unsigned char> rgb2lab(const unsigned char r,const unsigned char g,const unsigned char b){
//     r = r / 255,
// 	g = g / 255,
// 	b = b / 255,
// 	double x, y, z;

// 	r = (r > 0.04045) ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
// 	g = (g > 0.04045) ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
// 	b = (b > 0.04045) ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

// 	x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
// 	y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000;
// 	z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;

// 	x = (x > 0.008856) ? Math.pow(x, 1/3) : (7.787 * x) + 16/116;
// 	y = (y > 0.008856) ? Math.pow(y, 1/3) : (7.787 * y) + 16/116;
// 	z = (z > 0.008856) ? Math.pow(z, 1/3) : (7.787 * z) + 16/116;
// 	std::vector<unsigned char> lab;
// 	unsigned char l,a,bb;
// 	l = (116 * y) - 16;
// 	a = 500 * (x - y);
// 	bb = 200 * (y - z);
// 	lab.push_back(l);
// 	lab.push_back(a);
// 	lab.push_back(bb);
// 	return lab;
// }


// __device__ double color_distance(const unsigned char l1,const unsigned char a1,const unsigned char b1,
// 	                  const unsigned char l2,const unsigned char a2,const unsigned char b2){
// 	double eps = 1e-5;
//     double c1 = sqrtf(SQR(a1) + SQR(b1));
// 	double c2 = sqrtf(SQR(a2) + SQR(b2));
// 	double meanC = (c1 + c2) / 2.0;
// 	double meanC7 = POW7(meanC);

// 	double g = 0.5*(1 - sqrtf(meanC7 / (meanC7 + 6103515625.))); // 0.5*(1-sqrt(meanC^7/(meanC^7+25^7)))
// 	double a1p = a1 * (1 + g);
// 	double a2p = a2 * (1 + g);

// 	c1 = sqrtf(SQR(a1p) + SQR(b1));
// 	c2 = sqrtf(SQR(a2p) + SQR(b2));
// 	double h1 = fmodf(atan2f(b1, a1p) + 2*M_PI, 2*M_PI);
// 	double h2 = fmodf(atan2f(b2, a2p) + 2*M_PI, 2*M_PI);

// 	// compute deltaL, deltaC, deltaH
// 	double deltaL = l2 - l1;
// 	double deltaC = c2 - c1;
// 	double deltah;

// 	if (c1*c2 < eps) {
// 		deltah = 0;
// 	}
// 	if (std::abs(h2 - h1) <= M_PI) {
// 		deltah = h2 - h1;
// 	}
// 	else if (h2 > h1) {
// 		deltah = h2 - h1 - 2* M_PI;
// 	}
// 	else {
// 		deltah = h2 - h1 + 2 * M_PI;
// 	}

// 	double deltaH = 2 * sqrtf(c1*c2)*sinf(deltah / 2);

// 	// calculate CIEDE2000
// 	double meanL = (l1 + l2) / 2;
// 	meanC = (c1 + c2) / 2.0;
// 	meanC7 = POW7(meanC);
// 	double meanH;

// 	if (c1*c2 < eps) {
// 		meanH = h1 + h2;
// 	}
// 	if (std::abs(h1 - h2) <= M_PI + eps) {
// 		meanH = (h1 + h2) / 2;
// 	}
// 	else if (h1 + h2 < 2*M_PI) {
// 		meanH = (h1 + h2 + 2*M_PI) / 2;
// 	}
// 	else {
// 		meanH = (h1 + h2 - 2*M_PI) / 2;
// 	}

// 	double T = 1
// 		- 0.17*cosf(meanH - DegToRad(30))
// 		+ 0.24*cosf(2 * meanH)
// 		+ 0.32*cosf(3 * meanH + DegToRad(6))
// 		- 0.2*cosf(4 * meanH - DegToRad(63));
// 	double sl = 1 + (0.015*SQR(meanL - 50)) / sqrtf(20 + SQR(meanL - 50));
// 	double sc = 1 + 0.045*meanC;
// 	double sh = 1 + 0.015*meanC*T;
// 	double rc = 2 * sqrtf(meanC7 / (meanC7 + 6103515625.));
// 	double rt = -sinf(DegToRad(60 * expf(-SQR((RadToDeg(meanH) - 275) / 25)))) * rc;

// 	double cur_dist = sqrtf(SQR(deltaL / sl) + SQR(deltaC / sc) + SQR(deltaH / sh) + rt * deltaC / sc * deltaH / sh);
// 	return cur_dist;
// }

// __global__ void bgr_to_gray_kernel( unsigned char* input,
// 									unsigned char* input1,  
// 									unsigned char* output, 
// 									int width,
// 									int height,
// 									int colorWidthStep,
// 									int grayWidthStep)
// {
// 	//2D Index of current thread
// 	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
// 	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
// 	int real_distance;
// 	//Only valid threads perform memory I/O
// 	if((xIndex<width) && (yIndex<height))
// 	{
// 		//Location of colored pixel in input
// 		bool valid = false;
// 		int real_distance;
// 		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);
// 		const unsigned char l1	= input[color_tid];
// 		const unsigned char a1	= input[color_tid + 1];
// 		const unsigned char b1	= input[color_tid + 2];
// 		//Location of gray pixel in output
// 		const int gray_tid  = yIndex * grayWidthStep + xIndex;

		

// 		for(int i = -2; i <3;i++){
// 			int row = yIndex+i;
// 			int col = xIndex+i;
// 			if(row >= 0 && row <height && col >= 0 && col <width){
// 			    const int color_tid_input1 = (row) * colorWidthStep + (3 * col);
// 				const unsigned char l2	= input1[color_tid_input1];
// 				const unsigned char a2	= input1[color_tid_input1 + 1];
// 				const unsigned char b2	= input1[color_tid_input1 + 2];
// 			    double cur_dist=color_distance(l1,a1,b1,l2,a2,b2);
// 			    if(cur_dist < 20){
// 			        valid = true;
// 			    }
// 			    if(i==0){
// 			    	real_distance = cur_dist;
// 			    }
// 			}
// 		}
		
// 		float gray;
// 		if(valid){
// 			gray = 0;
// 		}else{
// 			gray = 1;
// 		}
		

// 		output[gray_tid] = static_cast<unsigned char>(gray);
// 	}
// }
// int *difffilter(const cv::Mat& input,const cv::Mat& input1, cv::Mat& output) 
// {
// 	//Calculate total number of bytes of input and output image
// 	const int colorBytes = input.step * input.rows;
// 	const int grayBytes = output.step * output.rows;

// 	unsigned char *d_input,*d_input1, *d_output;

// 	//Allocate device memory
// 	SAFE_CALL(cudaMalloc<unsigned char>(&d_input,colorBytes),"CUDA Malloc Failed");
// 	SAFE_CALL(cudaMalloc<unsigned char>(&d_input1,colorBytes),"CUDA Malloc Failed");
// 	SAFE_CALL(cudaMalloc<unsigned char>(&d_output,grayBytes),"CUDA Malloc Failed");

// 	//Copy data from OpenCV input image to device memory
// 	SAFE_CALL(cudaMemcpy(d_input,input.ptr(),colorBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
//     SAFE_CALL(cudaMemcpy(d_input1,input1.ptr(),colorBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
// 	//Specify a reasonable block size
// 	const dim3 block(16,16);

// 	//Calculate grid size to cover the whole image
// 	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);

// 	//Launch the color conversion kernel
// 	bgr_to_gray_kernel<<<grid,block>>>(d_input,d_input1,d_output,input.cols,input.rows,input.step,output.step);

// 	//Synchronize to check for any kernel launch errors
// 	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

// 	//Copy back data from destination device meory to OpenCV output image
// 	SAFE_CALL(cudaMemcpy(output.ptr(),d_output,grayBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

// 	//Free the device memory
// 	SAFE_CALL(cudaFree(d_input),"CUDA Free Failed");
// 	SAFE_CALL(cudaFree(d_input1),"CUDA Free Failed");
// 	SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
// 	return 0;
// }