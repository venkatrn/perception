#include "cuda_renderer/renderer.h"
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#ifdef CUDA_ON
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#endif

using namespace cv;

static std::string prefix = "/home/jessy/pose_refine/test/";

namespace helper {
cv::Mat view_dep(cv::Mat dep){
    cv::Mat map = dep;
    double min;
    double max;
    cv::minMaxIdx(map, &min, &max);
    cv::Mat adjMap;
    map.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min);
    cv::Mat falseColorsMap;
    applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_HOT);
    return falseColorsMap;
};

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s\n" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};
}
// void rgb2lab(uint8_t rr,uint8_t gg, uint8_t bbb, float* lab){
//     double r = rr / 255.0;
//     double g = gg / 255.0;
//     double b = bbb / 255.0;
//     double x;
//     double y;
//     double z;
//     r = ((r > 0.04045) ? pow((r + 0.055) / 1.055, 2.4) : (r / 12.92)) * 100.0;
//     g = ((g > 0.04045) ? pow((g + 0.055) / 1.055, 2.4) : (g / 12.92)) * 100.0;
//     b = ((b > 0.04045) ? pow((b + 0.055) / 1.055, 2.4) : (b / 12.92)) * 100.0;

//     x = r*0.4124564 + g*0.3575761 + b*0.1804375;
//     y = r*0.2126729 + g*0.7151522 + b*0.0721750;
//     z = r*0.0193339 + g*0.1191920 + b*0.9503041;

//     x = x / 95.047;
//     y = y / 100.00;
//     z = z / 108.883;

//     x = (x > 0.008856) ? cbrt(x) : (7.787 * x + 16.0 / 116.0);
//     y = (y > 0.008856) ? cbrt(y) : (7.787 * y + 16.0 / 116.0);
//     z = (z > 0.008856) ? cbrt(z) : (7.787 * z + 16.0 / 116.0);
//     float l,a,bb;

//     l = (116.0 * y) - 16;
//     a = 500 * (x - y);
//     bb = 200 * (y - z);

//     lab[0] = l;
//     lab[1] = a;
//     lab[2] = bb;
// }
int main(int argc, char const *argv[])
{
    const size_t width = 960; const size_t height = 540;
    helper::Timer timer;
    
    cuda_renderer::Model model(prefix+"test.ply");

    Mat K = (Mat_<float>(3,3) << 572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0);
    auto proj = cuda_renderer::compute_proj(K, width, height);

    Mat R_ren = (Mat_<float>(3,3) << 1, 0, 0.00000000, 0,
                 1, 0, 0, 0,1);
    Mat t_ren = (Mat_<float>(3,1) << 10, 0, 25);

    cuda_renderer::Model::mat4x4 mat4;
    mat4.init_from_cv(R_ren, t_ren);

    std::vector<cuda_renderer::Model::mat4x4> mat4_v(1, mat4);
    for(int i = -22; i <20; i ++){
        for(int j = -10; j <10; j ++){
            t_ren = (Mat_<float>(3,1) << i, j, 25);
            mat4.init_from_cv(R_ren, t_ren);
            // std::cout<<mat4.a3;
            mat4_v.push_back(mat4);
        }
        
    }
    
    std::cout << "test render nums: " << mat4_v.size() << std::endl;
    std::cout << "---------------------------------\n" << std::endl;
    

#ifdef CUDA_ON
    {  // gpu need sometime to warm up
        cudaFree(0);
//        cudaSetDevice(0);
    }

    if(true){   //render test
        std::cout << "\nrendering test" << std::endl;
        std::cout << "-----------------------\n" << std::endl;
        timer.reset();
        //timer.reset();
        // std::cout<<model.tris[0].color.v1;
        // std::vector<int> result_cpu = cuda_renderer::render_cpu(model.tris, mat4_v, width, height, proj);
        // timer.out("cpu render");
        cv::Mat src_host = cv::imread("/home/jessy/cv/000000.left.png", cv::IMREAD_COLOR);
        // cv::Mat lab;
        // cv::cvtColor(src_host,lab,cv::COLOR_BGR2Lab);
        // std::vector<float> l_v;
        // std::vector<float> a_v;
        // std::vector<float> b_v;
        std::vector<uint8_t> r_v;
        std::vector<uint8_t> g_v;
        std::vector<uint8_t> b_v;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                cv::Vec3b elem = src_host.at<cv::Vec3b>(y, x);
                // float lab[3];
                // rgb2lab(elem[0],elem[1],elem[2],lab);
                // l_v.push_back(lab[0]);
                // a_v.push_back(lab[1]);
                // b_v.push_back(lab[2]);
                r_v.push_back(elem[2]);
                g_v.push_back(elem[1]);
                b_v.push_back(elem[0]);
                
            }
        }
        // std::cout << "M = "<< std::endl << " "  << lab << std::endl << std::endl;
        std::vector<std::vector<uint8_t>> observed;
        observed.push_back(r_v);
        observed.push_back(g_v);
        observed.push_back(b_v);
        // std::cout<<height<<","<<width;
        // std::ofstream myfile;
        // myfile.open("lab.txt");
        // for(int i =0;i <g_v.size(); i ++){
            // if(g_v[i] < 0){
                // std::cout<<g_v[i]<<", ";
            // }
            // myfile << b_v[i] << ",";
            
        // }
        // myfile.close();
        
        
        // std::vector<std::vector<uint8_t>> result_gpu = cuda_renderer::render_cuda(model.tris, mat4_v, width, height, proj);
        // timer.out("only render");
        // size_t num_rendered = mat4_v.size();
        // std::vector<cv::Mat> test_mat;
        // for(int n = 0; n <num_rendered; n ++){
        //     cv::Mat cur_mat = cv::Mat(height,width,CV_8UC3);
        //     for(int i = 0; i < height; i ++){
        //         for(int j = 0; j <width; j ++){
        //             int index = n*width*height+(i*width+j);
        //             int red = result_gpu[0][index];
        //             int green = result_gpu[1][index];
        //             int blue = result_gpu[2][index];
        //             // std::cout<<red<<","<<green<<","<<blue<<std::endl;
        //             cur_mat.at<Vec3b>(i, j) = Vec3b(blue, green,red);
        //         }
        //     }
        //     // cv::Mat lab;
        //     // cv::cvtColor(cur_mat,lab,cv::COLOR_BGR2Lab);
        //     test_mat.push_back(lab);
        // }
        // timer.out("change to cv mat");
        // std::cout<<result_gpu[2].size()<<"aaa";
        // std::cout<<observed[2].size()<<"aaa";
        // std::vector<int> result_cost = cuda_renderer::compute_cost(result_gpu,observed,height,width,num_rendered);
        // timer.out("compute cost");
        // // std::cout<<"size"<< result_cost.size();
        // cv::Mat cur_mat = cv::Mat(height,width,CV_8UC3);
        // for(int n = 0; n <num_rendered; n ++){
            
        //     for(int i = 0; i < height; i ++){
        //         for(int j = 0; j <width; j ++){
        //             int index = n*width*height+(i*width+j);
        //             int dep = result_cost[index];
        //             if(dep!= 0){
        //                 cur_mat.at<Vec3b>(i, j) = Vec3b(100,100,100);
        //             }else{
        //                 cur_mat.at<Vec3b>(i, j) = Vec3b(0,0,0);;
        //             }
        // //             // std::cout<<red<<","<<green<<","<<blue<<std::endl;
                    
        //         }
        //     }
            
        // }
        // // cv::Mat depth = cv::Mat(height, width, CV_32SC1, result_cost.data());
        // cv::imshow("gpu_mask1", cur_mat);
        // cv::FileStorage file("some_name.txt", cv::FileStorage::WRITE);
        // file << "matName" << cur_mat;
        // std::cout<<"aaa";
        // cv::Mat color_mat = cv::Mat(height, width, CV_8UC3);;
        // for(int n = 0 ; n < 1; n ++){
        //     for(int i = 0; i < height; i ++){
        //         for(int j = 0; j <width; j ++){
        //             int index = n*width*height+(i*width+j);
        //             int red = observed[0][index];
        //             int green = observed[1][index];
        //             int blue = observed[2][index];
        //             // std::cout<<red<<","<<green<<","<<blue<<std::endl;
        //             color_mat.at<Vec3b>(i, j) = Vec3b(blue, green,red);
        //         }
        //     }
        //     std::string name;
        //     name = std::to_string(n);
        //     cv::imshow(name, color_mat);
        // }
        
        // imwrite( "/home/jessy/pose_refine/test/Image.jpg", depth );
        cv::waitKey(0);

        // cv::Mat depth = cv::Mat(height, width, CV_32SC1, result_gpu.data());
        // cv::FileStorage file("some_name.txt", cv::FileStorage::WRITE);
        // file << "matName" << depth;
        //std::cout << "M = "<< std::endl << " "  << depth << std::endl << std::endl;
        // cv::Mat depth1 = cv::Mat(height, width, CV_32SC1, result_gpu.data()+99*height*width);
        // cv::imshow("gpu_mask1", depth);
        // cv::imshow("gpu_depth1", helper::view_dep(depth1));
        // cv::imshow("gpu_mask", depth>0);
        // cv::imshow("gpu_depth", helper::view_dep(depth));
        // cv::waitKey(0);
    }


#else

#endif

    return 0;
}
