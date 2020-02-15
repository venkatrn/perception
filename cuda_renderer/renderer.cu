#include "cuda_renderer/renderer.h"
// #include <math.h> 
#include "cuda_fp16.h"
// #include <numeric> 
#define SQR(x) ((x)*(x))
#define POW2(x) SQR(x)
#define POW3(x) ((x)*(x)*(x))
#define POW4(x) (POW2(x)*POW2(x))
#define POW7(x) (POW3(x)*POW3(x)*(x))
#define DegToRad(x) ((x)*M_PI/180)
#define RadToDeg(x) ((x)/M_PI*180)

namespace cuda_renderer {
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


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


template<typename T>
device_vector_holder<T>::~device_vector_holder(){
    __free();
}

template<typename T>
void device_vector_holder<T>::__free(){
    if(valid){
        cudaFree(__gpu_memory);
        valid = false;
        __size = 0;
    }
}

template<typename T>
device_vector_holder<T>::device_vector_holder(size_t size_, T init)
{
    __malloc(size_);
    thrust::fill(begin_thr(), end_thr(), init);
}

template<typename T>
void device_vector_holder<T>::__malloc(size_t size_){
    if(valid) __free();
    cudaMalloc((void**)&__gpu_memory, size_ * sizeof(T));
    __size = size_;
    valid = true;
}

template<typename T>
device_vector_holder<T>::device_vector_holder(size_t size_){
    __malloc(size_);
}

template class device_vector_holder<int>;

void print_cuda_memory_usage(){
    // show memory usage of GPU

    size_t free_byte ;
    size_t total_byte ;
    auto cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

    if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

struct max2zero_functor{

    max2zero_functor(){}

    __host__ __device__
    int32_t operator()(const int32_t& x) const
    {
      return (x==INT_MAX)? 0: x;
    }
};

__device__ void rgb2lab(uint8_t rr,uint8_t gg, uint8_t bbb, float* lab){
    double r = rr / 255.0;
    double g = gg / 255.0;
    double b = bbb / 255.0;
    double x;
    double y;
    double z;
    r = ((r > 0.04045) ? pow((r + 0.055) / 1.055, 2.4) : (r / 12.92)) * 100.0;
    g = ((g > 0.04045) ? pow((g + 0.055) / 1.055, 2.4) : (g / 12.92)) * 100.0;
    b = ((b > 0.04045) ? pow((b + 0.055) / 1.055, 2.4) : (b / 12.92)) * 100.0;

    x = r*0.4124564 + g*0.3575761 + b*0.1804375;
    y = r*0.2126729 + g*0.7151522 + b*0.0721750;
    z = r*0.0193339 + g*0.1191920 + b*0.9503041;

    x = x / 95.047;
    y = y / 100.00;
    z = z / 108.883;

    x = (x > 0.008856) ? cbrt(x) : (7.787 * x + 16.0 / 116.0);
    y = (y > 0.008856) ? cbrt(y) : (7.787 * y + 16.0 / 116.0);
    z = (z > 0.008856) ? cbrt(z) : (7.787 * z + 16.0 / 116.0);
    float l,a,bb;

    l = (116.0 * y) - 16;
    a = 500 * (x - y);
    bb = 200 * (y - z);

    lab[0] = l;
    lab[1] = a;
    lab[2] = bb;
}
__device__ double color_distance(float l1,float a1,float b1,
                      float l2,float a2,float b2){
    double eps = 1e-5;
    double c1 = sqrtf(SQR(a1) + SQR(b1));
    double c2 = sqrtf(SQR(a2) + SQR(b2));
    double meanC = (c1 + c2) / 2.0;
    double meanC7 = POW7(meanC);

    double g = 0.5*(1 - sqrtf(meanC7 / (meanC7 + 6103515625.))); // 0.5*(1-sqrt(meanC^7/(meanC^7+25^7)))
    double a1p = a1 * (1 + g);
    double a2p = a2 * (1 + g);

    c1 = sqrtf(SQR(a1p) + SQR(b1));
    c2 = sqrtf(SQR(a2p) + SQR(b2));
    double h1 = fmodf(atan2f(b1, a1p) + 2*M_PI, 2*M_PI);
    double h2 = fmodf(atan2f(b2, a2p) + 2*M_PI, 2*M_PI);

    // compute deltaL, deltaC, deltaH
    double deltaL = l2 - l1;
    double deltaC = c2 - c1;
    double deltah;

    if (c1*c2 < eps) {
        deltah = 0;
    }
    if (std::abs(h2 - h1) <= M_PI) {
        deltah = h2 - h1;
    }
    else if (h2 > h1) {
        deltah = h2 - h1 - 2* M_PI;
    }
    else {
        deltah = h2 - h1 + 2 * M_PI;
    }

    double deltaH = 2 * sqrtf(c1*c2)*sinf(deltah / 2);

    // calculate CIEDE2000
    double meanL = (l1 + l2) / 2;
    meanC = (c1 + c2) / 2.0;
    meanC7 = POW7(meanC);
    double meanH;

    if (c1*c2 < eps) {
        meanH = h1 + h2;
    }
    if (std::abs(h1 - h2) <= M_PI + eps) {
        meanH = (h1 + h2) / 2;
    }
    else if (h1 + h2 < 2*M_PI) {
        meanH = (h1 + h2 + 2*M_PI) / 2;
    }
    else {
        meanH = (h1 + h2 - 2*M_PI) / 2;
    }

    double T = 1
        - 0.17*cosf(meanH - DegToRad(30))
        + 0.24*cosf(2 * meanH)
        + 0.32*cosf(3 * meanH + DegToRad(6))
        - 0.2*cosf(4 * meanH - DegToRad(63));
    double sl = 1 + (0.015*SQR(meanL - 50)) / sqrtf(20 + SQR(meanL - 50));
    double sc = 1 + 0.045*meanC;
    double sh = 1 + 0.015*meanC*T;
    double rc = 2 * sqrtf(meanC7 / (meanC7 + 6103515625.));
    double rt = -sinf(DegToRad(60 * expf(-SQR((RadToDeg(meanH) - 275) / 25)))) * rc;

    double cur_dist = sqrtf(SQR(deltaL / sl) + SQR(deltaC / sc) + SQR(deltaH / sh) + rt * deltaC / sc * deltaH / sh);
    return cur_dist;
}
__device__
void rasterization_with_source(const Model::Triangle dev_tri, Model::float3 last_row,
                                        int32_t* depth_entry, size_t width, size_t height,
                                        const Model::ROI roi, 
                                        uint8_t* red_entry,uint8_t* green_entry,uint8_t* blue_entry,
                                        int32_t* source_depth_entry,
                                        uint8_t* source_red_entry,uint8_t* source_green_entry,uint8_t* source_blue_entry,
                                        int* pose_occluded_entry,
                                        int32_t* lock_entry,
                                        int* pose_occluded_other_entry,
                                        float* pose_clutter_points_entry,
                                        float* pose_total_points_entry) {
                                        // float* l_entry,float* a_entry,float* b_entry){
    // refer to tiny renderer
    // https://github.com/ssloy/tinyrenderer/blob/master/our_gl.cpp
    float pts2[3][2];

    // viewport transform(0, 0, width, height)
    pts2[0][0] = dev_tri.v0.x/last_row.x*width/2.0f+width/2.0f; pts2[0][1] = dev_tri.v0.y/last_row.x*height/2.0f+height/2.0f;
    pts2[1][0] = dev_tri.v1.x/last_row.y*width/2.0f+width/2.0f; pts2[1][1] = dev_tri.v1.y/last_row.y*height/2.0f+height/2.0f;
    pts2[2][0] = dev_tri.v2.x/last_row.z*width/2.0f+width/2.0f; pts2[2][1] = dev_tri.v2.y/last_row.z*height/2.0f+height/2.0f;

    float bboxmin[2] = {FLT_MAX,  FLT_MAX};
    float bboxmax[2] = {-FLT_MAX, -FLT_MAX};

    float clamp_max[2] = {float(width-1), float(height-1)};
    float clamp_min[2] = {0, 0};

    size_t real_width = width;
    if(roi.width > 0 && roi.height > 0){  // depth will be flipped
        clamp_min[0] = roi.x;
        clamp_min[1] = height-1 - (roi.y + roi.height - 1);
        clamp_max[0] = (roi.x + roi.width) - 1;
        clamp_max[1] = height-1 - roi.y;
        real_width = roi.width;
    }


    for (int i=0; i<3; i++) {
        for (int j=0; j<2; j++) {
            bboxmin[j] = std__max(clamp_min[j], std__min(bboxmin[j], pts2[i][j]));
            bboxmax[j] = std__min(clamp_max[j], std__max(bboxmax[j], pts2[i][j]));
        }
    }

    size_t P[2];
    for(P[1] = size_t(bboxmin[1]+0.5f); P[1]<=bboxmax[1]; P[1] += 1){
        for(P[0] = size_t(bboxmin[0]+0.5f); P[0]<=bboxmax[0]; P[0] += 1){
            Model::float3 bc_screen  = barycentric(pts2[0], pts2[1], pts2[2], P);

            if (bc_screen.x<-0.0f || bc_screen.y<-0.0f || bc_screen.z<-0.0f ||
                    bc_screen.x>1.0f || bc_screen.y>1.0f || bc_screen.z>1.0f ) continue;

            Model::float3 bc_over_z = {bc_screen.x/last_row.x, bc_screen.y/last_row.y, bc_screen.z/last_row.z};

            // refer to https://en.wikibooks.org/wiki/Cg_Programming/Rasterization, Perspectively Correct Interpolation
//            float frag_depth = (dev_tri.v0.z * bc_over_z.x + dev_tri.v1.z * bc_over_z.y + dev_tri.v2.z * bc_over_z.z)
//                    /(bc_over_z.x + bc_over_z.y + bc_over_z.z);

            // this seems better
            float frag_depth = (bc_screen.x + bc_screen.y + bc_screen.z)
                    /(bc_over_z.x + bc_over_z.y + bc_over_z.z);

            size_t x_to_write = (P[0] + roi.x);
            size_t y_to_write = (height-1 - P[1] - roi.y);
            int32_t curr_depth = int32_t(frag_depth/**1000*/ + 0.5f);
            // printf("x:%d, y:%d, depth:%d\n", x_to_write, y_to_write, curr_depth);
            int32_t& depth_to_write = depth_entry[x_to_write+y_to_write*real_width];
            int32_t& source_depth = source_depth_entry[x_to_write+y_to_write*real_width];
            uint8_t source_red = source_red_entry[x_to_write+y_to_write*real_width];
            uint8_t source_green = source_green_entry[x_to_write+y_to_write*real_width];
            uint8_t source_blue = source_blue_entry[x_to_write+y_to_write*real_width];

            // if(depth_to_write > curr_depth){
            //     red_entry[x_to_write+y_to_write*real_width] = (uint8_t)(dev_tri.color.v0);
            //     green_entry[x_to_write+y_to_write*real_width] = (uint8_t)(dev_tri.color.v1);
            //     blue_entry[x_to_write+y_to_write*real_width] = (uint8_t)(dev_tri.color.v2);
            // }
            // atomicMin(&depth_to_write, curr_depth);
            bool wait = true;
            while(wait){
                if(0 == atomicExch(&lock_entry[x_to_write+y_to_write*real_width], 1)){
                    if(curr_depth < depth_entry[x_to_write+y_to_write*real_width]){
                        depth_entry[x_to_write+y_to_write*real_width] = curr_depth;
                        red_entry[x_to_write+y_to_write*real_width] = (uint8_t)(dev_tri.color.v0);
                        green_entry[x_to_write+y_to_write*real_width] = (uint8_t)(dev_tri.color.v1);
                        blue_entry[x_to_write+y_to_write*real_width] = (uint8_t)(dev_tri.color.v2);
                    }
                    lock_entry[x_to_write+y_to_write*real_width] = 0;
                    wait = false;
                }
            }
            int32_t& new_depth = depth_entry[x_to_write+y_to_write*real_width];
            if(new_depth > source_depth && source_depth > 0){
                // when we are rendering at x,y where source pixel is also present at depth closer to camera
                // valid condition as source occludes render
                // if (false)
                // {
                //     // add source pixels
                //     red_entry[x_to_write+y_to_write*real_width] = source_red;
                //     green_entry[x_to_write+y_to_write*real_width] = source_green;
                //     blue_entry[x_to_write+y_to_write*real_width] = source_blue;
                //     atomicMin(&new_depth, source_depth);
                // }
                // else
                // {
                    // add black
                    red_entry[x_to_write+y_to_write*real_width] = 0;
                    green_entry[x_to_write+y_to_write*real_width] = 0;
                    blue_entry[x_to_write+y_to_write*real_width] = 0;
                    atomicMax(&new_depth, INT_MAX);
                    atomicOr(pose_occluded_other_entry, 1);
                    if (source_depth <=  new_depth - 5)
                    {
                        atomicAdd(pose_clutter_points_entry, 1);
                    }
                // }
            }
            // invalid condition where source pixel is behind and we are rendering a pixel at same x,y with lesser depth 
            else if(new_depth <= source_depth && source_depth > 0){
                // invalid as render occludes source
                atomicOr(pose_occluded_entry, 1);
                // printf("Occlusion\n");
            }
            atomicAdd(pose_total_points_entry, 1);

        }
    }
}
__device__
void rasterization(const Model::Triangle dev_tri, Model::float3 last_row,
                                        int32_t* depth_entry, size_t width, size_t height,
                                        const Model::ROI roi, uint8_t* red_entry,uint8_t* green_entry,uint8_t* blue_entry){
                                        // float* l_entry,float* a_entry,float* b_entry){
    // refer to tiny renderer
    // https://github.com/ssloy/tinyrenderer/blob/master/our_gl.cpp
    float pts2[3][2];

    // viewport transform(0, 0, width, height)
    pts2[0][0] = dev_tri.v0.x/last_row.x*width/2.0f+width/2.0f; pts2[0][1] = dev_tri.v0.y/last_row.x*height/2.0f+height/2.0f;
    pts2[1][0] = dev_tri.v1.x/last_row.y*width/2.0f+width/2.0f; pts2[1][1] = dev_tri.v1.y/last_row.y*height/2.0f+height/2.0f;
    pts2[2][0] = dev_tri.v2.x/last_row.z*width/2.0f+width/2.0f; pts2[2][1] = dev_tri.v2.y/last_row.z*height/2.0f+height/2.0f;

    float bboxmin[2] = {FLT_MAX,  FLT_MAX};
    float bboxmax[2] = {-FLT_MAX, -FLT_MAX};

    float clamp_max[2] = {float(width-1), float(height-1)};
    float clamp_min[2] = {0, 0};

    size_t real_width = width;
    if(roi.width > 0 && roi.height > 0){  // depth will be flipped
        clamp_min[0] = roi.x;
        clamp_min[1] = height-1 - (roi.y + roi.height - 1);
        clamp_max[0] = (roi.x + roi.width) - 1;
        clamp_max[1] = height-1 - roi.y;
        real_width = roi.width;
    }


    for (int i=0; i<3; i++) {
        for (int j=0; j<2; j++) {
            bboxmin[j] = std__max(clamp_min[j], std__min(bboxmin[j], pts2[i][j]));
            bboxmax[j] = std__min(clamp_max[j], std__max(bboxmax[j], pts2[i][j]));
        }
    }

    size_t P[2];
    for(P[1] = size_t(bboxmin[1]+0.5f); P[1]<=bboxmax[1]; P[1] += 1){
        for(P[0] = size_t(bboxmin[0]+0.5f); P[0]<=bboxmax[0]; P[0] += 1){
            Model::float3 bc_screen  = barycentric(pts2[0], pts2[1], pts2[2], P);

            if (bc_screen.x<-0.0f || bc_screen.y<-0.0f || bc_screen.z<-0.0f ||
                    bc_screen.x>1.0f || bc_screen.y>1.0f || bc_screen.z>1.0f ) continue;

            Model::float3 bc_over_z = {bc_screen.x/last_row.x, bc_screen.y/last_row.y, bc_screen.z/last_row.z};

            // refer to https://en.wikibooks.org/wiki/Cg_Programming/Rasterization, Perspectively Correct Interpolation
//            float frag_depth = (dev_tri.v0.z * bc_over_z.x + dev_tri.v1.z * bc_over_z.y + dev_tri.v2.z * bc_over_z.z)
//                    /(bc_over_z.x + bc_over_z.y + bc_over_z.z);

            // this seems better
            float frag_depth = (bc_screen.x + bc_screen.y + bc_screen.z)
                    /(bc_over_z.x + bc_over_z.y + bc_over_z.z);

            size_t x_to_write = (P[0] + roi.x);
            size_t y_to_write = (height-1 - P[1] - roi.y);

            int32_t depth = int32_t(frag_depth/**1000*/ + 0.5f);
            int32_t& depth_to_write = depth_entry[x_to_write+y_to_write*real_width];

            if(depth_entry[x_to_write+y_to_write*real_width] > depth){
                red_entry[x_to_write+y_to_write*real_width] = (uint8_t)(dev_tri.color.v0);
                green_entry[x_to_write+y_to_write*real_width] = (uint8_t)(dev_tri.color.v1);
                blue_entry[x_to_write+y_to_write*real_width] = (uint8_t)(dev_tri.color.v2);
            }
            atomicMin(&depth_to_write,depth);
        }
    }
}

__global__ void render_triangle(Model::Triangle* device_tris_ptr, size_t device_tris_size,
                                Model::mat4x4* device_poses_ptr, size_t device_poses_size,
                                int32_t* depth_image_vec, size_t width, size_t height, const Model::mat4x4 proj_mat,
                                 const Model::ROI roi,uint8_t* red_image_vec,uint8_t* green_image_vec,uint8_t* blue_image_vec){
                                 // float* l_vec,float* a_vec,float* b_vec){
    size_t pose_i = blockIdx.y;
    size_t tri_i = blockIdx.x*blockDim.x + threadIdx.x;

    if(tri_i>=device_tris_size) return;
//    if(pose_i>=device_poses_size) return;

    size_t real_width = width;
    size_t real_height = height;
    if(roi.width > 0 && roi.height > 0){
        real_width = roi.width;
        real_height = roi.height;
    }

    int32_t* depth_entry = depth_image_vec + pose_i*real_width*real_height; //length: width*height 32bits int
    uint8_t* red_entry = red_image_vec + pose_i*real_width*real_height;
    uint8_t* green_entry = green_image_vec + pose_i*real_width*real_height;
    uint8_t* blue_entry = blue_image_vec + pose_i*real_width*real_height;
    // float* l_entry = l_vec + pose_i*real_width*real_height;
    // float* a_entry = a_vec + pose_i*real_width*real_height;
    // float* b_entry = b_vec + pose_i*real_width*real_height;
    Model::mat4x4* pose_entry = device_poses_ptr + pose_i; // length: 16 32bits float
    Model::Triangle* tri_entry = device_tris_ptr + tri_i; // length: 9 32bits float

    // model transform
    Model::Triangle local_tri = transform_triangle(*tri_entry, *pose_entry);
//    if(normal_functor::is_back(local_tri)) return; //back face culling, need to be disable for not well defined surfaces?

    // assume last column of projection matrix is  0 0 1 0
    Model::float3 last_row = {
        local_tri.v0.z,
        local_tri.v1.z,
        local_tri.v2.z
    };
    // projection transform
    local_tri = transform_triangle(local_tri, proj_mat);

    // rasterization(local_tri, last_row, depth_entry, width, height, roi,red_entry,green_entry,blue_entry,l_entry,a_entry,b_entry);
    rasterization(local_tri, last_row, depth_entry, width, height, roi,red_entry,green_entry,blue_entry);
}
__global__ void render_triangle_multi(
                                Model::Triangle* device_tris_ptr, size_t device_tris_size,
                                Model::mat4x4* device_poses_ptr, size_t device_poses_size,
                                int32_t* depth_image_vec, size_t width, size_t height,
                                int* device_pose_model_map_ptr, int* device_tris_model_count_low_ptr,  
                                int* device_tris_model_count_high_ptr,
                                const Model::mat4x4 proj_mat, const Model::ROI roi,
                                uint8_t* red_image_vec,uint8_t* green_image_vec,uint8_t* blue_image_vec,
                                int32_t* device_source_depth_vec,
                                uint8_t* device_source_red_vec,uint8_t* device_source_green_vec,uint8_t* device_source_blue_vec,
                                int* pose_occluded_vec,
                                int* device_single_result_image,
                                int32_t* lock_int_vec,
                                int* pose_occluded_other_vec,
                                float* pose_clutter_points_vec, 
                                float* pose_total_points_vec) {
    size_t pose_i = blockIdx.y;
    int model_id = device_pose_model_map_ptr[pose_i];
    size_t tri_i = blockIdx.x*blockDim.x + threadIdx.x;

    if(tri_i>=device_tris_size) return;

    if (!(tri_i < device_tris_model_count_high_ptr[model_id] && tri_i >= device_tris_model_count_low_ptr[model_id]))
        return; 

    size_t real_width = width;
    size_t real_height = height;
    if(roi.width > 0 && roi.height > 0){
        real_width = roi.width;
        real_height = roi.height;
    }
    int32_t* depth_entry;
    int32_t* lock_entry;
    uint8_t* red_entry;
    uint8_t* green_entry;
    uint8_t* blue_entry;
    int* pose_occluded_entry;
    int* pose_occluded_other_entry;
    float* pose_clutter_points_entry;
    float* pose_total_points_entry;
    // printf("device_single_result_image:%d\n",device_single_result_image);
    if (*device_single_result_image)
    {
        depth_entry = depth_image_vec; //length: width*height 32bits int
        red_entry = red_image_vec;
        green_entry = green_image_vec;
        blue_entry = blue_image_vec;
        pose_occluded_entry = pose_occluded_vec;
        lock_entry = lock_int_vec;
        pose_occluded_other_entry = pose_occluded_other_vec;
        pose_clutter_points_entry = pose_clutter_points_vec;
        pose_total_points_entry = pose_total_points_vec;
    }
    else
    {
        depth_entry = depth_image_vec + pose_i*real_width*real_height; //length: width*height 32bits int
        lock_entry = lock_int_vec + pose_i*real_width*real_height;
        red_entry = red_image_vec + pose_i*real_width*real_height;
        green_entry = green_image_vec + pose_i*real_width*real_height;
        blue_entry = blue_image_vec + pose_i*real_width*real_height;
        pose_occluded_entry = pose_occluded_vec + pose_i;
        pose_occluded_other_entry = pose_occluded_other_vec + pose_i;
        pose_clutter_points_entry = pose_clutter_points_vec + pose_i;
        pose_total_points_entry = pose_total_points_vec + pose_i;
    }
    

    Model::mat4x4* pose_entry = device_poses_ptr + pose_i; // length: 16 32bits float
    Model::Triangle* tri_entry = device_tris_ptr + tri_i; // length: 9 32bits float

    // model transform
    Model::Triangle local_tri = transform_triangle(*tri_entry, *pose_entry);

    // assume last column of projection matrix is  0 0 1 0
    Model::float3 last_row = {
        local_tri.v0.z,
        local_tri.v1.z,
        local_tri.v2.z
    };
    // projection transform
    local_tri = transform_triangle(local_tri, proj_mat);

    // rasterization(local_tri, last_row, depth_entry, width, height, roi,red_entry,green_entry,blue_entry);
    rasterization_with_source(
        local_tri, last_row, depth_entry, width, height, roi,
        red_entry,green_entry,blue_entry,
        device_source_depth_vec,
        device_source_red_vec, device_source_green_vec, device_source_blue_vec,
        pose_occluded_entry,
        lock_entry,
        pose_occluded_other_entry,
        pose_clutter_points_entry,
        pose_total_points_entry);
}
__global__ void bgr_to_gray_kernel( uint8_t* red_in,uint8_t* green_in,uint8_t* blue_in,
                                    uint8_t* red_ob, uint8_t* green_ob,uint8_t* blue_ob, 
                                    int32_t* output, 
                                    int width,
                                    int height,
                                    int num_rendered)
{
 //2D Index of current thread
    int num = (int)floorf((blockIdx.x * blockDim.x + threadIdx.x)/width);
    const int xIndex = (blockIdx.x * blockDim.x + threadIdx.x)%width;
    // const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int real_distance;
    //Only valid threads perform memory I/O
    if(xIndex == 213 && yIndex == 143 ){
        // printf("bbbb%d\n", num);
    }
    if((xIndex<width) && (yIndex<height))
    {
        //Location of colored pixel in input
        int valid;
        int cur_id = yIndex * width + xIndex+ num*width*height;
        // int input_id = num*width*height+cur_id;
        
        uint8_t red = red_in[cur_id];
        uint8_t green = green_in[cur_id];
        uint8_t blue = blue_in[cur_id];
       // if(red != 0 || green != 0|| blue != 0){
        float l1,a1,b1,l2,a2,b2; 
        if(red ==0 && green == 0 && blue ==0){
            l1  = 0;
            a1  = 0;
            b1  = 0;
            int id_b = yIndex * width + xIndex;
            if(red_ob[id_b]==0 && green_ob[id_b]==0 && blue_ob[id_b]==0){

            }else{
                uint8_t v =0;
                for(int i = -2; i <3;i++){
                    id_b = (yIndex+i) * width + xIndex+i;
                    if(red_ob[id_b]==0 && green_ob[id_b]==0 && blue_ob[id_b]==0){
                        v = 1;
                    }
                }
                if(v == 0){
                    output[cur_id] = 1;
                }
                
            }

        }else{
            output[cur_id] = 1;
            float lab[3];
            rgb2lab(red,green,blue,lab);
            l1  = lab[0];
            a1  = lab[1];
            b1  = lab[2];
            output[cur_id] = 1;

            for(int i = -2; i <3;i++){
                int row = yIndex+i;
                int col = xIndex+i;
                if(row >= 0 && row <height && col >= 0 && col <width){
                    int id = (row) * width + col;
                    uint8_t red2  = red_ob[id];
                    uint8_t green2  = green_ob[id];
                    uint8_t blue2  = blue_ob[id];
                    if(red2 ==0 && green2 == 0 && blue2 ==0){
                    l2  = 0;
                    a2  = 0;
                    b2  = 0;

                    }else{
                        float lab2[3];
                        rgb2lab(red2,green2,blue2,lab2);
                        l2  = lab2[0];
                        a2  = lab2[1];
                        b2  = lab2[2];
                    }
                    double cur_dist=color_distance(l1,a1,b1,l2,a2,b2);
                    if(cur_dist<20){
                        valid = 1;
                        output[cur_id] = 0;

                    }
                         
                        // if(i==0){
                        //     real_distance = cur_dist;
                        // }
                }

            }
        }   
    }
}

std::vector<int> compute_rgb_cost(const std::vector<std::vector<uint8_t>> input,
                                  const std::vector<std::vector<uint8_t>> observed,
                                  size_t height, size_t width,size_t num_rendered) 
{
    //Calculate total number of bytes of input and output image
    // std::cout<<"aaa";
    size_t bytes = input[0].size();
    size_t bytes_ob = observed[0].size();
   
    // //Allocate device memory
    thrust::device_vector<int> d_output(num_rendered*width*height, 0);
    // this contains all images
    thrust::device_vector<uint8_t> d_red_in = input[0];
    thrust::device_vector<uint8_t> d_green_in = input[1];
    thrust::device_vector<uint8_t> d_blue_in = input[2];
    thrust::device_vector<uint8_t> d_red_ob = observed[0];
    thrust::device_vector<uint8_t> d_green_ob = observed[1];
    thrust::device_vector<uint8_t> d_blue_ob = observed[2];
    {

        int32_t* depth_vec = thrust::raw_pointer_cast(d_output.data());
        uint8_t* red_in = thrust::raw_pointer_cast(d_red_in.data());
        uint8_t* green_in = thrust::raw_pointer_cast(d_green_in.data());
        uint8_t* blue_in = thrust::raw_pointer_cast(d_blue_in.data());
        uint8_t* red_ob = thrust::raw_pointer_cast(d_red_ob.data());
        uint8_t* green_ob = thrust::raw_pointer_cast(d_green_ob.data());
        uint8_t* blue_ob = thrust::raw_pointer_cast(d_blue_ob.data());

        dim3 block(16,16);
        dim3 grid((width*num_rendered + block.x - 1)/block.x, (height + block.y - 1)/block.y);
        bgr_to_gray_kernel<<<grid,block>>>(red_in,green_in,blue_in,
                                       red_ob,green_ob,blue_ob,
                                       depth_vec,
                                       width,height,num_rendered);
        cudaDeviceSynchronize();
        // gpuErrchk(cudaPeekAtLastError());
    }

   
    std::vector<int> result_depth(num_rendered*width*height);
    {
        thrust::transform(d_output.begin(), d_output.end(),
                          d_output.begin(), max2zero_functor());
        thrust::copy(d_output.begin(), d_output.end(), result_depth.begin());

    }
    
    // //Copy back data from destination device meory to OpenCV output image
    std::vector<int> cost(num_rendered);
    for(int i = 0 ; i < num_rendered; i ++){
        cost[i] = std::accumulate(result_depth.begin()+i*width*height,result_depth.begin()+(i+1)*width*height,0);
    }
    // std::cout<< cost[0] <<"!!!!!!!!!!";
    
    return cost;
}
__global__ void merge_with_source( uint8_t* merged_red,uint8_t* merged_green,uint8_t* merged_blue,
                                    int32_t* merged_depth,
                                    uint8_t* rendered_red, uint8_t* rendered_green,uint8_t* rendered_blue, 
                                    int32_t* rendered_depth,
                                    uint8_t* source_red, uint8_t* source_green,uint8_t* source_blue, 
                                    int32_t* source_depth, 
                                    int width,
                                    int height,
                                    int num_rendered)
{
    //2D Index of current thread
    int n = (int)floorf((blockIdx.x * blockDim.x + threadIdx.x)/width);
    const int x = (blockIdx.x * blockDim.x + threadIdx.x)%width;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width) return;
    if(y >= height) return;
    uint32_t idx_rendered = n * width * height + x + y*width;
    uint32_t idx_source = x + y*width;

    if (source_depth[idx_source] > 0 && rendered_depth[idx_rendered] == INT_MAX)
    {

    }

}
__global__ void copy_source_to_render(uint8_t* rendered_red, uint8_t* rendered_green,uint8_t* rendered_blue, 
                                    int32_t* rendered_depth,
                                    uint8_t* source_red, uint8_t* source_green,uint8_t* source_blue, 
                                    int32_t* source_depth, 
                                    int width,
                                    int height,
                                    int num_rendered)
{
    //2D Index of current thread
    int n = (int)floorf((blockIdx.x * blockDim.x + threadIdx.x)/width);
    const int x = (blockIdx.x * blockDim.x + threadIdx.x)%width;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width) return;
    if(y >= height) return;
    uint32_t idx_rendered = n * width * height + x + y*width;
    uint32_t idx_source = x + y*width;

    if (source_depth[idx_source] > 0)
    {
        rendered_red[idx_rendered] = source_red[idx_source];
        rendered_green[idx_rendered] = source_green[idx_source];
        rendered_blue[idx_rendered] = source_blue[idx_source];
        rendered_depth[idx_rendered] = source_depth[idx_source];
    }

}
device_vector_holder<int> render_cuda_multi(
                            const std::vector<Model::Triangle>& tris,
                            const std::vector<Model::mat4x4>& poses,
                            const std::vector<int> pose_model_map,
                            const std::vector<int> tris_model_count,
                            size_t width, size_t height, const Model::mat4x4& proj_mat,
                            const std::vector<int32_t>& source_depth,
                            const std::vector<std::vector<uint8_t>>& source_color,
                            std::vector<int32_t>& result_depth, 
                            std::vector<std::vector<uint8_t>>& result_color,
                            std::vector<int>& pose_occluded,
                            int single_result_image,
                            std::vector<int>& pose_occluded_other,
                            std::vector<float>& clutter_cost) {

    // Create device inputs
    int* device_single_result_image;
    cudaMalloc((void**)&device_single_result_image, sizeof(int));
    cudaMemcpy(device_single_result_image, &single_result_image, sizeof(int), cudaMemcpyHostToDevice);
    int num_images;
    if (single_result_image)
    {
        num_images = 1;
    }
    else
    {
        num_images = poses.size();
    }
    const Model::ROI roi= {0, 0, 0, 0};
    const size_t threadsPerBlock = 256;
    // std::cout <<tris[0].color.v1;
    thrust::device_vector<Model::Triangle> device_tris = tris;
    thrust::device_vector<Model::mat4x4> device_poses = poses;
    thrust::device_vector<int> device_tris_model_count_low = tris_model_count;
    thrust::device_vector<int> device_tris_model_count_high = tris_model_count;
    thrust::device_vector<int> device_pose_model_map = pose_model_map;

    thrust::device_vector<int32_t> device_source_depth = source_depth;
    thrust::device_vector<uint8_t> device_source_color_red = source_color[0];
    thrust::device_vector<uint8_t> device_source_color_green = source_color[1];
    thrust::device_vector<uint8_t> device_source_color_blue = source_color[2];
    // thrust::copy(
    //     device_tris_model_count.begin(),
    //     device_tris_model_count.end(), 
    //     std::ostream_iterator<int>(std::cout, " ")
    // );
    thrust::exclusive_scan(
        device_tris_model_count_low.begin(), device_tris_model_count_low.end(), 
        device_tris_model_count_low.begin(), 0
    ); // in-place scan
    thrust::inclusive_scan(
        device_tris_model_count_high.begin(), device_tris_model_count_high.end(), 
        device_tris_model_count_high.begin()
    ); // in-place scan
    // thrust::copy(
    //     device_tris_model_count_low.begin(),
    //     device_tris_model_count_low.end(), 
    //     std::ostream_iterator<int>(std::cout, " ")
    // );
    // printf("\n");
    // thrust::copy(
    //     device_tris_model_count_high.begin(),
    //     device_tris_model_count_high.end(), 
    //     std::ostream_iterator<int>(std::cout, " ")
    // );
    // printf("\n");
    // thrust::copy(
    //     device_pose_model_map.begin(),
    //     device_pose_model_map.end(), 
    //     std::ostream_iterator<int>(std::cout, " ")
    // );
    printf("\nNumber of triangles : %d\n", tris.size());
    printf("Number of poses : %d\n", num_images);

    size_t real_width = width;
    size_t real_height = height;

    // atomic min only support int32
    
    // Create device outputs
    thrust::device_vector<int> device_pose_occluded(num_images, 0);
    thrust::device_vector<int> device_pose_occluded_other(num_images, 0);
    thrust::device_vector<float> device_pose_clutter_points(num_images, 0);
    thrust::device_vector<float> device_pose_total_points(num_images, 0);

    device_vector_holder<int32_t> device_depth_int(num_images*real_width*real_height, INT_MAX);
    // thrust::device_vector<int32_t> device_depth_int(poses.size()*real_width*real_height, INT_MAX);
    thrust::device_vector<int32_t> device_lock_int(num_images*real_width*real_height, 0);
    thrust::device_vector<uint8_t> device_red_int(num_images*real_width*real_height, 0);
    thrust::device_vector<uint8_t> device_green_int(num_images*real_width*real_height, 0);
    thrust::device_vector<uint8_t> device_blue_int(num_images*real_width*real_height, 0);

  
    Model::Triangle* device_tris_ptr = thrust::raw_pointer_cast(device_tris.data());
    Model::mat4x4* device_poses_ptr = thrust::raw_pointer_cast(device_poses.data());

    // Mapping each pose to model
    int* device_pose_model_map_ptr = thrust::raw_pointer_cast(device_pose_model_map.data());

    // Mapping each model to triangle range
    int* device_tris_model_count_low_ptr = thrust::raw_pointer_cast(device_tris_model_count_low.data());
    int* device_tris_model_count_high_ptr = thrust::raw_pointer_cast(device_tris_model_count_high.data());
    // int32_t* depth_image_vec = thrust::raw_pointer_cast(device_depth_int.data());

    int* device_pose_occluded_vec = thrust::raw_pointer_cast(device_pose_occluded.data());
    int* device_pose_occluded_other_vec = thrust::raw_pointer_cast(device_pose_occluded_other.data());
    float* device_pose_clutter_points_vec = thrust::raw_pointer_cast(device_pose_clutter_points.data());
    float* device_pose_total_points_vec = thrust::raw_pointer_cast(device_pose_total_points.data());

    int32_t* device_source_depth_vec = thrust::raw_pointer_cast(device_source_depth.data());
    uint8_t* device_source_red_vec = thrust::raw_pointer_cast(device_source_color_red.data());
    uint8_t* device_source_green_vec = thrust::raw_pointer_cast(device_source_color_green.data());
    uint8_t* device_source_blue_vec = thrust::raw_pointer_cast(device_source_color_blue.data());

    int32_t* depth_image_vec = device_depth_int.data();
    int32_t* lock_int_vec = thrust::raw_pointer_cast(device_lock_int.data());
    uint8_t* red_image_vec = thrust::raw_pointer_cast(device_red_int.data());
    uint8_t* green_image_vec = thrust::raw_pointer_cast(device_green_int.data());
    uint8_t* blue_image_vec = thrust::raw_pointer_cast(device_blue_int.data());

    // Initialize rendered images with source images
    dim3 block(16,16);
    dim3 grid((real_width*num_images + block.x - 1)/block.x, (real_height + block.y - 1)/block.y);
    // copy_source_to_render<<<grid,block>>>(red_image_vec,green_image_vec,blue_image_vec,
    //                             depth_image_vec,
    //                             device_source_red_vec, device_source_green_vec, device_source_blue_vec,
    //                             device_source_depth_vec,
    //                             width,height,num_images);
    // cudaDeviceSynchronize();

    // Render all poses
    dim3 numBlocks((tris.size() + threadsPerBlock - 1) / threadsPerBlock, poses.size());
    render_triangle_multi<<<numBlocks, threadsPerBlock>>>(device_tris_ptr, tris.size(),
                                                    device_poses_ptr, poses.size(),
                                                    depth_image_vec, width, height, 
                                                    device_pose_model_map_ptr, device_tris_model_count_low_ptr,
                                                    device_tris_model_count_high_ptr,
                                                    proj_mat, roi,
                                                    red_image_vec,green_image_vec,blue_image_vec,
                                                    device_source_depth_vec,
                                                    device_source_red_vec, device_source_green_vec, device_source_blue_vec,
                                                    device_pose_occluded_vec,
                                                    device_single_result_image,
                                                    lock_int_vec,
                                                    device_pose_occluded_other_vec,
                                                    device_pose_clutter_points_vec,
                                                    device_pose_total_points_vec);
    // cudaDeviceSynchronize();
    // Objects occluding other objects already in the scene
    printf("Pose Occlusions\n");
    thrust::copy(
        device_pose_occluded.begin(),
        device_pose_occluded.end(), 
        std::ostream_iterator<int>(std::cout, " ")
    );
    printf("\n");
    thrust::copy(device_pose_occluded.begin(), device_pose_occluded.end(), pose_occluded.begin());
    
    // Objects occluded by existing objects in the scene, need to do ICP again for these
    printf("Pose Occlusions Other\n");
    thrust::copy(
        device_pose_occluded_other.begin(),
        device_pose_occluded_other.end(), 
        std::ostream_iterator<int>(std::cout, " ")
    );
    printf("\n");
    thrust::copy(device_pose_occluded_other.begin(), device_pose_occluded_other.end(), pose_occluded_other.begin());

    printf("Pose Clutter Ratio\n");
    thrust::transform(
        device_pose_clutter_points.begin(), device_pose_clutter_points.end(), 
        device_pose_total_points.begin(), device_pose_clutter_points.begin(), 
        thrust::divides<float>()
    );
    thrust::device_vector<float> rendered_multiplier_val(num_images, 100);
    thrust::transform(
        device_pose_clutter_points.begin(), device_pose_clutter_points.end(), 
        rendered_multiplier_val.begin(), device_pose_clutter_points.begin(), 
        thrust::multiplies<float>()
    );
    thrust::copy(device_pose_clutter_points.begin(), device_pose_clutter_points.end(), clutter_cost.begin());

    thrust::copy(
        device_pose_clutter_points.begin(),
        device_pose_clutter_points.end(), 
        std::ostream_iterator<float>(std::cout, " ")
    );
    printf("\n");

    result_depth.resize(num_images*real_width*real_height);
    {
        thrust::device_vector<int32_t> v3(depth_image_vec, depth_image_vec + num_images*real_width*real_height);
        thrust::transform(v3.begin(), v3.end(),v3.begin(), max2zero_functor());
        thrust::copy(v3.begin(), v3.end(), result_depth.begin());

    }
    
    std::vector<uint8_t> result_red(num_images*real_width*real_height);
    std::vector<uint8_t> result_green(num_images*real_width*real_height);
    std::vector<uint8_t> result_blue(num_images*real_width*real_height);
    {
        thrust::transform(device_red_int.begin(), device_red_int.end(),
                          device_red_int.begin(), max2zero_functor());
        thrust::copy(device_red_int.begin(), device_red_int.end(), result_red.begin());
        thrust::transform(device_green_int.begin(), device_green_int.end(),
                          device_green_int.begin(), max2zero_functor());
        thrust::copy(device_green_int.begin(), device_green_int.end(), result_green.begin());
        thrust::transform(device_blue_int.begin(), device_blue_int.end(),
                          device_blue_int.begin(), max2zero_functor());
        thrust::copy(device_blue_int.begin(), device_blue_int.end(), result_blue.begin());

    }
    if (result_color.size() > 0) result_color.clear();
    result_color.push_back(result_red);
    result_color.push_back(result_green);
    result_color.push_back(result_blue);


    thrust::transform(device_depth_int.begin_thr(), device_depth_int.end_thr(),
                      device_depth_int.begin_thr(), max2zero_functor());
    return device_depth_int;
}

device_vector_holder<int> render_cuda(const std::vector<Model::Triangle>& tris,const std::vector<Model::mat4x4>& poses,
                            size_t width, size_t height, const Model::mat4x4& proj_mat,
                            std::vector<int32_t>& result_depth, std::vector<std::vector<uint8_t>>& result_color, const Model::ROI roi){

    const size_t threadsPerBlock = 256;
    // std::cout <<tris[0].color.v1;
    thrust::device_vector<Model::Triangle> device_tris = tris;
    thrust::device_vector<Model::mat4x4> device_poses = poses;

    size_t real_width = width;
    size_t real_height = height;
    if(roi.width > 0 && roi.height > 0){
        real_width = roi.width;
        real_height = roi.height;
        assert(roi.x + roi.width <= width && "roi out of image");
        assert(roi.y + roi.height <= height && "roi out of image");
    }
    // atomic min only support int32
    
    device_vector_holder<int32_t> device_depth_int(poses.size()*real_width*real_height, INT_MAX);

    // thrust::device_vector<int32_t> device_depth_int(poses.size()*real_width*real_height, INT_MAX);
    thrust::device_vector<uint8_t> device_red_int(poses.size()*real_width*real_height, 0);
    thrust::device_vector<uint8_t> device_green_int(poses.size()*real_width*real_height, 0);
    thrust::device_vector<uint8_t> device_blue_int(poses.size()*real_width*real_height, 0);
    // thrust::device_vector<float> l(poses.size()*real_width*real_height, 0);
    // thrust::device_vector<float> a(poses.size()*real_width*real_height, 0);
    // thrust::device_vector<float> b(poses.size()*real_width*real_height, 0);
    // {
        Model::Triangle* device_tris_ptr = thrust::raw_pointer_cast(device_tris.data());
        Model::mat4x4* device_poses_ptr = thrust::raw_pointer_cast(device_poses.data());
        // int32_t* depth_image_vec = thrust::raw_pointer_cast(device_depth_int.data());
        int32_t* depth_image_vec = device_depth_int.data();
        uint8_t* red_image_vec = thrust::raw_pointer_cast(device_red_int.data());
        uint8_t* green_image_vec = thrust::raw_pointer_cast(device_green_int.data());
        uint8_t* blue_image_vec = thrust::raw_pointer_cast(device_blue_int.data());
        // float* l_vec = thrust::raw_pointer_cast(l.data());
        // float* a_vec = thrust::raw_pointer_cast(a.data());
        // float* b_vec = thrust::raw_pointer_cast(b.data());

        dim3 numBlocks((tris.size() + threadsPerBlock - 1) / threadsPerBlock, poses.size());
        render_triangle<<<numBlocks, threadsPerBlock>>>(device_tris_ptr, tris.size(),
                                                        device_poses_ptr, poses.size(),
                                                        depth_image_vec, width, height, proj_mat, roi,
                                                        red_image_vec,green_image_vec,blue_image_vec);
        cudaDeviceSynchronize();
        // gpuErrchk(cudaPeekAtLastError());
    // }


    result_depth.resize(poses.size()*real_width*real_height);
    {
        thrust::device_vector<int32_t> v3(depth_image_vec, depth_image_vec + poses.size()*real_width*real_height);
        thrust::transform(v3.begin(), v3.end(),v3.begin(), max2zero_functor());
        thrust::copy(v3.begin(), v3.end(), result_depth.begin());

    }
    
    std::vector<uint8_t> result_red(poses.size()*real_width*real_height);
    std::vector<uint8_t> result_green(poses.size()*real_width*real_height);
    std::vector<uint8_t> result_blue(poses.size()*real_width*real_height);
    {
        thrust::transform(device_red_int.begin(), device_red_int.end(),
                          device_red_int.begin(), max2zero_functor());
        thrust::copy(device_red_int.begin(), device_red_int.end(), result_red.begin());
        thrust::transform(device_green_int.begin(), device_green_int.end(),
                          device_green_int.begin(), max2zero_functor());
        thrust::copy(device_green_int.begin(), device_green_int.end(), result_green.begin());
        thrust::transform(device_blue_int.begin(), device_blue_int.end(),
                          device_blue_int.begin(), max2zero_functor());
        thrust::copy(device_blue_int.begin(), device_blue_int.end(), result_blue.begin());

    }
    result_color.push_back(result_red);
    result_color.push_back(result_green);
    result_color.push_back(result_blue);

    // device_vector_holder<int> device_depth_int_v(poses.size()*real_width*real_height, result_depth);
    // device_vector_holder<int> device_depth_int_v = device_depth_int;
    // device_vector_holder<int> device_depth_int_v;
    // thrust::copy(device_depth_int.begin(), device_depth_int.end(), device_depth_int_v.begin());
    
    // device_depth_int_v.__gpu_memory = thrust::raw_pointer_cast(device_depth_int.data());
    // device_depth_int_v.__size = device_depth_int.size();
    // device_depth_int_v.valid = true;
    // return device_depth_int_v;
    // thrust::transform(device_depth_int.begin(), device_depth_int.end(),device_depth_int.begin(), max2zero_functor());
    thrust::transform(device_depth_int.begin_thr(), device_depth_int.end_thr(),
                      device_depth_int.begin_thr(), max2zero_functor());
    return device_depth_int;
}

}

