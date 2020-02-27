#pragma once

#include "cuda_fp16.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>


#ifndef COMMONCUDA_H
#define COMMONCUDA_H

#define SQR(x) ((x)*(x))
#define POW2(x) SQR(x)
#define POW3(x) ((x)*(x)*(x))
#define POW4(x) (POW2(x)*POW2(x))
#define POW7(x) (POW3(x)*POW3(x)*(x))
#define DegToRad(x) ((x)*M_PI/180)
#define RadToDeg(x) ((x)/M_PI*180)

namespace cuda_renderer {
    __global__ void depth_to_mask(
        int32_t* depth, int* mask, int width, int height, int stride, int* pose_occluded)
    {
        /**
         * Creates a mask corresponding to valid depth points by using the depth data
         *
        */
        int n = (int)floorf((blockIdx.x * blockDim.x + threadIdx.x)/(width/stride));
        int x = (blockIdx.x * blockDim.x + threadIdx.x)%(width/stride);
        int y = blockIdx.y*blockDim.y + threadIdx.y;
        x = x*stride;
        y = y*stride;
        if(x >= width) return;
        if(y >= height) return;
        uint32_t idx_depth = n * width * height + x + y*width;
        uint32_t idx_mask = n * width * height + x + y*width;
    
        // if(depth[idx_depth] > 0 && !pose_occluded[n]) 
        if(depth[idx_depth] > 0) 
        {
            mask[idx_mask] = 1;
        }
    }
    
    __global__ void depth_to_cloud(
        int32_t* depth, uint8_t* r_in, uint8_t* g_in, uint8_t* b_in, float* cloud, uint8_t* cloud_color, int cloud_rendered_cloud_point_num, int* mask, int width, int height, 
        float kCameraCX, float kCameraCY, float kCameraFX, float kCameraFY, float depth_factor,
        int stride, int* cloud_pose_map, uint8_t* label_mask_data,  int* cloud_mask_label)
    {
        /**
         * Creates a point cloud by combining a mask corresponding to valid depth pixels and depth data using the camera params
         * Optionally also records the correct color of the points and their mask label
        */
        int n = (int)floorf((blockIdx.x * blockDim.x + threadIdx.x)/(width/stride));
        int x = (blockIdx.x * blockDim.x + threadIdx.x)%(width/stride);
        int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        // uint32_t x = blockIdx.x*blockDim.x + threadIdx.x;
        // uint32_t y = blockIdx.y*blockDim.y + threadIdx.y;
        x = x*stride;
        y = y*stride;
        if(x >= width) return;
        if(y >= height) return;
        uint32_t idx_depth = n * width * height + x + y*width;
    
        if(depth[idx_depth] <= 0) return;
    
        // printf("depth:%d\n", depth[idx_depth]);
        // uchar depth_val = depth[idx_depth];
        float z_pcd = static_cast<float>(depth[idx_depth])/depth_factor;
        float x_pcd = (static_cast<float>(x) - kCameraCX)/kCameraFX * z_pcd;
        float y_pcd = (static_cast<float>(y) - kCameraCY)/kCameraFY * z_pcd;
        // printf("kCameraCX:%f,kCameraFX:%f, kCameraCY:%f, kCameraCY:%f\n", kCameraCX,kCameraFX,kCameraCY, y_pcd, z_pcd);
    
        // printf("x:%d,y:%d, x_pcd:%f, y_pcd:%f, z_pcd:%f\n", x,y,x_pcd, y_pcd, z_pcd);
        uint32_t idx_mask = n * width * height + x + y*width;
        int cloud_idx = mask[idx_mask];
        cloud[cloud_idx + 0*cloud_rendered_cloud_point_num] = x_pcd;
        cloud[cloud_idx + 1*cloud_rendered_cloud_point_num] = y_pcd;
        cloud[cloud_idx + 2*cloud_rendered_cloud_point_num] = z_pcd;
    
        cloud_color[cloud_idx + 0*cloud_rendered_cloud_point_num] = r_in[idx_depth];
        cloud_color[cloud_idx + 1*cloud_rendered_cloud_point_num] = g_in[idx_depth];
        cloud_color[cloud_idx + 2*cloud_rendered_cloud_point_num] = b_in[idx_depth];
    
        cloud_pose_map[cloud_idx] = n;
        if (label_mask_data != NULL)
        {
            cloud_mask_label[cloud_idx] = label_mask_data[idx_depth];
        }
        // printf("cloud_idx:%d\n", label_mask_data[idx_depth]);
    
        // cloud[3*cloud_idx + 0] = x_pcd;
        // cloud[3*cloud_idx + 1] = y_pcd;
        // cloud[3*cloud_idx + 2] = z_pcd;
    }

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
    __global__ void compute_render_cost(
        float* cuda_knn_dist,
        int* cuda_knn_index,
        int* cuda_cloud_pose_map,
        int* cuda_poses_occluded,
        float* cuda_rendered_cost,
        float sensor_resolution,
        int rendered_cloud_point_num,
        int observed_cloud_point_num,
        float* cuda_pose_point_num,
        uint8_t* rendered_cloud_color,
        uint8_t* observed_cloud_color,
        float* rendered_cloud,
        uint8_t* cuda_observed_explained,
        int* pose_segmentation_label,
        int* result_observed_cloud_label,
        int type)
    {
        /**
        * Params -
        * @cuda_knn_dist : distance to nn from knn library
        * @cuda_knn_index : index of nn in observed cloud from knn library
        * @cuda_cloud_pose_map : the pose corresponding to every point in cloud
        * @*_cloud_color : color values of clouds, to compare rgb cost of NNs
        * @rendered_cloud : rendered point cloud of all poses, all objects
        * Returns :
        * @cuda_pose_point_num : Number of points in each rendered pose
        */
        size_t point_index = blockIdx.x*blockDim.x + threadIdx.x;
        if(point_index >= rendered_cloud_point_num) return;

        int pose_index = cuda_cloud_pose_map[point_index];
        int o_point_index = cuda_knn_index[point_index];
        if (cuda_poses_occluded[pose_index])
        {
            cuda_rendered_cost[pose_index] = -1;
        }
        else
        {
            // count total number of points in this pose for normalization later
            atomicAdd(&cuda_pose_point_num[pose_index], 1);
            // float camera_z = rendered_cloud[point_index + 2 * rendered_cloud_point_num];
            // float cost = 10 * camera_z;
            float cost = 1.0;
            // printf("KKN distance : %f\n", cuda_knn_dist[point_index]);
            if (cuda_knn_dist[point_index] > sensor_resolution)
            {
                atomicAdd(&cuda_rendered_cost[pose_index], cost);
            }
            else
            {
                // compute color cost
                // printf("%d, %d\n", pose_segmentation_label[pose_index], result_observed_cloud_label[o_point_index]);
                uint8_t red2  = rendered_cloud_color[point_index + 2*rendered_cloud_point_num];
                uint8_t green2  = rendered_cloud_color[point_index + 1*rendered_cloud_point_num];
                uint8_t blue2  = rendered_cloud_color[point_index + 0*rendered_cloud_point_num];

                uint8_t red1  = observed_cloud_color[o_point_index + 2*observed_cloud_point_num];
                uint8_t green1  = observed_cloud_color[o_point_index + 1*observed_cloud_point_num];
                uint8_t blue1  = observed_cloud_color[o_point_index + 0*observed_cloud_point_num];

                if (type == 1)
                {
                    
                    float lab2[3];
                    rgb2lab(red2,green2,blue2,lab2);
                    float lab1[3];
                    rgb2lab(red1,green1,blue1,lab1);
                    double cur_dist = color_distance(lab1[0],lab1[1],lab1[2],lab2[0],lab2[1],lab2[2]);
                    // printf("color distance :%f\n", cur_dist);
                    if(cur_dist > 12.5){
                        // add to render cost if color doesnt match
                        atomicAdd(&cuda_rendered_cost[pose_index], cost);
                    }
                    else {
                        // the point is explained, so mark corresponding observed point explained
                        // atomicOr(cuda_observed_explained[o_point_index], 1);
                        cuda_observed_explained[o_point_index + pose_index * observed_cloud_point_num] = 1;
                    }
                }
                else if (type == 0) {
                    // the point is explained, so mark corresponding observed point explained
                    // atomicOr(cuda_observed_explained[o_point_index], 1);
                    cuda_observed_explained[o_point_index + pose_index * observed_cloud_point_num] = 1;
                }
                else if (type == 2) {
                    if (pose_segmentation_label[pose_index] != result_observed_cloud_label[o_point_index])
                    {
                        // the euclidean distance is fine, but segmentation labels dont match
                        atomicAdd(&cuda_rendered_cost[pose_index], cost);
                    }
                    else
                    {
                        // the point is explained, so mark corresponding observed point explained
                        // atomicOr(cuda_observed_explained[o_point_index], 1);
                        // float lab2[3];
                        // rgb2lab(red2,green2,blue2,lab2);

                        // float lab1[3];
                        // rgb2lab(red1,green1,blue1,lab1);

                        // double cur_dist = color_distance(lab1[0],lab1[1],lab1[2],lab2[0],lab2[1],lab2[2]);
                        // if(cur_dist > 30)
                        //     atomicAdd(&cuda_rendered_cost[pose_index], cost);
                        // else
                        cuda_observed_explained[o_point_index + pose_index * observed_cloud_point_num] = 1;
                    }
                }
            }
        }
    }
    __global__ void compute_observed_cost(
        int num_poses,
        int observed_cloud_point_num,
        uint8_t* cuda_observed_explained,
        float* observed_total_explained)
    {
        size_t point_index = blockIdx.x*blockDim.x + threadIdx.x;
        if(point_index >= num_poses * observed_cloud_point_num) return;

        size_t pose_index = point_index/observed_cloud_point_num;
        atomicAdd(&observed_total_explained[pose_index], (float) cuda_observed_explained[point_index]);
        // printf("%d\n", cuda_observed_explained[point_index]);
    }

}
#endif