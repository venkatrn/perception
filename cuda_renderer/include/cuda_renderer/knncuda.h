/**
 * For each input query point, locates the k-NN (indexes and distances) among the reference points.
 * This implementation uses global memory to store reference and query points.
 *
 * @param ref        refence points
 * @param ref_nb     number of reference points
 * @param query      query points
 * @param query_nb   number of query points
 * @param dim        dimension of points
 * @param k          number of neighbors to consider
 * @param knn_dist   output array containing the query_nb x k distances
 * @param knn_index  output array containing the query_nb x k indexes
 */
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>
#include <iostream> 
#include <vector>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

namespace cuda_renderer {
    

    bool depth2cloud_global(int32_t* depth_data, 
                            std::vector<std::vector<u_int8_t>>& color_data,
                            float* &result_cloud, 
                            uint8_t* &result_cloud_color,
                            int* &dc_index,
                            int &point_num,
                            int* &cloud_pose_map,
                            int width, 
                            int height, 
                            int num_poses,
                            int* pose_occluded,
                            float kCameraCX, 
                            float kCameraCY, 
                            float kCameraFX, 
                            float kCameraFY,
                            float depth_factor,
                            int stride,
                            int point_dim,
                            int* &result_observed_cloud_label,
                            uint8_t* label_mask_data = NULL);

    bool compute_rgbd_cost(
        float &sensor_resolution,
        float* knn_dist,
        int* knn_index,
        int* poses_occluded,
        int* cloud_pose_map,
        float* result_observed_cloud,
        uint8_t* result_observed_cloud_color,
        float* result_rendered_cloud,
        uint8_t* result_rendered_cloud_color,
        int rendered_cloud_point_num,
        int observed_cloud_point_num,
        int num_poses,
        float* &rendered_cost,
        std::vector<float> pose_observed_points_total,
        float* &observed_cost,
        int* pose_segmentation_label,
        int* result_observed_cloud_label,
        int cost_type,
        bool calculate_observed_cost);

    bool knn_cuda_global(const float * ref,
                        int           ref_nb,
                        const float * query,
                        int           query_nb,
                        int           dim,
                        int           k,
                        float *       knn_dist,
                        int *         knn_index);


    /**
     * For each input query point, locates the k-NN (indexes and distances) among the reference points.
     * This implementation uses texture memory for storing reference points  and memory to store query points.
     *
     * @param ref        refence points
     * @param ref_nb     number of reference points
     * @param query      query points
     * @param query_nb   number of query points
     * @param dim        dimension of points
     * @param k          number of neighbors to consider
     * @param knn_dist   output array containing the query_nb x k distances
     * @param knn_index  output array containing the query_nb x k indexes
     */
    bool knn_cuda_texture(const float * ref,
                        int           ref_nb,
                        const float * query,
                        int           query_nb,
                        int           dim,
                        int           k,
                        float *       knn_dist,
                        int *         knn_index);


    /**
     * For each input query point, locates the k-NN (indexes and distances) among the reference points.
     * Using cuBLAS, the computation of the distance matrix can be faster in some cases than other
     * implementations despite being more complex.
     *
     * @param ref        refence points
     * @param ref_nb     number of reference points
     * @param query      query points
     * @param query_nb   number of query points
     * @param dim        dimension of points
     * @param k          number of neighbors to consider
     * @param knn_dist   output array containing the query_nb x k distances
     * @param knn_index  output array containing the query_nb x k indexes
     */
    bool knn_cublas(const float * ref,
                    int           ref_nb,
                    const float * query,
                    int           query_nb,
                    int           dim,
                    int           k,
                    float *       knn_dist,
                    int *         knn_index);

    bool knn_test(const float * ref,
                int           ref_nb,
                const float * query,
                int           query_nb,
                int           dim,
                int           k,
                float *       knn_dist,
                int *         knn_index);
}


