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

namespace cuda_renderer {
bool depth2cloud_global(
    int32_t* depth_data, 
    float* &result_cloud, 
    int* &dc_index,
    int &point_num,
    int width, 
    int height, 
    int num_poses,
    float kCameraCX, 
    float kCameraCY, 
    float kCameraFX, 
    float kCameraFY,
    float depth_factor
    );

// bool depth2cloud_global(
//     std::vector<int32_t> result_depth,  
//     int width, 
//     int height, 
//     cv::Mat cam_intrinsics);

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
}


