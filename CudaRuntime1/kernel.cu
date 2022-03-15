
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <stdio.h>

struct Coordinate
{
    double x, y;

    Coordinate(double a, double b)
    {
        x = a;
        y = b;
    }
};

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

cudaError_t pushBackWithCuda(const std::vector<Coordinate>* coordinates, std::vector<double> vec, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

//__global__ void pushBackKernel(const std::vector<Coordinate>* coordinates, std::vector<double>* vec)
//{
//    int i = threadIdx.x;
//    vec[i].push_back()
//}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size >>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

// Helper function for using CUDA to push back doubles in parallel.
//cudaError_t pushBackWithCuda(const std::vector<Coordinate>* coordinates, std::vector<double>* vec, unsigned int size)
//{
//    double* dev_coord = 0;
//    double* dev_vec = 0;
//    cudaError_t cudaStatus;
//    size_t req_size = size * sizeof(double);
//
//    // Choose GPU
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Clean;
//    }
//
//    // Allocate GPU memory for one input and one output
//    cudaStatus = cudaMalloc((void**)&dev_coord, req_size);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Clean;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_vec, req_size);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Clean;
//    }
//
//    // Copy input data from CPU to GPU
//    cudaStatus = cudaMemcpy(dev_coord, coordinates, req_size, cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Clean;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element
//    addKernel<<<1, size>>>(dev_coord, dev_vec);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Clean;
//    }
//
//    // Wait for kernel to finish
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Clean;
//    }
//
//    // Copy output data from GPU to CPU
//    cudaStatus = cudaMemcpy(vec, dev_vec, req_size, cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Clean;
//    }
//
//    // Clean up after operation
//    goto Clean;
//
//Clean:
//    cudaFree(dev_coord);
//    cudaFree(dev_vec);
//
//    return cudaStatus;
//}

double CalculateGradient(const std::vector<Coordinate>& coordinates)
{

    // num of coordinate
    int n = coordinates.size();

    // sum of x*y
    // loop through all coordinates
    // push corresponding x*y to vector
    // add all elements in the vector
    std::vector<double> vec_xy;
    for (int i = 0; i < n; i++)
    {
        vec_xy.push_back(coordinates.at(i).x * coordinates.at(i).y);
    }
    double sum_xy = std::accumulate(vec_xy.begin(), vec_xy.end(), 0.0);

    // sum of x
    std::vector<double> vec_x;
    for (int i = 0; i < n; i++)
    {
        vec_x.push_back(coordinates.at(i).x);
    }
    double sum_x = std::accumulate(vec_x.begin(), vec_x.end(), 0.0);

    // sum of y
    std::vector<double> vec_y;
    for (int i = 0; i < n; i++)
    {
        vec_y.push_back(coordinates.at(i).y);
    }
    double sum_y = std::accumulate(vec_y.begin(), vec_y.end(), 0.0);

    // sum of x*x
    std::vector<double> vec_xx;
    for (int i = 0; i < n; i++)
    {
        vec_xx.push_back(coordinates.at(i).x * coordinates.at(i).x);
    }
    double sum_xx = std::accumulate(vec_xx.begin(), vec_xx.end(), 0.0);

    return (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
}

double CalculateYIntercept(const std::vector<Coordinate>& coordinates, const double gradient)
{
    // num of coordinates
    int n = coordinates.size();

    // sum of y
    std::vector<double> vec_y;
    for (int i = 0; i < n; i++)
    {
        vec_y.push_back(coordinates.at(i).y);
    }
    double sum_y = std::accumulate(vec_y.begin(), vec_y.end(), 0.0);

    // sum of x
    std::vector<double> vec_x;
    for (int i = 0; i < n; i++)
    {
        vec_x.push_back(coordinates.at(i).x);
    }
    double sum_x = std::accumulate(vec_x.begin(), vec_x.end(), 0.0);

    return (1 / (double)n) * (sum_y - gradient * sum_x);
}


int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Test data: y = 2x + 2
    const std::vector<Coordinate> coordinates = { Coordinate{-1, 0}, Coordinate{0,2}, Coordinate{2,6}, Coordinate{4,10}, Coordinate{5,12} };

    const double gradient = CalculateGradient(coordinates);

    const double yIntercept = CalculateYIntercept(coordinates, gradient);

    printf("Gradient: %.2f, y-Intercept: %.2f\n", gradient, yIntercept);

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}