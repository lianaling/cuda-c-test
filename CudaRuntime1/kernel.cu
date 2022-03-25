
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <stdio.h>

#include "Timer.h"

const unsigned int ARR_SIZE = 1024;

struct Coordinate
{
    double x, y;

    Coordinate(double a, double b)
    {
        x = a;
        y = b;
    }
};

enum TargetCoordinateOperation
{
    X, Y, XX, XY
};

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

//cudaError_t pushBackWithCuda(const std::vector<Coordinate>* coordinates, std::vector<double> vec, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void pushBackKernel(const Coordinate *coord, double *vec, TargetCoordinateOperation op)
{
    int i = threadIdx.x;
    if (op == X)
        vec[i] = coord[i].x;
    else if (op == Y)
        vec[i] = coord[i].y;
    else if (op == XY)
        vec[i] = coord[i].x * coord[i].y;
    else if (op == XX)
        vec[i] = coord[i].x * coord[i].x;
}

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
cudaError_t pushBackWithCuda(const Coordinate* coordinates, double* vector, unsigned int size, TargetCoordinateOperation op)
{
    Coordinate* dev_coord;
    double* dev_vec;
    cudaError_t cudaStatus;
    //size_t req_size = size * sizeof(double);
    /*double coord[ARR_SIZE];
    double vec[ARR_SIZE];*/

    //for (int i = 0; i < size; i++)
    //{
    //    //coord[i] = coordinates->at(i).x * coordinates->at(i).y;
    //    //vec[i] = vector->at(i);
    //}

    // Choose GPU
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Clean;
    }

    // Allocate GPU memory for one input and one output
    cudaStatus = cudaMalloc((void**)&dev_coord, size * sizeof(Coordinate));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Clean;
    }

    cudaStatus = cudaMalloc((void**)&dev_vec, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Clean;
    }

    // Copy input data from CPU to GPU
    cudaStatus = cudaMemcpy(dev_coord, coordinates, size * sizeof(Coordinate), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Clean;
    }

    // Launch a kernel on the GPU with one thread for each element
    pushBackKernel<<<1, size >>>(dev_coord, dev_vec, op);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Clean;
    }

    // Wait for kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Clean;
    }

    // Copy output data from GPU to CPU
    cudaStatus = cudaMemcpy(vector, dev_vec, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Clean;
    }

    /*for (int i = 0; i < size; i++) {
        vector->push_back(vec[i]);
    }*/

    // Clean up after operation
    goto Clean;

Clean:
    cudaFree(dev_coord);
    cudaFree(dev_vec);

    return cudaStatus;
}

double accumulate(double sum, int n, const double vec[])
{
    for (int i = 0; i < n; i++) {
        sum += vec[i];
    }
    return sum;
}

double CalculateGradient(const std::vector<Coordinate>& coordinates)
{
    auto timer = Timer("CalculateGradient");

    //printf("3) x: %.2f y: %.2f\n", coordinates.at(2).x, coordinates.at(2).y);

    // num of coordinate
    int n = coordinates.size();
    // Declare array pointer
    const Coordinate* coord = &coordinates[0];

    // sum of x*y
    // loop through all coordinates
    // push corresponding x*y to vector
    // add all elements in the vector
    double vec_xy[ARR_SIZE], sum_xy = 0;
    /*for (int i = 0; i < n; i++)
    {
        vec_xy.push_back(coordinates.at(i).x * coordinates.at(i).y);
    }
    double sum_xy = std::accumulate(vec_xy.begin(), vec_xy.end(), 0.0);*/

    //printf("3) x: %.2f y: %.2f\n", coord[2].x, coord[2].y);
    pushBackWithCuda(coord, vec_xy, n, XY);

    //double sum_xy = std::accumulate(vec_xy.begin(), vec_xy.end(), 0.0);
    //for (int i = 0; i < n; i++) {
    //    sum_xy += vec_xy[i];
    //    printf("%d) x: %.2f y: %.2f Result: %.2f\n", i, coord[i].x, coord[i].y, vec_xy[i]); // FIXME: Coordinates all 0
    //}
    sum_xy = accumulate(sum_xy, n, vec_xy);
    printf("sum_xy: %.2f\n", sum_xy);

    // sum of x
    //std::vector<double> vec_x;
    double vec_x[ARR_SIZE], sum_x = 0;
    pushBackWithCuda(coord, vec_x, n, X);
    sum_x = accumulate(sum_x, n, vec_x);
    /*for (int i = 0; i < n; i++)
    {
        vec_x.push_back(coordinates.at(i).x);
    }*/
    //double sum_x = std::accumulate(vec_x.begin(), vec_x.end(), 0.0);

    // sum of y
    //std::vector<double> vec_y;
    double vec_y[ARR_SIZE], sum_y = 0;
    pushBackWithCuda(coord, vec_y, n, Y);
    sum_y = accumulate(sum_y, n, vec_y);
    /*for (int i = 0; i < n; i++)
    {
        vec_y.push_back(coordinates.at(i).y);
    }
    double sum_y = std::accumulate(vec_y.begin(), vec_y.end(), 0.0);*/

    // sum of x*x
    double vec_xx[ARR_SIZE], sum_xx = 0;
    pushBackWithCuda(coord, vec_xx, n, XX);
    sum_xx = accumulate(sum_xx, n, vec_xx);
    /*std::vector<double> vec_xx;
    for (int i = 0; i < n; i++)
    {
        vec_xx.push_back(coordinates.at(i).x * coordinates.at(i).x);
    }
    double sum_xx = std::accumulate(vec_xx.begin(), vec_xx.end(), 0.0);*/

    return (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    //return 0.0;
}

double CalculateYIntercept(const std::vector<Coordinate>& coordinates, const double gradient)
{
    // num of coordinates

    int n = coordinates.size();
    const Coordinate* coord = &coordinates[0];

    // sum of y
    //std::vector<double> vec_y;
    double vec_y[ARR_SIZE], sum_y = 0;
    pushBackWithCuda(coord, vec_y, n, Y);
    sum_y = accumulate(sum_y, n, vec_y);
    /*for (int i = 0; i < n; i++)
    {
        vec_y.push_back(coordinates.at(i).y);
    }
    double sum_y = std::accumulate(vec_y.begin(), vec_y.end(), 0.0);*/

    // sum of x
    //std::vector<double> vec_x;
    double vec_x[ARR_SIZE], sum_x = 0;
    pushBackWithCuda(coord, vec_x, n, X);
    sum_x = accumulate(sum_x, n, vec_x);
    /*for (int i = 0; i < n; i++)
    {
        vec_x.push_back(coordinates.at(i).x);
    }
    double sum_x = std::accumulate(vec_x.begin(), vec_x.end(), 0.0);*/

    return (1 / (double)n) * (sum_y - gradient * sum_x);
}


int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Test data: y = 2x + 2
    std::vector<Coordinate> coordinates = { Coordinate{-1, 0}, Coordinate{0,2}, Coordinate{2,6}, Coordinate{4,10}, Coordinate{5,12} };

    printf("main(): coordinates - x is %.2f and y is %.2f\n", coordinates.at(2).x, coordinates.at(2).y); // Coordinate values working

    const double gradient = CalculateGradient(coordinates);

    const double yIntercept = CalculateYIntercept(coordinates, gradient);

    printf("Gradient: %.2f, y-Intercept: %.2f\n", gradient, yIntercept);

    // Add vectors in parallel.
    /*cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);*/

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    /*std::vector<int> test{ 1,2,3,4,5 };
    int* test_arr = &test[0];

    for (int i = 0; i < 5; i++)
        std::cout << test_arr[i] << " ";
    std::cout << "\n";*/

    return 0;
}