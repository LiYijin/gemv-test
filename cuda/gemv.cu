#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cmath>

// CUDA 内核实现的 GEMV
__global__ void gemv_cuda(int M, int N, const float* A, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[row * N + j] * x[j];
        }
        y[row] = sum;
    }
}

// 检查 CUDA 错误
void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 检查 cuBLAS 错误
void checkCublasError(cublasStatus_t err) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error: " << err << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 检查两个浮点数是否近似相等
bool areEqual(float a, float b, float epsilon = 1e-5f) {
    return fabs(a - b) < epsilon;
}

int main() {
    const int M = 1000; // 行数
    const int N = 1000; // 列数

    // 初始化数据
    float* h_A = new float[M * N];
    float* h_x = new float[N];
    float* h_y_cuda = new float[M];
    float* h_y_cublas = new float[M];

    for (int i = 0; i < M * N; ++i) h_A[i] = static_cast<float>(rand() % 10) / RAND_MAX;
    for (int i = 0; i < N; ++i) h_x[i] = static_cast<float>(rand() % 10) / RAND_MAX;

    float *d_A, *d_x, *d_y_cuda, *d_y_cublas;
    checkCudaError(cudaMalloc((void**)&d_A, M * N * sizeof(float)));
    checkCudaError(cudaMalloc((void**)&d_x, N * sizeof(float)));
    checkCudaError(cudaMalloc((void**)&d_y_cuda, M * sizeof(float)));
    checkCudaError(cudaMalloc((void**)&d_y_cublas, M * sizeof(float)));

    // 拷贝数据到设备
    checkCudaError(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    // CUDA GEMV 热身
    gemv_cuda<<<M, 1>>>(M, N, d_A, d_x, d_y_cuda);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        // 处理错误或退出
    }    
    cudaDeviceSynchronize(); // 等待内核完成

    // cuBLAS 初始化
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle));

    // cuBLAS GEMV 热身
    float alpha = 1.0f;
    float beta = 0.0f;
    checkCublasError(cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_x, 1, &beta, d_y_cublas, 1));
    cudaDeviceSynchronize(); // 等待内核完成
    // 拷贝结果回主机
    checkCudaError(cudaMemcpy(h_y_cuda, d_y_cuda, M * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(h_y_cublas, d_y_cublas, M * sizeof(float), cudaMemcpyDeviceToHost));
    // 检查结果是否一致
    bool isEqual = true;
    for (int i = 0; i < M; ++i) {
        if (!areEqual(h_y_cuda[i], h_y_cublas[i])) {
            isEqual = false;
            std::cerr << "Mismatch at index " << i << ": CUDA result = " << h_y_cuda[i] << ", cuBLAS result = " << h_y_cublas[i] << std::endl;
            break;
        }
    }

    if (isEqual) {
        std::cout << "Results are consistent!" << std::endl;
    } else {
        std::cerr << "Results are inconsistent!" << std::endl;
    }


    // 计时变量
    double totalTimeCuda = 0.0;
    double totalTimeCublas = 0.0;
    const int iterations = 100;

    // CUDA GEMV 迭代
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        gemv_cuda<<<1, M>>>(M, N, d_A, d_x, d_y_cuda);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        totalTimeCuda += std::chrono::duration<double>(end - start).count();
    }

    // cuBLAS GEMV 迭代
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        checkCublasError(cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_x, 1, &beta, d_y_cublas, 1));
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        totalTimeCublas += std::chrono::duration<double>(end - start).count();
    }

    // 计算平均时间
    double averageTimeCuda = totalTimeCuda / iterations;
    double averageTimeCublas = totalTimeCublas / iterations;



    // 计算带宽 (以 GB/s 为单位)
    double bandwidthCuda = (2.0 * M * N * sizeof(float)) / (averageTimeCuda * 1e9); // GB/s
    double bandwidthCublas = (2.0 * M * N * sizeof(float)) / (averageTimeCublas * 1e9); // GB/s

    std::cout << "CUDA Average time taken: " << averageTimeCuda * 1e6 << " us" << std::endl;
    std::cout << "CUDA Bandwidth: " << bandwidthCuda << " GB/s" << std::endl;

    std::cout << "cuBLAS Average time taken: " << averageTimeCublas * 1e6 << " us" << std::endl;
    std::cout << "cuBLAS Bandwidth: " << bandwidthCublas << " GB/s" << std::endl;


    // 清理资源
    checkCublasError(cublasDestroy(handle));
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y_cuda);
    cudaFree(d_y_cublas);
    delete[] h_A;
    delete[] h_x;
    delete[] h_y_cuda;
    delete[] h_y_cublas;

    return 0;
}
