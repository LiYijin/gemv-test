#include <CL/cl.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

const char* kernelSource = R"(
__kernel void gemv(int M, int N, __global const float* A, __global const float* x, __global float* y) {
    int row = get_global_id(0);
    if (row < M) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[row * N + j] * x[j];
        }
        y[row] = sum;
    }
}
)";

void checkError(cl_int err) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL error: " << err << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 检查两个浮点数是否近似相等
bool areEqual(float a, float b, float epsilon = 1e-5f) {
    return fabs(a - b) < epsilon;
}

void gemv_cpu(int M, int N, float* A, float* x, float* y) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

int main() {
    const int M = 4096; // 行数
    const int N = 4096; // 列数

    // 初始化数据
    std::vector<float> h_A(M * N);
    std::vector<float> h_x(N);
    std::vector<float> h_y(M, 0.0f);
    std::vector<float> h_y_cpu(M, 0.0f);

    for (int i = 0; i < M * N; ++i) h_A[i] = static_cast<float>(rand() % 10) / RAND_MAX;
    for (int i = 0; i < N; ++i) h_x[i] = static_cast<float>(rand() % 10) / RAND_MAX;

    // 做CPU计算，以对比opencl的结果是否正确
    gemv_cpu(M, N, h_A.data(), h_x.data(), h_y_cpu.data());
    // OpenCL 变量
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_mem d_A, d_x, d_y;

    // 获取平台和设备
    err = clGetPlatformIDs(1, &platform, nullptr);
    checkError(err);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    checkError(err);

    // 创建上下文和命令队列
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkError(err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err);

    // 创建缓冲区
    d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, M * N * sizeof(float), h_A.data(), &err);
    checkError(err);
    d_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), h_x.data(), &err);
    checkError(err);
    d_y = clCreateBuffer(context, CL_MEM_WRITE_ONLY, M * sizeof(float), nullptr, &err);
    checkError(err);

    // 创建并编译内核
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    checkError(err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    checkError(err);

    cl_kernel kernel = clCreateKernel(program, "gemv", &err);
    checkError(err);

    // 热身
    {
        err = clSetKernelArg(kernel, 0, sizeof(int), &M);
        err |= clSetKernelArg(kernel, 1, sizeof(int), &N);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_A);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_x);
        err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_y);
        checkError(err);

        size_t globalWorkSize = M;
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
        clFinish(queue); // 等待内核完成
    }

    // 计时变量
    double totalTime = 0.0;
    const int iterations = 100;

    for (int i = 0; i < iterations; ++i) {
        err = clSetKernelArg(kernel, 0, sizeof(int), &M);
        err |= clSetKernelArg(kernel, 1, sizeof(int), &N);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_A);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_x);
        err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_y);
        checkError(err);

        // 计时开始
        auto start = std::chrono::high_resolution_clock::now();

        // 执行内核
        size_t globalWorkSize = M;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
        checkError(err);
        clFinish(queue); // 等待内核完成

        // 计时结束
        auto end = std::chrono::high_resolution_clock::now();
        totalTime += std::chrono::duration<double>(end - start).count();
    }

    // 计算平均时间
    double averageTime = totalTime / iterations;

    // 拷贝结果回主机
    err = clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, M * sizeof(float), h_y.data(), 0, nullptr, nullptr);
    checkError(err);

    // 检查结果是否一致
    bool isEqual = true;
    for (int i = 0; i < M; ++i) {
        if (!areEqual(h_y_cpu[i], h_y[i])) {
            isEqual = false;
            std::cerr << "Mismatch at index " << i << ": CPU result = " << h_y_cpu[i] << ", OpenCL result = " << h_y[i] << std::endl;
            break;
        }
    }

    if (isEqual) {
        std::cout << "Results are consistent!" << std::endl;
    } else {
        std::cerr << "Results are inconsistent!" << std::endl;
    }

    // 计算带宽 (以 GB/s 为单位)
    double bandwidth = (2.0 * M * N * sizeof(float)) / (averageTime * 1e9); // GB/s
    std::cout << "Average time taken: " << averageTime * 1e6 << " us" << std::endl;
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // 清理资源
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_y);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

