#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

const int rowsA = 500;
const int colsA = 500;
const int colsB = 500;

std::vector<float> matrixA(rowsA * colsA);
std::vector<float> matrixB(colsA * colsB);
std::vector<float> resultGPU(rowsA * colsB);
std::vector<float> resultCPU(rowsA * colsB);

const char* kernelSource = R"(
__kernel void matrix_multiply(__global const float* A, __global const float* B, __global float* C, const int rowsA, const int colsA, const int colsB) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; ++k) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}
)";

void fill_random(std::vector<float>& matrix) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    for (auto& elem : matrix) {
        elem = dis(gen);
    }
}

void print_matrix(const std::vector<float>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void multiply_opencl() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * rowsA * colsA, matrixA.data());
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * colsA * colsB, matrixB.data());
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * rowsA * colsB);

    cl::Program program(context, kernelSource);
    program.build({device});

    cl::Kernel kernel(program, "matrix_multiply");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    kernel.setArg(3, rowsA);
    kernel.setArg(4, colsA);
    kernel.setArg(5, colsB);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rowsA, colsB), cl::NullRange);
    queue.finish();
    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * rowsA * colsB, resultGPU.data());
}

void multiply_loop() {
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; ++k) {
                sum += matrixA[i * colsA + k] * matrixB[k * colsB + j];
            }
            resultCPU[i * colsB + j] = sum;
        }
    }
}

void validate(std::vector<float>& cpu_result, std::vector<float>& gpu_result, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (cpu_result[i * cols + j] != gpu_result[i * cols + j]) {
                std::cout << "Wrong result!" << std::endl;
                return;
            }
        }
    }
    std::cout << "Result valid" << std::endl;
}

int main() {
    fill_random(matrixA);
    fill_random(matrixB);

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (auto& platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        for (auto& device : devices) {
            std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

            auto start_gpu = std::chrono::high_resolution_clock::now();
            multiply_opencl();
            auto end_gpu = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
            std::cout << "Time taken on " << device.getInfo<CL_DEVICE_NAME>() << " (GPU): " << gpu_duration.count() << " seconds\n";

            auto start_cpu = std::chrono::high_resolution_clock::now();
            multiply_loop();
            auto end_cpu = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
            std::cout << "Time taken on CPU: " << cpu_duration.count() << " seconds\n";

            validate(resultCPU, resultGPU, rowsA, colsB);
        }
    }

    return 0;
}