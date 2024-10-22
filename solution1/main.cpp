#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

const int N = 100000;
std::vector<float> a(N);
std::vector<float> b(N);
std::vector<float> C(N);
std::vector<float> L(N);

const char* kernelSource = R"(
__kernel void vector_add(__global const float* A, __global const float* B, __global float* C, const int N) {
    int i = get_global_id(0);
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
)";

void add_opencl() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, a.data());
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, b.data());
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * N);

    cl::Program program(context, kernelSource);
    program.build({device});

    cl::Kernel kernel(program, "vector_add");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    kernel.setArg(3, N);

    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NullRange, nullptr, &event);
    queue.finish();

    cl_ulong start_time, end_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
    float gpu_duration = (end_time - start_time) * 1e-9;
    std::cout << "Time taken gpu: " << gpu_duration << std::endl;

    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * N, C.data());
}


void add_loop() {
    for(int i = 0; i < a.size(); i++) {
        L[i] = a[i] + b[i];
    }
}

void validate() {
    for(int i = 0; i < C.size(); i++) {
        if(C[i] != L[i]) {
            std::cout << "Wrong result!" << std::endl;
            return;
        }
    }
    std::cout << "Valid result" << std::endl;
}

void fill_random(std::vector<float>& v) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 10.0f);

    for (auto& elem : v) {
        elem = dis(gen);
    }
}

int main() {
    fill_random(a);
    fill_random(b);

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (auto& platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        for (auto& device : devices) {
            std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

            cl::Context context(device);
            cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

            cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, a.data());
            cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, b.data());
            cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * N);

            cl::Program program(context, kernelSource);
            program.build({device});

            cl::Kernel kernel(program, "vector_add");
            kernel.setArg(0, bufferA);
            kernel.setArg(1, bufferB);
            kernel.setArg(2, bufferC);
            kernel.setArg(3, N);

            cl::Event event;
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NullRange, nullptr, &event);
            queue.finish();

            cl_ulong start_time, end_time;
            event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
            event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
            float gpu_duration = (end_time - start_time) * 1e-9;
            std::cout << "Time taken on " << device.getInfo<CL_DEVICE_NAME>() << " (GPU): " << gpu_duration << " seconds\n";

            auto start_cpu = std::chrono::high_resolution_clock::now();
            add_loop();
            auto end_cpu = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
            std::cout << "Time taken on CPU: " << cpu_duration.count() << " seconds\n";

            validate();
        }
    }

    return 0;
}
