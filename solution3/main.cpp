#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cstdlib>
#include <chrono>

const int N = 65573;
std::vector<float> a(N);
std::vector<float> C(1);
float lres = 0;
float res = 0;

const char* kernelSource = R"(
__kernel void reduce_sum(__global const float* a, __global float* result, const int n) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    __local float local_sum[256];

    float sum = 0.0f;
    for (int i = gid; i < n; i += get_global_size(0)) {
        sum += a[i];
    }
    local_sum[lid] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = group_size / 2; offset > 0; offset >>= 1) {
        if (lid < offset) {
            local_sum[lid] += local_sum[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        result[get_group_id(0)] = local_sum[0];
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
    
    size_t localSize = 256;
    size_t numGroups = (N + localSize - 1) / localSize;
    cl::Buffer result(context, CL_MEM_WRITE_ONLY, sizeof(float) * numGroups);

    cl::Program program(context, kernelSource);
    program.build({device});

    cl::Kernel kernel(program, "reduce_sum");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, result);
    kernel.setArg(2, N);
    

    cl::NDRange globalSize(numGroups * localSize);
    cl::NDRange localSizeRange(localSize);

    cl::Event event;

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSizeRange, nullptr, &event);
    queue.finish();

    cl_ulong start_time, end_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
    float gpu_duration = (end_time - start_time) * 1e-9;
    std::cout << "Time taken gpu: " << gpu_duration << std::endl;

    std::vector<float> partialSums(numGroups);
    queue.enqueueReadBuffer(result, CL_TRUE, 0, sizeof(float) * numGroups, partialSums.data());

    for (float partial : partialSums) {
        res += partial;
    }
    std::cout << "GPU sum: " << res << std::endl;
}


void add_loop() {
    for(int i = 0; i < a.size(); i++) {
        lres += a[i];
    }
}

int main() {
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_real_distribution<> distr(1.0, 10.0);
    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<float>(distr(eng));
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    add_loop();
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "Time taken cpu: " << cpu_duration.count() << " seconds\n";
    std::cout << "CPU sum: " << lres << std::endl;

    add_opencl();

    return 0;
}