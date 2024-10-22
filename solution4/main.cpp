#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

const int N = 10000;
std::vector<int> data(N);
std::vector<int> ldata(N);

const char* kernelSource = R"(
__kernel void bitonic_sort(__global int* data, const int j, const int k) {
    int index = get_global_id(0);
    int ixj = index ^ j;

    if (ixj > index) {
        if ((index & k) == 0) {
            if (data[index] > data[ixj]) {
                int temp = data[index];
                data[index] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            if (data[index] < data[ixj]) {
                int temp = data[index];
                data[index] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}
)";

void fill_random(std::vector<int>& array) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
    for (auto& elem : array) {
        elem = dis(gen);
    }
}

int next_power_of_two(int x) {
    return std::pow(2, std::ceil(std::log2(x)));
}

void bitonic_sort_opencl(std::vector<int>& input, int padded_size, const cl::Device& device) {
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cl::Buffer bufferData(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * padded_size, input.data());

    cl::Program program(context, kernelSource);
    program.build({device});

    for (int k = 2; k <= padded_size; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            cl::Kernel kernel(program, "bitonic_sort");
            kernel.setArg(0, bufferData);
            kernel.setArg(1, j);
            kernel.setArg(2, k);
            
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(padded_size), cl::NullRange);
            queue.finish();
        }
    }

    queue.enqueueReadBuffer(bufferData, CL_TRUE, 0, sizeof(int) * padded_size, input.data());
}

void sort_cpu(std::vector<int>& a) {
    bool swapped;
    for (int i = 0; i < a.size(); i++) {
        swapped = false;
        for (int j = 0; j < a.size() - 1; j++) {
            if (a[j] > a[j + 1]) {
                std::swap(a[j], a[j + 1]);
                swapped = true;
            }
        }
        if (!swapped)
            break;
    }
}

void validate(const std::vector<int>& data, const std::vector<int>& ldata) {
    for(int i = 0; i < ldata.size(); i++) {
        if(data[i] != ldata[i]) {
            std::cout << "Wrong result!" << std::endl;
            return;
        }
    }
    std::cout << "Valid result" << std::endl;
}

int main() {
    fill_random(data);
    ldata = data;

    int size = data.size();
    int padded_size = next_power_of_two(data.size());
    data.resize(padded_size);
    ldata.resize(padded_size);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    sort_cpu(ldata);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "Time taken CPU: " << cpu_duration.count() << " seconds\n";

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (const auto& platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        for (const auto& device : devices) {
            std::cout << "Running on device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

            auto start_gpu = std::chrono::high_resolution_clock::now();
            bitonic_sort_opencl(data, padded_size, device);
            auto end_gpu = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
            std::cout << "Time taken GPU: " << gpu_duration.count() << " seconds\n";

            data.erase(data.begin(), data.begin() + padded_size - size);
            ldata.erase(ldata.begin(), ldata.begin() + padded_size - size);

            validate(data, ldata);
        }
    }

    return 0;
}
