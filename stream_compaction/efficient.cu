#include "common.h"
#include "efficient.h"
// #include <__clang_cuda_builtin_vars.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace StreamCompaction {
namespace Efficient {
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
  static PerformanceTimer timer;
  return timer;
}

const int threadsPerBlock = 1024;

__device__ int log2(int x) {
  int lg = 0;
  while (x >>= 1) {
    ++lg;
  }
  return lg;
}

__device__ int log2ceil(int x) { return x == 1 ? 0 : log2(x - 1) + 1; }

__global__ void kernUpSweep(int n, int *data) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid >= n) {

    return;
  }

  int d_pow = 1;
  for (int d = 0; d <= log2ceil(n) - 1; d++) {
    if ((n - tid) % (d_pow * 2) == 0) {
      data[tid + (d_pow * 2) - 1] += data[tid + d_pow - 1];
    }

    d_pow *= 2;
    __syncthreads();
  }
}

__global__ void kernDownSweep(int n, int *data) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid >= n) {
    return;
  }

  if (tid == n - 1) {
    data[tid] = 0;
  }

  int max_d = log2ceil(n) - 1;
  int d_pow = (int)powf(2, max_d);
  for (int d = max_d; d >= 0; d--) {
    if ((n - tid) % (d_pow * 2) == 0) {
      int t = data[tid + d_pow - 1];
      data[tid + d_pow - 1] = data[tid + (d_pow * 2) - 1];
      data[tid + (d_pow * 2) - 1] += t;
    }
    d_pow /= 2;
    __syncthreads();
  }
  __syncthreads();
}

int round_to_next_pow2(int num) {
  int n = num;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

bool is_power_of_2(unsigned int x) { return x && ((x & (x - 1)) == 0); }

void parallel_scan_power2(int n, int *data_device) {
  int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  kernUpSweep<<<numBlocks, threadsPerBlock>>>(n, data_device);
  kernDownSweep<<<numBlocks, threadsPerBlock>>>(n, data_device);
}

void parallel_scan(int n, int *data_device) {
  if (n == 0) {
    return;
  }

  int size = n;
  if (!is_power_of_2(n)) {
    size = round_to_next_pow2(n);
  }

  if (size == n) {
    parallel_scan_power2(size, data_device);
    return;
  }

  int *data_device_padded;
  cudaMalloc((void **)&data_device_padded, sizeof(int) * size);
  cudaMemcpy(data_device_padded, data_device, sizeof(int) * n,
             cudaMemcpyDeviceToDevice);
  cudaMemset(data_device_padded + n, 0, (size - n) * sizeof(int));

  parallel_scan_power2(size, data_device_padded);

  cudaMemcpy(data_device, data_device_padded, sizeof(int) * n,
             cudaMemcpyDeviceToDevice);
  cudaFree(data_device_padded);
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  if (n == 0) {
    return;
  }

  int *data_device;
  cudaMalloc((void **)&data_device, sizeof(int) * n);
  cudaMemcpy(data_device, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

  timer().startGpuTimer();
  parallel_scan(n, data_device);
  timer().endGpuTimer();

  cudaMemcpy(odata, data_device, sizeof(int) * n, cudaMemcpyDeviceToHost);
  cudaFree(data_device);
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {

  // Do padding if not power of 2
  int size = n;
  if (!is_power_of_2(n)) {
    size = round_to_next_pow2(n);
  }

  int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  // Setup device global memory
  int *d_idata;
  int *d_odata;
  int *d_bools;
  int *d_indices;

  cudaMalloc((void **)&d_idata, sizeof(int) * size);
  cudaMemcpy(d_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
  cudaMemset(d_idata + n, 0, (size - n) * sizeof(int));

  cudaMalloc((void **)&d_indices, sizeof(int) * size);
  cudaMalloc((void **)&d_odata, sizeof(int) * size);
  cudaMalloc((void **)&d_bools, sizeof(int) * size);

  // Start computation
  timer().startGpuTimer();

  // Map and copy to new buffer for in place scan
  StreamCompaction::Common::kernMapToBoolean<<<numBlocks, threadsPerBlock>>>(
      size, d_bools, d_idata, StreamCompaction::Common::IsNonZero{});
  cudaMemcpy(d_indices, d_bools, sizeof(int) * size, cudaMemcpyDeviceToDevice);

  // parallel scan to get indices
  parallel_scan_power2(size, d_indices);

  // Scatter to get output array
  StreamCompaction::Common::kernScatter<<<numBlocks, threadsPerBlock>>>(
      size, d_odata, d_idata, d_bools, d_indices);

  timer().endGpuTimer();
  int last_bool;
  int last_index;
  cudaMemcpy(&last_bool, d_bools + size - 1, sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&last_index, d_indices + size - 1, sizeof(int),
             cudaMemcpyDeviceToHost);

  int size_out = last_bool + last_index;
  cudaMemcpy(odata, d_odata, sizeof(int) * size_out, cudaMemcpyDeviceToHost);

  // Free up allmemory
  cudaFree(d_idata);
  cudaFree(d_odata);
  cudaFree(d_bools);
  cudaFree(d_indices);

  // Return
  return size_out;
}
} // namespace Efficient
} // namespace StreamCompaction
