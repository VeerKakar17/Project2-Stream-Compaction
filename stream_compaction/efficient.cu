#include "common.h"
#include "efficient.h"
// #include <__clang_cuda_builtin_vars.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) (((n) >> LOG_NUM_BANKS) + ((n) >> (2 * LOG_NUM_BANKS)))

namespace StreamCompaction {
namespace Efficient {
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
  static PerformanceTimer timer;
  return timer;
}

const int threadsPerBlock = 256;

__device__ int log2(int x) {
  int lg = 0;
  while (x >>= 1) {
    ++lg;
  }
  return lg;
}

__device__ int log2ceil(int x) { return x == 1 ? 0 : log2(x - 1) + 1; }

__global__ void kernScan(int n, int *data, int *d_block_sums) {
  //  up sweep
  int block_offset = 2 * blockDim.x * blockIdx.x;
  int chunk_size = min(2 * blockDim.x, n - block_offset);
  int tid = threadIdx.x;

  if (tid >= n / 2) {
    return;
  }

  
  int ai = tid;
  int bi = tid + (chunk_size / 2);
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
  
  extern __shared__ int s_data[];
  s_data[ai + bankOffsetA] = data[block_offset + ai];
  s_data[bi + bankOffsetB] = data[block_offset + bi];

  int offset = 1;
  __syncthreads();


  for (int d = chunk_size >> 1; d > 0; d >>= 1) {
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      s_data[bi] += s_data[ai];
    }
    offset *= 2;
    __syncthreads();
  }

  // Down sweep

  if (tid == (chunk_size / 2) - 1) {
    if (d_block_sums != NULL) {
      d_block_sums[blockIdx.x] = s_data[bi + CONFLICT_FREE_OFFSET(bi)];
    }
    s_data[bi + CONFLICT_FREE_OFFSET(bi)] = 0;
  }

  __syncthreads();

  for (int d = 1; d < chunk_size; d *= 2) {
    offset >>= 1;
    __syncthreads();
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int t = s_data[ai];
      s_data[ai] = s_data[bi];
      s_data[bi] += t;
    }
  }
  __syncthreads();

  data[block_offset + ai] = s_data[ai + bankOffsetA];
  data[block_offset + bi] = s_data[bi + bankOffsetB];  
}

__global__ void kernApplyBlockSums(int n, int b, int *data, int *block_sums) {
  if (threadIdx.x >= n) {
    return;
  }

  if (blockIdx.x >= b) {
    return;
  }

  int tid = threadIdx.x;
  int block_offset = 2 * blockDim.x * blockIdx.x;
  int chunk_size = min(2 * blockDim.x, n - block_offset);

  int ai = tid;
  int bi = tid + (chunk_size / 2);

  data[block_offset + ai] += block_sums[blockIdx.x];
  data[block_offset + bi] += block_sums[blockIdx.x];
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
  int elemsPerBlock = threadsPerBlock * 2;
  int numBlocks = (n + elemsPerBlock - 1) / elemsPerBlock;
  
  int chunk_size = n < elemsPerBlock ? n : elemsPerBlock;
  int shared_mem_size = (chunk_size + CONFLICT_FREE_OFFSET(chunk_size - 1)) * sizeof(int);
  
  if (numBlocks > 1) {
    int *d_block_sums;
    int block_size_arr_len = numBlocks;
    if (!is_power_of_2(block_size_arr_len)) {
      block_size_arr_len = round_to_next_pow2(block_size_arr_len);
    }

    cudaMalloc((void**)&d_block_sums, block_size_arr_len * sizeof(int));
    if (block_size_arr_len > numBlocks) {
      cudaMemset(d_block_sums+numBlocks, 0, sizeof(int) * (block_size_arr_len - numBlocks));
    }
    
    kernScan<<<numBlocks, threadsPerBlock, shared_mem_size>>>(n, data_device, d_block_sums);
    parallel_scan_power2(block_size_arr_len, d_block_sums);
    kernApplyBlockSums<<<numBlocks, threadsPerBlock>>>(n, numBlocks, data_device, d_block_sums);
 
    cudaFree(d_block_sums);
  } else {
    kernScan<<<numBlocks, threadsPerBlock, shared_mem_size>>>(n, data_device, NULL);
  }
  
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
  cudaMemcpy(&last_bool, d_bools + n - 1, sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&last_index, d_indices + n - 1, sizeof(int),
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
