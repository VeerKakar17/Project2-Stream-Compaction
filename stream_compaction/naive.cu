#include "common.h"
#include "naive.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace StreamCompaction {
namespace Naive {
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
  static PerformanceTimer timer;
  return timer;
}

const int threadsPerBlock = 256;

__global__ void kernNaiveScan(int n, const int *idata, int *temp_buf, int *d_block_sums) {
  int block_offset = blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  int id = tid + block_offset;
  int chunk_size = min(blockDim.x, n - block_offset);

  int pout = 0;
  int pin = 1;
  if (tid < chunk_size) {
    temp_buf[pout * n + id] = (tid > 0) ? idata[id - 1] : 0;
  }
  __syncthreads();

  for (int offset = 1; offset < chunk_size; offset *= 2) {
    pout = 1 - pout;
    pin = 1 - pin;
    if (tid < chunk_size) {
      if (tid >= offset) {
        temp_buf[pout * n + id] =
            temp_buf[pin * n + id] + temp_buf[pin * n + id - offset];
      } else {
        temp_buf[pout * n + id] = temp_buf[pin * n + id];
      }
    }
    __syncthreads();
  }

  if (tid < chunk_size && pout == 1) {
    temp_buf[pin * n + id] = temp_buf[pout * n + id];
  }

  if (d_block_sums != NULL && tid == chunk_size - 1) {
    d_block_sums[blockIdx.x] = temp_buf[id] + idata[id];
  }
}

__global__ void kernApplyBlockSums(int n, int b, int *data, int *block_sums) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id >= n) {
    return;
  }

  if (blockIdx.x >= b) {
    return;
  }

  data[id] += block_sums[blockIdx.x];
}

void parallel_scan(int n, const int *d_idata, int *d_odata) {
  int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
  int *d_block_sums = NULL;

  if (numBlocks > 1) {
    cudaMalloc((void**)&d_block_sums, sizeof(int) * numBlocks);
  }

  kernNaiveScan<<<numBlocks, threadsPerBlock>>>(n, d_idata, d_odata, d_block_sums);
  
  if (numBlocks > 1) {
    int *d_scanned_block_sums;
    cudaMalloc((void**)&d_scanned_block_sums, sizeof(int) * 2 * numBlocks);

    parallel_scan(numBlocks, d_block_sums, d_scanned_block_sums);
    kernApplyBlockSums<<<numBlocks, threadsPerBlock>>>(n, numBlocks, d_odata, d_scanned_block_sums);

    cudaFree(d_scanned_block_sums);
    cudaFree(d_block_sums);
  }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  if (n == 0) {
    return;
  }

  int *temp_buf;
  int *idata_cuda;
  cudaMalloc((void **)&temp_buf, sizeof(int) * 2 * n);
  cudaMalloc((void **)&idata_cuda, sizeof(int) * n);
  cudaMemcpy(idata_cuda, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
  timer().startGpuTimer();
  
  parallel_scan(n, idata_cuda, temp_buf);
  
  timer().endGpuTimer();
  cudaMemcpy(odata, temp_buf, sizeof(int) * n, cudaMemcpyDeviceToHost);
  cudaFree(temp_buf);
  cudaFree(idata_cuda);
}
} // namespace Naive
} // namespace StreamCompaction
