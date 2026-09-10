#include "common.h"
#include "efficient.h"
#include "radix_sort.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace StreamCompaction {
namespace Radix {
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
  static PerformanceTimer timer;
  return timer;
}

__global__ void kernCalculateTArr(int n, int *d_f, int *d_t, int *d_b) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid >= n) {
    return;
  }

  int total_falses = (1 - d_b[n - 1]) + d_f[n - 1];

  d_t[tid] = tid - d_f[tid] + total_falses;
}

__global__ void kernCalculateDArr(int n, int *d_d, int *d_b, int *d_t,
                                  int *d_f) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid >= n) {
    return;
  }

  if (d_b[tid]) {
    d_d[tid] = d_t[tid];
  } else {
    d_d[tid] = d_f[tid];
  }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void sort(int n, int *odata, const int *idata) {
  int *d_odata;
  int *d_idata;
  int *d_b;
  int *d_e;
  int *d_t;
  int *d_d;

  cudaMalloc((void **)&d_odata, sizeof(int) * n);
  cudaMalloc((void **)&d_idata, sizeof(int) * n);
  cudaMalloc((void **)&d_b, sizeof(int) * n);
  cudaMalloc((void **)&d_e, sizeof(int) * n);
  cudaMalloc((void **)&d_t, sizeof(int) * n);
  cudaMalloc((void **)&d_d, sizeof(int) * n);
  cudaMemcpy(d_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

  timer().startGpuTimer();

  int threadsPerBlock = 1024;
  int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
  for (int i = 0; i < 32; i++) {
    kernMapToBoolean<<<numBlocks, threadsPerBlock>>>(
        n, d_b, d_idata, StreamCompaction::Common::BitIsOne{i});
    kernMapToBoolean<<<numBlocks, threadsPerBlock>>>(
        n, d_e, d_b, StreamCompaction::Common::IsZero{});
    StreamCompaction::Efficient::parallel_scan(n, d_e);

    kernCalculateTArr<<<numBlocks, threadsPerBlock>>>(n, d_e, d_t, d_b);
    kernCalculateDArr<<<numBlocks, threadsPerBlock>>>(n, d_d, d_b, d_t, d_e);

    StreamCompaction::Common::
        kernScatterAllElems<<<numBlocks, threadsPerBlock>>>(n, d_odata, d_idata,
                                                            d_d);

    int *temp = d_odata;
    d_odata = d_idata;
    d_idata = temp;
  }

  timer().endGpuTimer();

  cudaMemcpy(odata, d_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
  cudaFree(d_odata);
  cudaFree(d_idata);
  cudaFree(d_b);
  cudaFree(d_e);
  cudaFree(d_t);
  cudaFree(d_d);
}
} // namespace Radix
} // namespace StreamCompaction
