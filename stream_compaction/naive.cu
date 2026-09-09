#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScan(int n, const int *idata, int *temp_buf) {
            int id = threadIdx.x + blockIdx.x * blockDim.x;
            if (id >= n) {
                return;
            }

            int pout = 0;
            int pin = 1;
            temp_buf[pout * n + id] = (id > 0) ? idata[id - 1] : 0;
            __syncthreads();

            for (int offset = 1; offset < n; offset *= 2) {
                pout = 1 - pout;
                pin = 1 - pin;
                if (id >= offset) {
                    temp_buf[pout * n + id] = temp_buf[pin*n+id] + temp_buf[pin * n + id - offset];
                } else {
                    temp_buf[pout * n + id] = temp_buf[pin*n + id];
                }
                __syncthreads();
            }

            if (pout == 1) {
                temp_buf[pin*n+id] = temp_buf[pout * n + id];
            } 
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            int threadsPerBlock = 1024;
            int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

            int *temp_buf;
            int *idata_cuda;
            cudaMalloc((void**) &temp_buf, sizeof(int) * 2 * n);
            cudaMalloc((void**)&idata_cuda, sizeof(int) * n);
            cudaMemcpy(idata_cuda, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            kernNaiveScan<<<numBlocks, threadsPerBlock>>>(n, idata_cuda, temp_buf);
            cudaMemcpy(odata, temp_buf, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(temp_buf);
            cudaFree(idata_cuda);
            timer().endGpuTimer();
        }
    }
}
