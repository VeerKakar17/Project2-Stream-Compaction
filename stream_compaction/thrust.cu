#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        struct IsZero {
            __host__ __device__ bool operator()(int x) const {
                return x == 0;
            }
        };

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            thrust::device_vector<int> d_idata(idata, idata + n);
            thrust::device_vector<int> d_odata(n);
            
            timer().startGpuTimer();

            thrust::exclusive_scan(d_idata.begin(), d_idata.end(), d_odata.begin());

            timer().endGpuTimer();

            thrust::copy(d_odata.begin(), d_odata.end(), odata);
        }

        int compact(int n, int *odata, const int *idata) {
            thrust::device_vector<int> d_data(idata, idata + n);

            timer().startGpuTimer();

            thrust::device_vector<int>::iterator end =
                thrust::remove_if(d_data.begin(), d_data.end(), IsZero{});

            timer().endGpuTimer();

            int count = static_cast<int>(end - d_data.begin());
            thrust::copy(d_data.begin(), end, odata);
            return count;
        }
    }
}
