#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void do_scan(int n, int *odata, const int *idata) {
            int prev = 0;
            for (int i = 0; i < n; i++) {
                odata[i] = prev;
                prev += idata[i];
            }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            do_scan(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int num_out = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[num_out] = idata[i];
                    num_out++;
                }
            }
            timer().endCpuTimer();
            return num_out;
        }

        static void check_to_include(int n, int *odata, const int *idata) {
            for (int i = 0; i < n; i++) {
                odata[i] = idata[i] != 0;
            }
        }

        static void scatter(int n, int *odata, const int *idata, const int *indexes, const int *to_incl_arr) {
            for (int i = 0; i < n; i++) {
                if (to_incl_arr[i] == 0) {
                    continue;
                }

                odata[indexes[i]] = idata[i];
            }
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int *to_incl_arr = (int*)malloc(sizeof(int) * n);
            check_to_include(n, to_incl_arr, idata);
            int *buf = (int*)malloc(sizeof(int) * n);
            do_scan(n, buf, to_incl_arr);
            int size_out = buf[n-1] + to_incl_arr[n-1];
            scatter(n, odata, idata, buf, to_incl_arr);
            free(buf);
            free(to_incl_arr);
            timer().endCpuTimer();
            return size_out;
        }
    }
}
