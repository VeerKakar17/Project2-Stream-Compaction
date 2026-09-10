#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);
        
        void parallel_scan_power2(int n, int *data_device);
        void parallel_scan(int n, int *data_device);
        int compact(int n, int *odata, const int *idata);
    }
}
