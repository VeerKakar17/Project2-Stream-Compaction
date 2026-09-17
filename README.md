CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

  Veer Kakar
  * [LinkedIn](www.linkedin.com/in/VeerKakar), [personal website](https://veerkakar17.github.io/PortfolioWebsite)
* Tested on: Linux Fedora 44 (Dual Boot from Windows Laptop), Intel Ultra 9 275HX, NVIDIA 5070 laptop

# Overview
The following program is a parallel CUDA implementation of the Stream Compaction algorithm. This algorithm removes all elements that don't meet a certain condition (in this case, is non-zero), and outputs a new array with just the remaining elements in the same order.

To do this, I also had to implement the scan algorithm, which computes an exclusive prefix sum.

I also created a parallel CUDA implementation of Radix sort.

For each of these, I created tests at various array sizes and compared the runtime for each at their optimal block sizes, and provided the results below.

### List of Features
- Stream Compaction and Scan on the CPU
- Stream Compaction and Scan with Thrust
- Stream Compaction with CUDA using 
- Efficient Stream Compaction and Scan with CUDA
- Efficient Stream Compaction and Scan with CUDA using shared memory and additional hardware optimizations
- Parallel Radix Sort with CUDA

## Scan

There are 4 implementations I created for the Scan algorithm. This algorithm computes an exclusive prefix sum array.

### CPU Scan
This was the baseline prefix sum algorithm done on the CPU for benchmarking.

### Naive Scan
This algorithm ran across d levels (for d as logn), where each level d adds all elements to the index `2^(d-1)` indices to the right of it. 
To prevent race conditions, this uses a buffer of size `2n` which ping-pongs within itself; the first half is considered input and the second half output for the first level, then it swaps to the second half being input and the first half as output for the second level, etc.

### Not-So-Efficient Scan
This algorithm includes the efficient-GPU scan with up-sweep to compute partial prefix-sums, and a down-sweep using those to compute the full sum.
This implementation has no hardware optimizations and uses global memory, resulting in a much slower time.

For non power-of-two array sizes, this works by allocating a buffer for the next power of 2 size up and padding the end with 0s. By returning the array without the padded elements, this will give us the same prefix sum output and allow it to work with any array size.

For array sizes greater than one block size, this splits up the array by blocks and calculates an independent scan for each one. This then creates another array for the block sums, and preforms an exclusive prefix sum on that. After computing this sum, it can run another kernel to add the corresponding sum to each element in the corresponding block to get the full prefix sums.
If the block sums array is greater than one block, then it will recursively do the same algorithm here.

### Efficient Scan
This is the same efficient-GPU Scan with the up-sweep and down-sweep and non power-of-2 and large array semantics, except with hardware optimizations. This includes shared memory optimizations with no bank conflicts, minimal kernel calls, and optimizing access order.

### Thrust Scan
This is the library GPU scan algorithm used for a baseline and benchmarking.

## Stream Compaction
This algorithm is made to remove all elements equal to `0` from an array, and return a new array with all the remaining elements. 
All implementations of this follow the same format:
- Use Map to create a boolean array, where each index corresponds to a `1` or `0` for if that corresponding element in the input array is nonzero (should be included)
- Uses Scan on the boolean array to determine size of the output array and index of each element to be included
- Use scatter to put all the elements with a `1` in the map array to the corresponding index in the output array at index specified in the scan array.

All implementations are as follows:
### CPU Without Scan
This is a baseline implementation done sequentially, optimized for the CPU. Used for benchmarking.

### CPU With Scan
This is the baseline CPU implementation following the format specified above for parallellization. This is also used for benchmarking.

### Not-So-Efficient
This is the GPU Stream Compaction done using the not-so-efficient GPU scan

### Efficient
This is the GPU Stream Compaction done using the efficient GPU scan.

### Thrust
This is the thrust library GPU Stream Compaction used for benchmarking.

## Sort
An additional feature I implemented is a parallel Radix sort.
This is benchmarked against a standard CPU sort for reference, and uses the efficient Stream compaction with Map and Scatter to preform radix sort parallelized.

# Performance Analysis

## Testing
For testing the performance of each implementation, I ran scan, sort, and compact for each implementation against arrays of size 2^8 through 2^26, with each array having a size with an exponent 2 higher (i.e. 2^8, 2^10, 2^12, etc).
I also ran each against non-power of 2 arrays, where it used the same array size but subtracted the size by the number in the exponent (i.e. 2^8 became 2^8-8).
I then used the CPU and GPU timers to track the milliseconds it took each test to run, and plotted them in the following graphs.

## Scan

![Scan Power of 2](img/Scan%20%28Power%20of%202%29.png)

| Array Size | CPU Scan (ms) | Naive Scan (ms) | Not-so-efficient Scan (ms) | Work-efficient Scan (ms) | Thrust Scan (ms) |
| ---------- | ------------: | --------------: | -------------------------: | -----------------------: | ---------------: |
| 2^8        |      0.000164 |        0.011072 |                   0.033184 |                 0.010688 |         0.019808 |
| 2^10       |      0.000721 |        0.042848 |                   0.031264 |                 0.035072 |         0.017376 |
| 2^12       |      0.002597 |        0.022496 |                   0.029408 |                  0.01904 |         0.017664 |
| 2^14       |        0.0097 |        0.019392 |                   0.027808 |                 0.014944 |         0.017696 |
| 2^16       |      0.034915 |        0.016096 |                   0.041728 |                 0.020352 |         0.019296 |
| 2^18       |      0.131919 |        0.041184 |                    0.06112 |                 0.023808 |         0.093248 |
| 2^20       |      0.522991 |        0.145184 |                   0.214496 |                 0.118752 |         0.109696 |
| 2^22       |       2.92735 |        0.352352 |                    0.55056 |                 0.206848 |         0.140256 |
| 2^24       |         9.444 |         1.16045 |                    2.16147 |                 0.863104 |         0.501824 |
| 2^26       |       39.5457 |         4.36179 |                    8.30688 |                  3.27434 |          1.70848 |

![Scan Non-Power of 2](img/Scan%20%28Non-Power%20of%202%29.png)

| Array Size | CPU Scan (ms) | Naive Scan (ms) | Not-so-efficient Scan (ms) | Work-efficient Scan (ms) | Thrust Scan (ms) |
| ---------- | ------------: | --------------: | -------------------------: | -----------------------: | ---------------: |
| 2^8-8      |      0.000403 |          0.0232 |                   0.081088 |                 0.050432 |         0.030464 |
| 2^10-10    |      0.000628 |        0.031232 |                   0.047936 |                 0.039392 |         0.022464 |
| 2^12-12    |      0.002292 |        0.023232 |                   0.046848 |                 0.034272 |         0.019424 |
| 2^14-14    |       0.01021 |        0.019392 |                   0.043968 |                 0.030688 |         0.019712 |
| 2^16-16    |      0.037827 |         0.02384 |                   0.049888 |                  0.02432 |         0.019392 |
| 2^18-18    |      0.155152 |         0.04096 |                    0.10768 |                  0.06976 |         0.092192 |
| 2^20-20    |      0.723269 |        0.143904 |                   0.297696 |                 0.202944 |         0.101024 |
| 2^22-22    |       2.28233 |        0.344288 |                   0.614592 |                 0.311584 |         0.133088 |
| 2^24-24    |       9.23744 |         1.15792 |                    2.89005 |                  1.65146 |         0.495584 |
| 2^26-26    |       46.9527 |          4.4393 |                    11.5193 |                  6.69107 |          1.71424 |

Here we observed CPU being much more efficient at smaller sizes (2^14 and lower), with all GPU implementations being more efficient as the size got higher. This is due to the increased overhead that comes up CUDA such as invoking kernels. 

There were mixed results with the implementations at lower sizes, but as the size gets really big, we observed thrust being the most efficient with CPU being the least. The not-so-efficient scan was the next work besides CPU, which is most likely due to the high cost of global memory reads.

However, unexpectedly, the naive vs work-efficient scans performed differently depending on the power of 2 vs non-power-of-2 arrays. For power of 2, the work-efficient scan is more efficient than naive, but with non-power-of-2 the naive is more efficient.
I believe this is because we start the GPU timer before checking if we have a power of 2 array, so the time to cudaMalloc a new power of 2 size array and cudaMemCpy the data over is included in the timer, and these are known to be very slow operations. However, with power of 2, we already have a device vector of a valid size so we can just run our algorithm, hence we observe that it is more efficient at large sizes. For naive, it does not need to do this as it works regardless of if the array is a power of 2 size or not.

At smaller sizes, we observe thrust generally being the fastest and our not-so-efficient scan being the slowest, with a lot of variation between naive and work-efficient, with each being slightly more efficient than the other across most sizes in their respective cases (power of 2 vs non-power of 2).

## Stream Compaction

![Stream Compaction Power of 2 Sizes](img/Stream%20Compaction%20%28Power%20of%202%20Sizes%29.png)
| Array Size | CPU Compact Without Scan (ms) | CPU Compact With Scan (ms) | Not-so-efficient Compact (ms) | Work-efficient Compact (ms) | Thrust Compact (ms) |
| ---------- | ----------------------------: | -------------------------: | ----------------------------: | --------------------------: | ------------------: |
| 2^8        |                      0.000853 |                   0.001144 |                      0.043104 |                    0.015808 |            0.027904 |
| 2^10       |                      0.003008 |                   0.004062 |                      0.040864 |                    0.030848 |             0.02576 |
| 2^12       |                       0.01365 |                   0.017179 |                       0.03904 |                    0.025472 |            0.026112 |
| 2^14       |                      0.058622 |                   0.069453 |                      0.039872 |                      0.0256 |            0.024384 |
| 2^16       |                      0.239431 |                   0.377465 |                      0.049312 |                     0.02048 |             0.02656 |
| 2^18       |                      0.846882 |                     2.0321 |                      0.147424 |                    0.104896 |            0.028384 |
| 2^20       |                       3.38621 |                    9.89456 |                      0.239328 |                    0.149216 |            0.127648 |
| 2^22       |                       13.5596 |                    34.5266 |                       0.69136 |                    0.378944 |            0.182368 |
| 2^24       |                       54.2623 |                    135.612 |                        3.5513 |                     2.31549 |            0.421504 |
| 2^26       |                       218.174 |                    647.645 |                       14.1046 |                     9.18109 |             1.54742 |

![Stream Compaction Non Power of 2 Sizes](img/Stream%20Compaction%20%28Non%20Power%20of%202%20Sizes%29.png)
| Array Size | CPU Compact Without Scan (ms) | CPU Compact With Scan (ms) | Not-so-efficient Compact (ms) | Work-efficient Compact (ms) | Thrust Compact (ms) |
| ---------- | ----------------------------: | -------------------------: | ----------------------------: | --------------------------: | ------------------: |
| 2^8-8      |                      0.000812 |                   0.002703 |                      0.071776 |                    0.038112 |            0.057056 |
| 2^10-10    |                      0.002815 |                   0.003975 |                      0.040576 |                     0.03264 |            0.028448 |
| 2^12-12    |                      0.011162 |                   0.014394 |                      0.037504 |                     0.02672 |            0.027456 |
| 2^14-14    |                      0.050091 |                   0.062422 |                       0.03904 |                    0.025376 |            0.026848 |
| 2^16-16    |                      0.212313 |                   0.250695 |                      0.047744 |                    0.019328 |            0.026528 |
| 2^18-18    |                      0.829646 |                    1.10137 |                       0.14976 |                    0.107424 |            0.029408 |
| 2^20-20    |                       3.56252 |                    5.01837 |                      0.240352 |                    0.151648 |            0.133952 |
| 2^22-22    |                        13.355 |                    35.9151 |                       0.68752 |                    0.380608 |            0.175072 |
| 2^24-24    |                        54.887 |                    135.611 |                       3.54474 |                     2.31475 |            0.429984 |
| 2^26-26    |                       212.818 |                    547.446 |                       14.0906 |                     9.18307 |             1.54579 |

Here, we observe our CPU compacts once again being more efficient at small sizes (below 2^14 elements), and being less efficient at any size larger than that. This is once again do to the CUDA algorithms being more efficient, but there being increased overhead cost due to kernel invocations.

Throughout all of these, we still observe compact with the not-so-efficient scan as the least efficient GPU algorithm due to global-memory accesses and additional kernel invocations being very slow.
At smaller sizes (below 2^18) we see a lot of variation between thrust and work-efficient being more or less efficient, but at larger array sizes, thrust is still more efficient than the work-efficient compact.

There is no notable differences between this data for non-power of 2 sizes and power of 2 sizes.

## Sort

![Sort Power of 2 Sizes](img/Sort%20%28Power%20of%202%20Sizes%29.png)
| Array Size | CPU Sort (ms) | Radix Sort GPU (ms) |
| ---------- | ------------: | ------------------: |
| 2^8        |       0.00717 |            0.452864 |
| 2^10       |      0.042982 |            0.791136 |
| 2^12       |      0.167698 |            0.833408 |
| 2^14       |      0.583486 |            0.828704 |
| 2^16       |       2.40702 |             0.82512 |
| 2^18       |       9.47884 |             2.85565 |
| 2^20       |       38.8742 |             4.17242 |
| 2^22       |       162.197 |              20.352 |
| 2^24       |        641.05 |             106.038 |
| 2^26       |       2596.26 |              418.54 |

![Sort Non Power of 2 Sizes](img/Sort%20%28Non%20Power%20of%202%20Sizes%29.png)

| Array Size | CPU Sort (ms) | Radix Sort GPU (ms) |
| ---------- | ------------: | ------------------: |
| 2^8-8      |      0.010099 |             2.07494 |
| 2^10-10    |      0.039502 |             1.25936 |
| 2^12-12    |      0.154569 |              1.3337 |
| 2^14-14    |      0.589485 |             1.32458 |
| 2^16-16    |       2.47289 |             1.26698 |
| 2^18-18    |       9.88619 |              4.3119 |
| 2^20-20    |       40.4968 |             7.21626 |
| 2^22-22    |       158.457 |             26.4758 |
| 2^24-24    |       649.645 |             139.636 |
| 2^26-26    |       2645.81 |              556.53 |

Here, the GPU Radix sort is found to be more efficient at array sizes larger than 2^16, with it being almost identical at 2^14 for power of 2 sized arrays.
This is because of the several passes the parallel radix sort needs to go through and updating several buffers for each of the 32 bits in an integer. While this causes the CUDA overhead to be increased with all of the kernel invocations and additional steps, this increased work does not scale with size, so as our array size gets bigger the benefit from parallelizing the algorithm is increased.
This results in our algorithm being much more efficient as the array size gets bigger compared to a standard CPU sort.

### Test Output

```
****************
** SCAN TESTS **
****************
    [  41  31   4  19  33  37   1   0  49  39  10  24   6 ...  17   0 ]
==== cpu scan, n=256 ====
   elapsed time: 0.000164ms    (std::chrono Measured)
    [   0  41  72  76  95 128 165 166 166 215 254 264 288 ... 6659 6676 ]
    [  41  31   4  19  33  37   1   0  49  39  10  24   6 ...   2   0 ]
==== cpu scan, n=1024 ====
   elapsed time: 0.000721ms    (std::chrono Measured)
    [   0  41  72  76  95 128 165 166 166 215 254 264 288 ... 25132 25134 ]
    [  41  31   4  19  33  37   1   0  49  39  10  24   6 ...  10   0 ]
==== cpu scan, n=4096 ====
   elapsed time: 0.002597ms    (std::chrono Measured)
    [   0  41  72  76  95 128 165 166 166 215 254 264 288 ... 100693 100703 ]
    [  41  31   4  19  33  37   1   0  49  39  10  24   6 ...  10   0 ]
==== cpu scan, n=16384 ====
   elapsed time: 0.0097ms    (std::chrono Measured)
    [   0  41  72  76  95 128 165 166 166 215 254 264 288 ... 400801 400811 ]
    [  41  31   4  19  33  37   1   0  49  39  10  24   6 ...  13   0 ]
==== cpu scan, n=65536 ====
   elapsed time: 0.034915ms    (std::chrono Measured)
    [   0  41  72  76  95 128 165 166 166 215 254 264 288 ... 1609757 1609770 ]
    [  41  31   4  19  33  37   1   0  49  39  10  24   6 ...  30   0 ]
==== cpu scan, n=262144 ====
   elapsed time: 0.131919ms    (std::chrono Measured)
    [   0  41  72  76  95 128 165 166 166 215 254 264 288 ... 6417694 6417724 ]
    [  41  31   4  19  33  37   1   0  49  39  10  24   6 ...  45   0 ]
==== cpu scan, n=1048576 ====
   elapsed time: 0.522991ms    (std::chrono Measured)
    [   0  41  72  76  95 128 165 166 166 215 254 264 288 ... 25692682 25692727 ]
    [  41  31   4  19  33  37   1   0  49  39  10  24   6 ...  35   0 ]
==== cpu scan, n=4194304 ====
   elapsed time: 2.92735ms    (std::chrono Measured)
    [   0  41  72  76  95 128 165 166 166 215 254 264 288 ... 102818467 102818502 ]
    [  41  31   4  19  33  37   1   0  49  39  10  24   6 ...  37   0 ]
==== cpu scan, n=16777216 ====
   elapsed time: 9.444ms    (std::chrono Measured)
    [   0  41  72  76  95 128 165 166 166 215 254 264 288 ... 411104516 411104553 ]
    [  41  31   4  19  33  37   1   0  49  39  10  24   6 ...  21   0 ]
==== cpu scan, n=67108864 ====
   elapsed time: 39.5457ms    (std::chrono Measured)
    [   0  41  72  76  95 128 165 166 166 215 254 264 288 ... 1644150388 1644150409 ]
    [  21   1  38  44   9  23  20   8  27  32  20  16  14 ...  16   0 ]
==== cpu scan, n=248 ====
   elapsed time: 0.000403ms    (std::chrono Measured)
    [   0  21  22  60 104 113 136 156 164 191 223 243 259 ... 5681 5697 ]
    [  21   1  38  44   9  23  20   8  27  32  20  16  14 ...  29   0 ]
==== cpu scan, n=1014 ====
   elapsed time: 0.000628ms    (std::chrono Measured)
    [   0  21  22  60 104 113 136 156 164 191 223 243 259 ... 24064 24093 ]
    [  21   1  38  44   9  23  20   8  27  32  20  16  14 ...  12   0 ]
==== cpu scan, n=4084 ====
   elapsed time: 0.002292ms    (std::chrono Measured)
    [   0  21  22  60 104 113 136 156 164 191 223 243 259 ... 98590 98602 ]
    [  21   1  38  44   9  23  20   8  27  32  20  16  14 ...  27   0 ]
==== cpu scan, n=16370 ====
   elapsed time: 0.01021ms    (std::chrono Measured)
    [   0  21  22  60 104 113 136 156 164 191 223 243 259 ... 397379 397406 ]
    [  21   1  38  44   9  23  20   8  27  32  20  16  14 ...  21   0 ]
==== cpu scan, n=65520 ====
   elapsed time: 0.037827ms    (std::chrono Measured)
    [   0  21  22  60 104 113 136 156 164 191 223 243 259 ... 1602507 1602528 ]
    [  21   1  38  44   9  23  20   8  27  32  20  16  14 ...  18   0 ]
==== cpu scan, n=262126 ====
   elapsed time: 0.155152ms    (std::chrono Measured)
    [   0  21  22  60 104 113 136 156 164 191 223 243 259 ... 6421111 6421129 ]
    [  21   1  38  44   9  23  20   8  27  32  20  16  14 ...  33   0 ]
==== cpu scan, n=1048556 ====
   elapsed time: 0.723269ms    (std::chrono Measured)
    [   0  21  22  60 104 113 136 156 164 191 223 243 259 ... 25684085 25684118 ]
    [  21   1  38  44   9  23  20   8  27  32  20  16  14 ...  43   0 ]
==== cpu scan, n=4194282 ====
   elapsed time: 2.28233ms    (std::chrono Measured)
    [   0  21  22  60 104 113 136 156 164 191 223 243 259 ... 102771839 102771882 ]
    [  21   1  38  44   9  23  20   8  27  32  20  16  14 ...  41   0 ]
==== cpu scan, n=16777192 ====
   elapsed time: 9.23744ms    (std::chrono Measured)
    [   0  21  22  60 104 113 136 156 164 191 223 243 259 ... 411105874 411105915 ]
    [  21   1  38  44   9  23  20   8  27  32  20  16  14 ...   6   0 ]
==== cpu scan, n=67108838 ====
   elapsed time: 46.9527ms    (std::chrono Measured)
    [   0  21  22  60 104 113 136 156 164 191 223 243 259 ... 1644402942 1644402948 ]
==== naive scan, n=256 ====
   elapsed time: 0.011072ms    (CUDA Measured)
    passed 
==== naive scan, n=1024 ====
   elapsed time: 0.042848ms    (CUDA Measured)
    passed 
==== naive scan, n=4096 ====
   elapsed time: 0.022496ms    (CUDA Measured)
    passed 
==== naive scan, n=16384 ====
   elapsed time: 0.019392ms    (CUDA Measured)
    passed 
==== naive scan, n=65536 ====
   elapsed time: 0.016096ms    (CUDA Measured)
    passed 
==== naive scan, n=262144 ====
   elapsed time: 0.041184ms    (CUDA Measured)
    passed 
==== naive scan, n=1048576 ====
   elapsed time: 0.145184ms    (CUDA Measured)
    passed 
==== naive scan, n=4194304 ====
   elapsed time: 0.352352ms    (CUDA Measured)
    passed 
==== naive scan, n=16777216 ====
   elapsed time: 1.16045ms    (CUDA Measured)
    passed 
==== naive scan, n=67108864 ====
   elapsed time: 4.36179ms    (CUDA Measured)
    passed 
==== naive scan, n=248 ====
   elapsed time: 0.0232ms    (CUDA Measured)
    passed 
==== naive scan, n=1014 ====
   elapsed time: 0.031232ms    (CUDA Measured)
    passed 
==== naive scan, n=4084 ====
   elapsed time: 0.023232ms    (CUDA Measured)
    passed 
==== naive scan, n=16370 ====
   elapsed time: 0.019392ms    (CUDA Measured)
    passed 
==== naive scan, n=65520 ====
   elapsed time: 0.02384ms    (CUDA Measured)
    passed 
==== naive scan, n=262126 ====
   elapsed time: 0.04096ms    (CUDA Measured)
    passed 
==== naive scan, n=1048556 ====
   elapsed time: 0.143904ms    (CUDA Measured)
    passed 
==== naive scan, n=4194282 ====
   elapsed time: 0.344288ms    (CUDA Measured)
    passed 
==== naive scan, n=16777192 ====
   elapsed time: 1.15792ms    (CUDA Measured)
    passed 
==== naive scan, n=67108838 ====
   elapsed time: 4.4393ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=256 ====
   elapsed time: 0.033184ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=1024 ====
   elapsed time: 0.031264ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=4096 ====
   elapsed time: 0.029408ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=16384 ====
   elapsed time: 0.027808ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=65536 ====
   elapsed time: 0.041728ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=262144 ====
   elapsed time: 0.06112ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=1048576 ====
   elapsed time: 0.214496ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=4194304 ====
   elapsed time: 0.55056ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=16777216 ====
   elapsed time: 2.16147ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=67108864 ====
   elapsed time: 8.30688ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=248 ====
   elapsed time: 0.081088ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=1014 ====
   elapsed time: 0.047936ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=4084 ====
   elapsed time: 0.046848ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=16370 ====
   elapsed time: 0.043968ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=65520 ====
   elapsed time: 0.049888ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=262126 ====
   elapsed time: 0.10768ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=1048556 ====
   elapsed time: 0.297696ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=4194282 ====
   elapsed time: 0.614592ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=16777192 ====
   elapsed time: 2.89005ms    (CUDA Measured)
    passed 
==== not-so-efficient scan, n=67108838 ====
   elapsed time: 11.5193ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=256 ====
   elapsed time: 0.010688ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=1024 ====
   elapsed time: 0.035072ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=4096 ====
   elapsed time: 0.01904ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=16384 ====
   elapsed time: 0.014944ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=65536 ====
   elapsed time: 0.020352ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=262144 ====
   elapsed time: 0.023808ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=1048576 ====
   elapsed time: 0.118752ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=4194304 ====
   elapsed time: 0.206848ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=16777216 ====
   elapsed time: 0.863104ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=67108864 ====
   elapsed time: 3.27434ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=248 ====
   elapsed time: 0.050432ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=1014 ====
   elapsed time: 0.039392ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=4084 ====
   elapsed time: 0.034272ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=16370 ====
   elapsed time: 0.030688ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=65520 ====
   elapsed time: 0.02432ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=262126 ====
   elapsed time: 0.06976ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=1048556 ====
   elapsed time: 0.202944ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=4194282 ====
   elapsed time: 0.311584ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=16777192 ====
   elapsed time: 1.65146ms    (CUDA Measured)
    passed 
==== work-efficient scan, n=67108838 ====
   elapsed time: 6.69107ms    (CUDA Measured)
    passed 
==== thrust scan, n=256 ====
   elapsed time: 0.019808ms    (CUDA Measured)
    passed 
==== thrust scan, n=1024 ====
   elapsed time: 0.017376ms    (CUDA Measured)
    passed 
==== thrust scan, n=4096 ====
   elapsed time: 0.017664ms    (CUDA Measured)
    passed 
==== thrust scan, n=16384 ====
   elapsed time: 0.017696ms    (CUDA Measured)
    passed 
==== thrust scan, n=65536 ====
   elapsed time: 0.019296ms    (CUDA Measured)
    passed 
==== thrust scan, n=262144 ====
   elapsed time: 0.093248ms    (CUDA Measured)
    passed 
==== thrust scan, n=1048576 ====
   elapsed time: 0.109696ms    (CUDA Measured)
    passed 
==== thrust scan, n=4194304 ====
   elapsed time: 0.140256ms    (CUDA Measured)
    passed 
==== thrust scan, n=16777216 ====
   elapsed time: 0.501824ms    (CUDA Measured)
    passed 
==== thrust scan, n=67108864 ====
   elapsed time: 1.70848ms    (CUDA Measured)
    passed 
==== thrust scan, n=248 ====
   elapsed time: 0.030464ms    (CUDA Measured)
    passed 
==== thrust scan, n=1014 ====
   elapsed time: 0.022464ms    (CUDA Measured)
    passed 
==== thrust scan, n=4084 ====
   elapsed time: 0.019424ms    (CUDA Measured)
    passed 
==== thrust scan, n=16370 ====
   elapsed time: 0.019712ms    (CUDA Measured)
    passed 
==== thrust scan, n=65520 ====
   elapsed time: 0.019392ms    (CUDA Measured)
    passed 
==== thrust scan, n=262126 ====
   elapsed time: 0.092192ms    (CUDA Measured)
    passed 
==== thrust scan, n=1048556 ====
   elapsed time: 0.101024ms    (CUDA Measured)
    passed 
==== thrust scan, n=4194282 ====
   elapsed time: 0.133088ms    (CUDA Measured)
    passed 
==== thrust scan, n=16777192 ====
   elapsed time: 0.495584ms    (CUDA Measured)
    passed 
==== thrust scan, n=67108838 ====
   elapsed time: 1.71424ms    (CUDA Measured)
    passed 

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   0   2   3   2   3   3   1   3   1   3   1   1 ...   2   0 ]
==== cpu compact without scan, n=256 ====
   elapsed time: 0.000853ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   1   2 ]
    passed 
==== cpu compact with scan, n=256 ====
   elapsed time: 0.001144ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   1   2 ]
    passed 
    [   1   0   2   3   2   3   3   1   3   1   3   1   1 ...   0   0 ]
==== cpu compact without scan, n=1024 ====
   elapsed time: 0.003008ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   3   2 ]
    passed 
==== cpu compact with scan, n=1024 ====
   elapsed time: 0.004062ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   3   2 ]
    passed 
    [   1   0   2   3   2   3   3   1   3   1   3   1   1 ...   0   0 ]
==== cpu compact without scan, n=4096 ====
   elapsed time: 0.01365ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   3   2 ]
    passed 
==== cpu compact with scan, n=4096 ====
   elapsed time: 0.017179ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   3   2 ]
    passed 
    [   1   0   2   3   2   3   3   1   3   1   3   1   1 ...   1   0 ]
==== cpu compact without scan, n=16384 ====
   elapsed time: 0.058622ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   1   1 ]
    passed 
==== cpu compact with scan, n=16384 ====
   elapsed time: 0.069453ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   1   1 ]
    passed 
    [   1   0   2   3   2   3   3   1   3   1   3   1   1 ...   0   0 ]
==== cpu compact without scan, n=65536 ====
   elapsed time: 0.239431ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   1   3 ]
    passed 
==== cpu compact with scan, n=65536 ====
   elapsed time: 0.377465ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   1   3 ]
    passed 
    [   1   0   2   3   2   3   3   1   3   1   3   1   1 ...   3   0 ]
==== cpu compact without scan, n=262144 ====
   elapsed time: 0.846882ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   2   3 ]
    passed 
==== cpu compact with scan, n=262144 ====
   elapsed time: 2.0321ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   2   3 ]
    passed 
    [   1   0   2   3   2   3   3   1   3   1   3   1   1 ...   3   0 ]
==== cpu compact without scan, n=1048576 ====
   elapsed time: 3.38621ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   1   3 ]
    passed 
==== cpu compact with scan, n=1048576 ====
   elapsed time: 9.89456ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   1   3 ]
    passed 
    [   1   0   2   3   2   3   3   1   3   1   3   1   1 ...   2   0 ]
==== cpu compact without scan, n=4194304 ====
   elapsed time: 13.5596ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   3   2 ]
    passed 
==== cpu compact with scan, n=4194304 ====
   elapsed time: 34.5266ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   3   2 ]
    passed 
    [   1   0   2   3   2   3   3   1   3   1   3   1   1 ...   3   0 ]
==== cpu compact without scan, n=16777216 ====
   elapsed time: 54.2623ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   2   3 ]
    passed 
==== cpu compact with scan, n=16777216 ====
   elapsed time: 135.612ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   2   3 ]
    passed 
    [   1   0   2   3   2   3   3   1   3   1   3   1   1 ...   1   0 ]
==== cpu compact without scan, n=67108864 ====
   elapsed time: 218.174ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   2   1 ]
    passed 
==== cpu compact with scan, n=67108864 ====
   elapsed time: 647.645ms    (std::chrono Measured)
    [   1   2   3   2   3   3   1   3   1   3   1   1   1 ...   2   1 ]
    passed 
    [   1   1   3   1   0   3   0   3   2   3   2   0   1 ...   3   0 ]
==== cpu compact without scan, n=248 ====
   elapsed time: 0.000812ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   3   3 ]
    passed 
==== cpu compact with scan, n=248 ====
   elapsed time: 0.002703ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   3   3 ]
    passed 
    [   1   1   3   1   0   3   0   3   2   3   2   0   1 ...   3   0 ]
==== cpu compact without scan, n=1014 ====
   elapsed time: 0.002815ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   1   3 ]
    passed 
==== cpu compact with scan, n=1014 ====
   elapsed time: 0.003975ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   1   3 ]
    passed 
    [   1   1   3   1   0   3   0   3   2   3   2   0   1 ...   0   0 ]
==== cpu compact without scan, n=4084 ====
   elapsed time: 0.011162ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   2   2 ]
    passed 
==== cpu compact with scan, n=4084 ====
   elapsed time: 0.014394ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   2   2 ]
    passed 
    [   1   1   3   1   0   3   0   3   2   3   2   0   1 ...   0   0 ]
==== cpu compact without scan, n=16370 ====
   elapsed time: 0.050091ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   2   2 ]
    passed 
==== cpu compact with scan, n=16370 ====
   elapsed time: 0.062422ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   2   2 ]
    passed 
    [   1   1   3   1   0   3   0   3   2   3   2   0   1 ...   1   0 ]
==== cpu compact without scan, n=65520 ====
   elapsed time: 0.212313ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   3   1 ]
    passed 
==== cpu compact with scan, n=65520 ====
   elapsed time: 0.250695ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   3   1 ]
    passed 
    [   1   1   3   1   0   3   0   3   2   3   2   0   1 ...   2   0 ]
==== cpu compact without scan, n=262126 ====
   elapsed time: 0.829646ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   1   2 ]
    passed 
==== cpu compact with scan, n=262126 ====
   elapsed time: 1.10137ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   1   2 ]
    passed 
    [   1   1   3   1   0   3   0   3   2   3   2   0   1 ...   1   0 ]
==== cpu compact without scan, n=1048556 ====
   elapsed time: 3.56252ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   2   1 ]
    passed 
==== cpu compact with scan, n=1048556 ====
   elapsed time: 5.01837ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   2   1 ]
    passed 
    [   1   1   3   1   0   3   0   3   2   3   2   0   1 ...   3   0 ]
==== cpu compact without scan, n=4194282 ====
   elapsed time: 13.355ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   3   3 ]
    passed 
==== cpu compact with scan, n=4194282 ====
   elapsed time: 35.9151ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   3   3 ]
    passed 
    [   1   1   3   1   0   3   0   3   2   3   2   0   1 ...   3   0 ]
==== cpu compact without scan, n=16777192 ====
   elapsed time: 54.887ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   3   3 ]
    passed 
==== cpu compact with scan, n=16777192 ====
   elapsed time: 135.611ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   3   3 ]
    passed 
    [   1   1   3   1   0   3   0   3   2   3   2   0   1 ...   1   0 ]
==== cpu compact without scan, n=67108838 ====
   elapsed time: 212.818ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   1   1 ]
    passed 
==== cpu compact with scan, n=67108838 ====
   elapsed time: 547.446ms    (std::chrono Measured)
    [   1   1   3   1   3   3   2   3   2   1   3   3   2 ...   1   1 ]
    passed 
==== not-so-efficient compact, n=256 ====
   elapsed time: 0.043104ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=1024 ====
   elapsed time: 0.040864ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=4096 ====
   elapsed time: 0.03904ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=16384 ====
   elapsed time: 0.039872ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=65536 ====
   elapsed time: 0.049312ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=262144 ====
   elapsed time: 0.147424ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=1048576 ====
   elapsed time: 0.239328ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=4194304 ====
   elapsed time: 0.69136ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=16777216 ====
   elapsed time: 3.5513ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=67108864 ====
   elapsed time: 14.1046ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=248 ====
   elapsed time: 0.071776ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=1014 ====
   elapsed time: 0.040576ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=4084 ====
   elapsed time: 0.037504ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=16370 ====
   elapsed time: 0.03904ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=65520 ====
   elapsed time: 0.047744ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=262126 ====
   elapsed time: 0.14976ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=1048556 ====
   elapsed time: 0.240352ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=4194282 ====
   elapsed time: 0.68752ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=16777192 ====
   elapsed time: 3.54474ms    (CUDA Measured)
    passed 
==== not-so-efficient compact, n=67108838 ====
   elapsed time: 14.0906ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=256 ====
   elapsed time: 0.015808ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=1024 ====
   elapsed time: 0.030848ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=4096 ====
   elapsed time: 0.025472ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=16384 ====
   elapsed time: 0.0256ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=65536 ====
   elapsed time: 0.02048ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=262144 ====
   elapsed time: 0.104896ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=1048576 ====
   elapsed time: 0.149216ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=4194304 ====
   elapsed time: 0.378944ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=16777216 ====
   elapsed time: 2.31549ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=67108864 ====
   elapsed time: 9.18109ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=248 ====
   elapsed time: 0.038112ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=1014 ====
   elapsed time: 0.03264ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=4084 ====
   elapsed time: 0.02672ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=16370 ====
   elapsed time: 0.025376ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=65520 ====
   elapsed time: 0.019328ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=262126 ====
   elapsed time: 0.107424ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=1048556 ====
   elapsed time: 0.151648ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=4194282 ====
   elapsed time: 0.380608ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=16777192 ====
   elapsed time: 2.31475ms    (CUDA Measured)
    passed 
==== work-efficient compact, n=67108838 ====
   elapsed time: 9.18307ms    (CUDA Measured)
    passed 
==== thrust compact, n=256 ====
   elapsed time: 0.027904ms    (CUDA Measured)
    passed 
==== thrust compact, n=1024 ====
   elapsed time: 0.02576ms    (CUDA Measured)
    passed 
==== thrust compact, n=4096 ====
   elapsed time: 0.026112ms    (CUDA Measured)
    passed 
==== thrust compact, n=16384 ====
   elapsed time: 0.024384ms    (CUDA Measured)
    passed 
==== thrust compact, n=65536 ====
   elapsed time: 0.02656ms    (CUDA Measured)
    passed 
==== thrust compact, n=262144 ====
   elapsed time: 0.028384ms    (CUDA Measured)
    passed 
==== thrust compact, n=1048576 ====
   elapsed time: 0.127648ms    (CUDA Measured)
    passed 
==== thrust compact, n=4194304 ====
   elapsed time: 0.182368ms    (CUDA Measured)
    passed 
==== thrust compact, n=16777216 ====
   elapsed time: 0.421504ms    (CUDA Measured)
    passed 
==== thrust compact, n=67108864 ====
   elapsed time: 1.54742ms    (CUDA Measured)
    passed 
==== thrust compact, n=248 ====
   elapsed time: 0.057056ms    (CUDA Measured)
    passed 
==== thrust compact, n=1014 ====
   elapsed time: 0.028448ms    (CUDA Measured)
    passed 
==== thrust compact, n=4084 ====
   elapsed time: 0.027456ms    (CUDA Measured)
    passed 
==== thrust compact, n=16370 ====
   elapsed time: 0.026848ms    (CUDA Measured)
    passed 
==== thrust compact, n=65520 ====
   elapsed time: 0.026528ms    (CUDA Measured)
    passed 
==== thrust compact, n=262126 ====
   elapsed time: 0.029408ms    (CUDA Measured)
    passed 
==== thrust compact, n=1048556 ====
   elapsed time: 0.133952ms    (CUDA Measured)
    passed 
==== thrust compact, n=4194282 ====
   elapsed time: 0.175072ms    (CUDA Measured)
    passed 
==== thrust compact, n=16777192 ====
   elapsed time: 0.429984ms    (CUDA Measured)
    passed 
==== thrust compact, n=67108838 ====
   elapsed time: 1.54579ms    (CUDA Measured)
    passed 

**********************
** RADIX SORT TESTS **
**********************
    [   9  41  14   9  20  43  30  16  49  31  22   6  34 ...  39  13 ]
==== cpu sort, n=256 ====
   elapsed time: 0.00717ms    (std::chrono Measured)
    [   0   0   0   0   0   1   2   2   2   2   3   3   3 ...  49  49 ]
    [   9  41  14   9  20  43  30  16  49  31  22   6  34 ...  27   6 ]
==== cpu sort, n=1024 ====
   elapsed time: 0.042982ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [   9  41  14   9  20  43  30  16  49  31  22   6  34 ...   4   8 ]
==== cpu sort, n=4096 ====
   elapsed time: 0.167698ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [   9  41  14   9  20  43  30  16  49  31  22   6  34 ...  10  14 ]
==== cpu sort, n=16384 ====
   elapsed time: 0.583486ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [   9  41  14   9  20  43  30  16  49  31  22   6  34 ...   3  11 ]
==== cpu sort, n=65536 ====
   elapsed time: 2.40702ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [   9  41  14   9  20  43  30  16  49  31  22   6  34 ...   6   8 ]
==== cpu sort, n=262144 ====
   elapsed time: 9.47884ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [   9  41  14   9  20  43  30  16  49  31  22   6  34 ...  40  49 ]
==== cpu sort, n=1048576 ====
   elapsed time: 38.8742ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [   9  41  14   9  20  43  30  16  49  31  22   6  34 ...  39  19 ]
==== cpu sort, n=4194304 ====
   elapsed time: 162.197ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [   9  41  14   9  20  43  30  16  49  31  22   6  34 ...  10  38 ]
==== cpu sort, n=16777216 ====
   elapsed time: 641.05ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [  17  10  20  29  41  37  26   0  45  46  16   5  36 ...  38  13 ]
==== cpu sort, n=67108864 ====
   elapsed time: 2596.26ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [  31  28  36  25  41  16  36  16  42  46  21  26   3 ...  11  31 ]
==== cpu sort, n=248 ====
   elapsed time: 0.010099ms    (std::chrono Measured)
    [   0   0   0   1   1   1   2   2   2   2   2   2   2 ...  49  49 ]
    [  31  28  36  25  41  16  36  16  42  46  21  26   3 ...   2  47 ]
==== cpu sort, n=1014 ====
   elapsed time: 0.039502ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [  31  28  36  25  41  16  36  16  42  46  21  26   3 ...   8  15 ]
==== cpu sort, n=4084 ====
   elapsed time: 0.154569ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [  31  28  36  25  41  16  36  16  42  46  21  26   3 ...  42  16 ]
==== cpu sort, n=16370 ====
   elapsed time: 0.589485ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [  31  28  36  25  41  16  36  16  42  46  21  26   3 ...  43  28 ]
==== cpu sort, n=65520 ====
   elapsed time: 2.47289ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [  31  28  36  25  41  16  36  16  42  46  21  26   3 ...  32  23 ]
==== cpu sort, n=262126 ====
   elapsed time: 9.88619ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [  31  28  36  25  41  16  36  16  42  46  21  26   3 ...  41  29 ]
==== cpu sort, n=1048556 ====
   elapsed time: 40.4968ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [  31  28  36  25  41  16  36  16  42  46  21  26   3 ...  22  15 ]
==== cpu sort, n=4194282 ====
   elapsed time: 158.457ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [  31  28  36  25  41  16  36  16  42  46  21  26   3 ...  35  18 ]
==== cpu sort, n=16777192 ====
   elapsed time: 649.645ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    [  46  31   4  30  30  43   1   3  10  38  40   5  20 ...   7  29 ]
==== cpu sort, n=67108838 ====
   elapsed time: 2645.81ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
==== radix sort, n=256 ====
   elapsed time: 0.452864ms    (CUDA Measured)
    passed 
==== radix sort, n=1024 ====
   elapsed time: 0.791136ms    (CUDA Measured)
    passed 
==== radix sort, n=4096 ====
   elapsed time: 0.833408ms    (CUDA Measured)
    passed 
==== radix sort, n=16384 ====
   elapsed time: 0.828704ms    (CUDA Measured)
    passed 
==== radix sort, n=65536 ====
   elapsed time: 0.82512ms    (CUDA Measured)
    passed 
==== radix sort, n=262144 ====
   elapsed time: 2.85565ms    (CUDA Measured)
    passed 
==== radix sort, n=1048576 ====
   elapsed time: 4.17242ms    (CUDA Measured)
    passed 
==== radix sort, n=4194304 ====
   elapsed time: 20.352ms    (CUDA Measured)
    passed 
==== radix sort, n=16777216 ====
   elapsed time: 106.038ms    (CUDA Measured)
    passed 
==== radix sort, n=67108864 ====
   elapsed time: 418.54ms    (CUDA Measured)
    passed 
==== radix sort, n=248 ====
   elapsed time: 2.07494ms    (CUDA Measured)
    passed 
==== radix sort, n=1014 ====
   elapsed time: 1.25936ms    (CUDA Measured)
    passed 
==== radix sort, n=4084 ====
   elapsed time: 1.3337ms    (CUDA Measured)
    passed 
==== radix sort, n=16370 ====
   elapsed time: 1.32458ms    (CUDA Measured)
    passed 
==== radix sort, n=65520 ====
   elapsed time: 1.26698ms    (CUDA Measured)
    passed 
==== radix sort, n=262126 ====
   elapsed time: 4.3119ms    (CUDA Measured)
    passed 
==== radix sort, n=1048556 ====
   elapsed time: 7.21626ms    (CUDA Measured)
    passed 
==== radix sort, n=4194282 ====
   elapsed time: 26.4758ms    (CUDA Measured)
    passed 
==== radix sort, n=16777192 ====
   elapsed time: 139.636ms    (CUDA Measured)
    passed 
==== radix sort, n=67108838 ====
   elapsed time: 556.53ms    (CUDA Measured)
    passed 
```