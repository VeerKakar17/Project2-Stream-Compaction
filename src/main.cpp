/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix_sort.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 8; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
const int LARGE_SIZE = 1 << 20; // Larger than one CUDA block
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];
int *largeA = new int[LARGE_SIZE];
int *largeB = new int[LARGE_SIZE];
int *largeC = new int[LARGE_SIZE];

int main(int argc, char* argv[]) {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); */

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    genArray(LARGE_SIZE - 1, largeA, 50);
    largeA[LARGE_SIZE - 1] = 0;
    printArray(LARGE_SIZE, largeA, true);

    zeroArray(LARGE_SIZE, largeB);
    printDesc("cpu scan, large");
    StreamCompaction::CPU::scan(LARGE_SIZE, largeB, largeA);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(LARGE_SIZE, largeB, true);

    zeroArray(LARGE_SIZE, largeC);
    printDesc("work-efficient scan, large");
    StreamCompaction::Efficient::scan(LARGE_SIZE, largeC, largeA);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(LARGE_SIZE, largeC, true);
    printCmpResult(LARGE_SIZE, largeB, largeC);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    genArray(LARGE_SIZE - 1, largeA, 4);
    largeA[LARGE_SIZE - 1] = 0;
    printArray(LARGE_SIZE, largeA, true);

    zeroArray(LARGE_SIZE, largeB);
    printDesc("cpu compact without scan, large");
    count = StreamCompaction::CPU::compactWithoutScan(LARGE_SIZE, largeB, largeA);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, largeB, true);
    printCmpLenResult(count, expectedCount, largeB, largeB);

    zeroArray(LARGE_SIZE, largeC);
    printDesc("work-efficient compact, large");
    count = StreamCompaction::Efficient::compact(LARGE_SIZE, largeC, largeA);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, largeC, true);
    printCmpLenResult(count, expectedCount, largeB, largeC);

    printf("\n");
    printf("**********************\n");
    printf("** RADIX SORT TESTS **\n");
    printf("**********************\n");

    // Radix sort tests

    genArray(SIZE, a, 50);
    printArray(SIZE, a, true);

    zeroArray(SIZE, b);
    printDesc("cpu sort, power-of-two");
    StreamCompaction::CPU::sort(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("radix sort, power-of-two");
    StreamCompaction::Radix::sort(SIZE, c, a);
    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, b);
    printDesc("cpu sort, non-power-of-two");
    StreamCompaction::CPU::sort(NPOT, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, b, true);

    zeroArray(SIZE, c);
    printDesc("radix sort, non-power-of-two");
    StreamCompaction::Radix::sort(NPOT, c, a);
    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    genArray(LARGE_SIZE, largeA, 50);
    printArray(LARGE_SIZE, largeA, true);

    zeroArray(LARGE_SIZE, largeB);
    printDesc("cpu sort, large");
    StreamCompaction::CPU::sort(LARGE_SIZE, largeB, largeA);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(LARGE_SIZE, largeB, true);

    zeroArray(LARGE_SIZE, largeC);
    printDesc("radix sort, large");
    StreamCompaction::Radix::sort(LARGE_SIZE, largeC, largeA);
    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(LARGE_SIZE, largeC, true);
    printCmpResult(LARGE_SIZE, largeB, largeC);

    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] largeA;
    delete[] largeB;
    delete[] largeC;
}
