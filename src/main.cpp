/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/not_so_efficient.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix_sort.h>
#include "testing_helpers.hpp"

void printSizedDesc(const char *desc, int n) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s, n=%d", desc, n);
    printDesc(buf);
}

std::vector<std::vector<int>> makeTestArrays(const std::vector<int> &sizes, int maxval, bool zeroLast) {
    std::vector<std::vector<int>> arrays;
    arrays.reserve(sizes.size());

    for (int n : sizes) {
        arrays.emplace_back(n);
        genArray(n, arrays.back().data(), maxval);

        if (zeroLast && n > 0) {
            arrays.back()[n - 1] = 0;
        }
    }

    return arrays;
}

void testScan(const std::vector<int> &sizes) {
    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    if (sizes.empty()) {
        return;
    }

    std::vector<std::vector<int>> inputs = makeTestArrays(sizes, 50, true);
    std::vector<std::vector<int>> expected(sizes.size());
    {
        int n = sizes[0];
        std::vector<int> result(n);
        StreamCompaction::CPU::scan(n, result.data(), inputs[0].data());
    }

    for (size_t i = 0; i < sizes.size(); i++) {
        int n = sizes[i];
        expected[i].resize(n);

        printArray(n, inputs[i].data(), true);

        zeroArray(n, expected[i].data());
        printSizedDesc("cpu scan", n);
        StreamCompaction::CPU::scan(n, expected[i].data(), inputs[i].data());
        printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
        printArray(n, expected[i].data(), true);
    }
    {
        int n = sizes[0];
        std::vector<int> result(n);
        StreamCompaction::Naive::scan(n, result.data(), inputs[0].data());
    }
    for (size_t i = 0; i < sizes.size(); i++) {
        int n = sizes[i];
        std::vector<int> result(n);

        zeroArray(n, result.data());
        printSizedDesc("naive scan", n);
        StreamCompaction::Naive::scan(n, result.data(), inputs[i].data());
        printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        //printArray(n, result.data(), true);
        printCmpResult(n, expected[i].data(), result.data());
    }

    {
        int n = sizes[0];
        std::vector<int> result(n);
        StreamCompaction::NotSoEfficient::scan(n, result.data(), inputs[0].data());
    }

    for (size_t i = 0; i < sizes.size(); i++) {
        int n = sizes[i];
        std::vector<int> result(n);

        zeroArray(n, result.data());
        printSizedDesc("not-so-efficient scan", n);
        StreamCompaction::NotSoEfficient::scan(n, result.data(), inputs[i].data());
        printElapsedTime(StreamCompaction::NotSoEfficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        //printArray(n, result.data(), true);
        printCmpResult(n, expected[i].data(), result.data());
    }

    {
        int n = sizes[0];
        std::vector<int> result(n);
        StreamCompaction::Efficient::scan(n, result.data(), inputs[0].data());
    }

    for (size_t i = 0; i < sizes.size(); i++) {
        int n = sizes[i];
        std::vector<int> result(n);

        zeroArray(n, result.data());
        printSizedDesc("work-efficient scan", n);
        StreamCompaction::Efficient::scan(n, result.data(), inputs[i].data());
        printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        //printArray(n, result.data(), true);
        printCmpResult(n, expected[i].data(), result.data());
    }

    {
        int n = sizes[0];
        std::vector<int> result(n);
        StreamCompaction::Thrust::scan(n, result.data(), inputs[0].data());
    }

    for (size_t i = 0; i < sizes.size(); i++) {
        int n = sizes[i];
        std::vector<int> result(n);

        zeroArray(n, result.data());
        printSizedDesc("thrust scan", n);
        StreamCompaction::Thrust::scan(n, result.data(), inputs[i].data());
        printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        //printArray(n, result.data(), true);
        printCmpResult(n, expected[i].data(), result.data());
    }
}

void testCompact(const std::vector<int> &sizes) {
    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    if (sizes.empty()) {
        return;
    }

    std::vector<std::vector<int>> inputs = makeTestArrays(sizes, 4, true);
    std::vector<std::vector<int>> expected(sizes.size());
    std::vector<int> expectedCounts(sizes.size());

    {
        int n = sizes[0];
        std::vector<int> result(n);
        StreamCompaction::CPU::compactWithoutScan(n, result.data(), inputs[0].data());
        StreamCompaction::CPU::compactWithScan(n, result.data(), inputs[0].data());
    }

    for (size_t i = 0; i < sizes.size(); i++) {
        int n = sizes[i];
        expected[i].resize(n);
        std::vector<int> result(n);

        printArray(n, inputs[i].data(), true);

        zeroArray(n, expected[i].data());
        printSizedDesc("cpu compact without scan", n);
        expectedCounts[i] = StreamCompaction::CPU::compactWithoutScan(n, expected[i].data(), inputs[i].data());
        printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
        printArray(expectedCounts[i], expected[i].data(), true);
        printCmpLenResult(expectedCounts[i], expectedCounts[i], expected[i].data(), expected[i].data());

        zeroArray(n, result.data());
        printSizedDesc("cpu compact with scan", n);
        int count = StreamCompaction::CPU::compactWithScan(n, result.data(), inputs[i].data());
        printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
        printArray(count, result.data(), true);
        printCmpLenResult(count, expectedCounts[i], expected[i].data(), result.data());
    }

    {
        int n = sizes[0];
        std::vector<int> result(n);
        StreamCompaction::NotSoEfficient::compact(n, result.data(), inputs[0].data());
    }

    for (size_t i = 0; i < sizes.size(); i++) {
        int n = sizes[i];
        std::vector<int> result(n);

        zeroArray(n, result.data());
        printSizedDesc("not-so-efficient compact", n);
        int count = StreamCompaction::NotSoEfficient::compact(n, result.data(), inputs[i].data());
        printElapsedTime(StreamCompaction::NotSoEfficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        //printArray(count, result.data(), true);
        printCmpLenResult(count, expectedCounts[i], expected[i].data(), result.data());
    }

    {
        int n = sizes[0];
        std::vector<int> result(n);
        StreamCompaction::Efficient::compact(n, result.data(), inputs[0].data());
    }

    for (size_t i = 0; i < sizes.size(); i++) {
        int n = sizes[i];
        std::vector<int> result(n);

        zeroArray(n, result.data());
        printSizedDesc("work-efficient compact", n);
        int count = StreamCompaction::Efficient::compact(n, result.data(), inputs[i].data());
        printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        //printArray(count, result.data(), true);
        printCmpLenResult(count, expectedCounts[i], expected[i].data(), result.data());
    }

    {
        int n = sizes[0];
        std::vector<int> result(n);
        StreamCompaction::Thrust::compact(n, result.data(), inputs[0].data());
    }

    for (size_t i = 0; i < sizes.size(); i++) {
        int n = sizes[i];
        std::vector<int> result(n);

        zeroArray(n, result.data());
        printSizedDesc("thrust compact", n);
        int count = StreamCompaction::Thrust::compact(n, result.data(), inputs[i].data());
        printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        //printArray(count, result.data(), true);
        printCmpLenResult(count, expectedCounts[i], expected[i].data(), result.data());
    }
}

void testSort(const std::vector<int> &sizes) {
    printf("\n");
    printf("**********************\n");
    printf("** RADIX SORT TESTS **\n");
    printf("**********************\n");

    if (sizes.empty()) {
        return;
    }

    std::vector<std::vector<int>> inputs = makeTestArrays(sizes, 50, false);
    std::vector<std::vector<int>> expected(sizes.size());

    {
        int n = sizes[0];
        std::vector<int> result(n);
        StreamCompaction::CPU::sort(n, result.data(), inputs[0].data());
    }

    for (size_t i = 0; i < sizes.size(); i++) {
        int n = sizes[i];
        expected[i].resize(n);

        printArray(n, inputs[i].data(), true);

        zeroArray(n, expected[i].data());
        printSizedDesc("cpu sort", n);
        StreamCompaction::CPU::sort(n, expected[i].data(), inputs[i].data());
        printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
        printArray(n, expected[i].data(), true);
    }

    {
        int n = sizes[0];
        std::vector<int> result(n);
        StreamCompaction::Radix::sort(n, result.data(), inputs[0].data());
    }

    for (size_t i = 0; i < sizes.size(); i++) {
        int n = sizes[i];
        std::vector<int> result(n);

        zeroArray(n, result.data());
        printSizedDesc("radix sort", n);
        StreamCompaction::Radix::sort(n, result.data(), inputs[i].data());
        printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        //printArray(n, result.data(), true);
        printCmpResult(n, expected[i].data(), result.data());
    }
}

int main(int argc, char* argv[]) {
    std::vector<int> testSizes;

    for (int i = 8; i < 27; i += 2) {
        testSizes.push_back(1 << i);
    }

    for (int i = 8; i < 27; i += 2) {
        testSizes.push_back((1 << i) - i);
    }

    testScan(testSizes);
    testCompact(testSizes);
    testSort(testSizes);

    system("pause"); // stop Win32 console from closing on exit
}
