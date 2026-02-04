#include <iostream>
#include <vector>
#include <array>
#include <thread>
#include <algorithm>
#include <cstring>
#include <atomic>
#include <mutex>
#include <chrono>

using std::cout;
using std::endl;

// HOW TO COMPILE?
// $ g++ -std=c++17 -O3 -pthread -fopenmp parallel_sort.cpp -o parallel_sort
//

struct double3{
  float x, y, z;
};

struct P{
  int ir;
  int id;
  double3 r;
  double3 p;
  P():ir(-1),id(-1),r{},p{}{}
  P(int _ir, int _id):ir(_ir),id(_id),r{},p{}{}
  friend bool operator==(const P & l, const P & r)
  {
    return l.ir == r.ir;
  }
  friend bool operator<(const P & l, const P & r)
  {
    return l.ir < r.ir;
  }
};

class ParallelSorter {
private:
    std::vector<P>& input_;
    std::vector<P> output_;
    size_t num_threads_;
    size_t n_;

    struct ThreadData {
        size_t start_idx;
        size_t end_idx;
        std::vector<P> sorted_subarray;
        std::array<size_t, 4> counts = {0, 0, 0, 0}; // count of 0,1,2,3
        std::array<size_t, 4> write_offsets = {0, 0, 0, 0}; // offsets for writing
    };

    std::vector<ThreadData> thread_data_;
    std::vector<std::thread> threads_;
    std::array<std::atomic<size_t>, 4> global_offsets_;

    std::vector<size_t> bin_sizes_;
    std::vector<size_t> shift;//
public:
  ParallelSorter(std::vector<P>& input, size_t num_threads = std::thread::hardware_concurrency())
        : input_(input), n_(input.size()), num_threads_(num_threads) {
        output_.resize(n_);

        // Initialize atomic offsets
        for (auto& offset : global_offsets_) {
            offset.store(0);
        }

        bin_sizes_.resize(num_threads_);
        shift.resize(num_threads_);
        for(int i=0; i<num_threads_; ++i)
        {
          shift[i]=0;
          bin_sizes_[i]=0;
        }

        // Calculate subarray boundaries
        thread_data_.resize(num_threads_);
        size_t base_chunk_size = n_ / num_threads_;
        size_t remainder = n_ % num_threads_;

        size_t current_start = 0;
        for (size_t i = 0; i < num_threads_; ++i) {
            size_t chunk_size = base_chunk_size + (i < remainder ? 1 : 0);
            thread_data_[i].start_idx = current_start;
            thread_data_[i].end_idx = current_start + chunk_size;
            /*
            cout<<"Thread #"<<i
                <<" start_idx="<<thread_data_[i].start_idx
                <<" end_idx="<<thread_data_[i].end_idx
                <<" chunk_size="<<chunk_size
                <<endl;
            */
            bin_sizes_[i]=chunk_size;
            current_start += chunk_size;
        }
    }

    // Phase 1: Sort subarrays and count values
    void sort_and_count() {
        threads_.clear();

        for (size_t i = 0; i < num_threads_; ++i) {
            threads_.emplace_back([this, i]() {
                auto& data = thread_data_[i];
                size_t size = data.end_idx - data.start_idx;
                /*
                cout<<"BlockB #"<<i<<" "<<data.start_idx<<"|"<<data.end_idx<<":   ";
                for(int j=data.start_idx; j<data.end_idx; ++j)
                  cout<<input_[j].ir<<" ";
                cout<<endl;
                */
                // 1. Extract and sort subarray
                std::sort(input_.begin() + data.start_idx, input_.begin() + data.end_idx, [](const P & l, const P & r){return l.ir<r.ir;});
                // 2. Count occurrences of 0,1,2,3
                for(int j=0; j<size; ++j) {
                    int val=input_[data.start_idx+j].ir;
                    //cout<<"#-# val="<<val<<endl;
                    if (val >= 0 && val <= 3) {
                        data.counts[val]++;
                        ///cout<<"#+# val="
                        ///    <<val<<" data.counts["
                        ///    <<val<<"]="
                        ///    <<data.counts[val]
                        ///    <<endl;
                    }
                }
                /*
                cout<<"BlockA #"<<i<<" "<<data.start_idx<<"|"<<data.end_idx<<":   ";
                for(int j=data.start_idx; j<data.end_idx; ++j)
                  cout<<input_[j].ir<<" ";
                cout<<endl;
                */
            });
        }
        // 
        // Wait for all threads to complete
        for (auto& thread : threads_) {
            thread.join();
        }
        /*
        cout<<endl<<endl<<endl;
        cout<<"CHECK COUNTS:"<<endl;
        for(size_t i = 0; i < num_threads_; ++i)
        {
          cout<<"thread_data_["<<i<<"].counts: ";
          for(int j=0; j<4; ++j)
            cout<<thread_data_[i].counts[j]<<" ";
          cout<<endl;
        }
        cout<<endl;
        cout<<"Check bins contain:"<<endl;
        for(size_t i = 0; i < num_threads_; ++i)
        {
          cout<<"Bin #"<<i<<" contains:   ";
          int size=
            thread_data_[i].end_idx-
            thread_data_[i].start_idx;
          for(int j=0; j<size; ++j)
            cout<<input_[thread_data_[i].start_idx+j].ir<<" ";
          cout<<endl;
        }
        cout<<endl<<endl<<endl;
        */
    }

    // Phase 2: Calculate offsets for writing blocks
    void calculate_offsets() {
        size_t offset = 0;
        // Calculate global offsets for each value
        for (int val = 0; val <= 3; ++val) {
            size_t offset_global = 0;
            for (size_t i = 0; i < num_threads_; ++i) {
                thread_data_[i].write_offsets[val] = offset;
                offset += thread_data_[i].counts[val];
                offset_global += thread_data_[i].counts[val];
            }
            global_offsets_[val].store(offset_global);
        }
        /*
        cout<<"global_offsets_: ";
        for(int i=0; i<4; ++i) cout<<i<<"->"<<global_offsets_[i].load()<<"   ";
        cout<<endl;
        for(size_t i = 0; i < num_threads_; ++i)
        {
          cout<<"thread_data_["<<i<<"].write_offsets: ";
          for(int j=0; j<4; ++j) cout<<j<<"->"<<thread_data_[i].write_offsets[j]<<"   ";
          cout<<endl;
        }
        */
        for(size_t i=0; i<num_threads_; ++i)
          for(int j=0; j<i; ++j)
            shift[i] += bin_sizes_[j];
        /*
        cout<<"Check shifts:   ";
        for(int j=0; j<num_threads_; ++j)
          cout<<shift[j]<<" ";
        cout<<endl;
        cout<<"Check bin_sizes_:   ";
        for(int j=0; j<num_threads_; ++j)
          cout<<shift[j]<<" ";
        cout<<endl;
        */
    }

    // Phase 3: Copy blocks to output array using memcpy
    void copy_blocks() {
        threads_.clear();

        //cout<<"STEP #123"<<endl;

        for (size_t i = 0; i < num_threads_; ++i) {
          threads_.emplace_back([this, i]()
          {
                //cout<<"Thread #"<<i<<":"<<endl;
                auto& data = thread_data_[i];
                // For each value (0,1,2,3), find its block and copy it
                size_t read_pos = 0;
                size_t dest_pos = 0;
                for (int val = 0; val <= 3; ++val)
                {
                    size_t count = data.counts[val];
                    //cout<<"val="<<val<<" count="<<count<<endl;
                    if (count > 0)
                    {
                        // Calculate destination position
                        dest_pos = data.write_offsets[val];
                        // Copy the block of this value
                        std::memcpy(output_.data() + dest_pos,
                                    input_.data() + shift[i] + read_pos,
                                    count * sizeof(P));
                        /*
                        cout<<"ACopy of block #"<<i<<" "<<data.start_idx<<"|"<<data.end_idx<<":   ";
                        for(int j=data.start_idx; j<data.end_idx; ++j)
                          cout<<input_[j].ir<<" ";
                        cout<<endl;
                        */
                        /*
                        cout<<"val="<<val<<" BCopy of block #"<<i<<" "
                            <<"offset="<<(shift[i]+read_pos)<<"  "
                            <<"dest_pos="<<dest_pos<<"  count="<<count<<":   ";
                        cout<<endl;
                        */
                        read_pos += count;
                    }
                }
            });
        //cout<<"STEP #199"<<endl;
        }

        // Wait for all threads to complete
        for (auto& thread : threads_) {
            thread.join();
        }
        //cout<<"STEP #200"<<endl;
    }

    // Main function to execute all phases
    std::vector<P> sort() {
        auto start = std::chrono::high_resolution_clock::now();

        // Phase 1: Sort subarrays and count
        sort_and_count();

        // Phase 2: Calculate offsets
        calculate_offsets();

        // Phase 3: Copy blocks
        copy_blocks();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "Parallel sort completed in " << duration.count() << " microseconds" << std::endl;

        return output_;
    }

    // Verification function
    bool verify() const {
        // Check that output is sorted
        for (size_t i = 1; i < n_; ++i) {
            if (output_[i].ir < output_[i - 1].ir) {
                std::cout << "Output not sorted at index " << i << std::endl;
                return false;
            }
        }

        // Check that all values are 0-3
        for (const P & p : output_) {
            int val=p.ir;
            if (val < 0 || val > 3) {
                std::cout << "Invalid value found: " << val << std::endl;
                return false;
            }
        }

        // Check counts match
        std::array<size_t, 4> output_counts = {0, 0, 0, 0};
        for (const P & p : output_) {
            int val=p.ir;
            output_counts[val]++;
        }

        std::array<size_t, 4> input_counts = {0, 0, 0, 0};
        for (const P & p : input_) {
            int val=p.ir;
            input_counts[val]++;
        }

        for (int i = 0; i < 4; ++i) {
            if (input_counts[i] != output_counts[i]) {
                std::cout << "Count mismatch for value " << i
                         << ": input=" << input_counts[i]
                         << ", output=" << output_counts[i] << std::endl;
                return false;
            }
        }

        return true;
    }

    // Print statistics
    void print_stats() const {
        std::cout << "Input size: " << n_ << std::endl;
        std::cout << "Number of threads: " << num_threads_ << std::endl;

        std::array<size_t, 4> total_counts = {0, 0, 0, 0};
        for (const auto& data : thread_data_) {
            for (int i = 0; i < 4; ++i) {
                total_counts[i] += data.counts[i];
            }
        }

        int Total=0;
        for(int i=0; i<4; ++i)
          Total += total_counts[i];

        std::cout << "Counts: 0=" << total_counts[0]
                 << ", 1=" << total_counts[1]
                 << ", 2=" << total_counts[2]
                 << ", 3=" << total_counts[3] << std::endl;
        cout<<"Total="<<Total<<endl;
    }
};

// Alternative implementation using OpenMP for simpler parallelization
//#ifdef _OPENMP
#include <omp.h>

class OpenMPParallelSorter {
private:
    std::vector<P>& input_;
    std::vector<P> output_;
    size_t n_;

public:
    OpenMPParallelSorter(std::vector<P>& input)
        : input_(input), n_(input.size()) {
        output_.resize(n_);
    }

    std::vector<P> sort() {
        auto start = std::chrono::high_resolution_clock::now();
        int num_threads;
        #pragma omp parallel
        {
          #pragma omp single
            num_threads = omp_get_num_threads();
        }
        std::vector<std::array<size_t, 4>> counts(num_threads);
        std::vector<std::array<size_t, 4>> offsets(num_threads);
        // Phase 1: Sort subarrays and count (parallel)
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            size_t chunk_size = n_ / num_threads;
            size_t remainder = n_ % num_threads;
            size_t start_idx = tid * chunk_size + std::min(tid, (int)remainder);
            size_t end_idx = start_idx + chunk_size + (tid < remainder ? 1 : 0);
            std::sort(input_.data()+start_idx, input_.data()+end_idx);
            for(int j=0; j<(end_idx-start_idx); ++j)
            {
              int val=(*(input_.data()+start_idx+j)).ir;
                if(val >= 0 && val <= 3)
                    counts[tid][val]++;
            }
        }
        /*
        cout<<"Check subarrays:"<<endl;
        for(int tid=0; tid<4; ++tid)
        {
          cout<<"Sub array #"<<tid<<":"<<endl;
          for(int j=0; j<sorted_subarrays.at(tid).size(); ++j)
            cout<<sorted_subarrays[tid][j]<<endl;
        }
        cout<<"Check counts:"<<endl;
        for(int tid=0; tid<4; ++tid)
        {
          for(int val=0; val<4; ++val)
            cout<<"counts["<<tid<<"]["<<val<<"]="<<counts[tid][val]<<endl;
        }
        */

        // Phase 2: Calculate offsets (sequential)
        std::array<size_t, 4> global_offsets = {0, 0, 0, 0};
        size_t offset = 0;
        for (int val = 0; val <= 3; ++val) {
            size_t offset_global = 0;
            for (int tid = 0; tid < num_threads; ++tid) {
                offsets[tid][val] = offset;
                offset        += counts[tid][val];
                offset_global += counts[tid][val];
            }
            global_offsets[val] = offset_global;
        }

        /*
        cout<<"global_offsets: ";
        for(int i=0; i<4; ++i) cout<<i<<"->"<<global_offsets[i]<<"   ";
        cout<<endl;
        for(size_t tid=0; tid<num_threads ; ++tid)
        {
          cout<<"offsets["<<tid<<"]: ";
          for(int j=0; j<4; ++j) cout<<offsets[tid][j]<<"  ";
          cout<<endl;
        }
        */

        // Phase 3: Copy blocks (parallel)
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            size_t chunk_size = n_ / num_threads;
            size_t remainder = n_ % num_threads;
            size_t start_idx = tid * chunk_size + std::min(tid, (int)remainder);
            size_t read_pos = 0;
            for (int val = 0; val <= 3; ++val) {
                size_t count = counts[tid][val];
                if (count > 0) {
                    size_t dest_pos = offsets[tid][val];
                    std::memcpy(output_.data() + dest_pos,
                                input_.data()+start_idx+read_pos,
                                count * sizeof(P));
                    read_pos += count;
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout<<"OpenMP sort completed in "<<duration.count()
                 <<" microseconds using "<<num_threads<<" threads"
                 <<std::endl;

        return output_;
    }
};
//#endif

// Helper function to generate test data
std::vector<P> generate_test_data(size_t n) {
    std::vector<P> data(n);
////std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (size_t i = 0; i < n; ++i) {
        data[i].ir = std::rand() % 4; // Random values 0-3
    }

    return data;
}

// Sequential reference implementation for verification
std::vector<P> sequential_sort(const std::vector<P>& input) {
    std::vector<P> result = input;
    std::sort(result.begin(), result.end());
    return result;
}

int main() {
    // Test with different sizes
  std::vector<size_t> test_sizes = {1'000, 10'000, 100'000, 1'000'000, 10'000'000, 20'000'000};

    for (size_t size : test_sizes) {
        std::cout << "\n=== Testing with size " << size << " ===" << std::endl;

        // Generate test data
        auto input = generate_test_data(size);

        /*
        cout<<">>>Initial array: ";
        for(int i=0; i<input.size(); ++i) cout<<input[i].ir<<" ";
        cout<<endl;
        */

        // Run sequential sort for reference
        auto start_seq = std::chrono::high_resolution_clock::now();
        auto seq_result = sequential_sort(input);
        /*
        cout<<">>>Result of sequential sort: ";
        for(int i=0; i<seq_result.size(); ++i) cout<<seq_result[i].ir<<" ";
        cout<<endl;
        */
        auto end_seq = std::chrono::high_resolution_clock::now();
        auto seq_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_seq - start_seq);

        // Run parallel sort
        ParallelSorter sorter(input);
        auto par_result = sorter.sort();
        sorter.print_stats();
        /*
        cout<<">>>Result of parallel sort: ";
        for(int i=0; i<par_result.size(); ++i) cout<<par_result[i].ir<<" ";
        cout<<endl;
        */

        // Verify results
        if (sorter.verify()) {
            std::cout << "Verification passed!" << std::endl;
        } else {
            std::cout << "Verification failed!" << std::endl;
        }

        // Check against sequential result
        if (seq_result == par_result) {
            std::cout << "Results match sequential sort!" << std::endl;
        } else {
            std::cout << "Results differ from sequential sort!" << std::endl;
        }

        std::cout << "Sequential time: " << seq_duration.count() << " microseconds" << std::endl;

    ////#ifdef _OPENMP
        // Test OpenMP version
        OpenMPParallelSorter omp_sorter(input);
        auto omp_result = omp_sorter.sort();
        /*
        cout<<">>>Result of OpenMP parallel sort: ";
        for(int i=0; i<omp_result.size(); ++i) cout<<omp_result[i].ir<<" ";
        cout<<endl;
        */
        if (seq_result == omp_result) {
            std::cout << "OpenMP results match!" << std::endl;
        }
    ////#endif

        std::cout << "=============================" << std::endl;
    }

    return 0;
}

/*
  
Key Features of the Implementation:

1. Three-Phase Design:

-· Phase 1: Each thread sorts its subarray and counts 0s, 1s, 2s, 3s
-· Phase 2: Calculate write offsets based on counts (prefix sums)
-· Phase 3: Copy sorted blocks to output using calculated offsets

2. Efficient Memory Usage:

-· Each thread works on its own subarray copy
-· Uses memcpy for bulk memory transfers
-· No locks during the parallel copy phase

3. Thread-Safe Offset Calculation:

std::array<std::atomic<size_t>, 4> global_offsets_;

Atomic counters ensure safe offset accumulation

4. Proper Subarray Division:

Handles cases where array size isn't divisible by thread count:

size_t base_chunk_size = n_ / num_threads_;
size_t remainder = n_ % num_threads_;
size_t chunk_size = base_chunk_size + (i < remainder ? 1 : 0);

5. Verification Functions:

Â· Checks that output is sorted
Â· Verifies all values are 0-3
Â· Ensures input and output counts match

Compilation and Usage:

# Compile with C++17 and threading support
g++ -std=c++17 -O3 -pthread parallel_sort.cpp -o parallel_sort

# Run the program
./parallel_sort

# For OpenMP version
g++ -std=c++17 -O3 -fopenmp parallel_sort.cpp -o parallel_sort_omp
./parallel_sort_omp

Performance Considerations:

1. Thread Overhead: For small arrays (< 1000 elements), sequential sort might be faster
2. Memory Bandwidth: Bulk memcpy operations are efficient for large arrays
3. Cache Friendly: Each thread works on contiguous memory chunks
4. Load Balancing: Uneven subarray sizes when N not divisible by thread count

Alternative Implementation:

The code also includes an OpenMP version which is simpler and often more efficient for CPU-parallel workloads. OpenMP handles thread management automatically and provides better load balancing.

This implementation efficiently sorts arrays containing only values 0, 1, 2, 3 using parallel processing, with proper synchronization and verification.

*/
