//!!!ATTENTION!!!//
//To run this code, one shoul have installed
//g++
//nvidia-cuda-toolkit/nvidia-hpc-sdk (the latter is much bigger and has much more libraries)
//CUDA thrust library (it is typically installed along with nvidia-cuda-toolkit/nvidia-hpc-sdk)
//Also, the CUDA cub library should be installed.
//
//On my PC Intel Core i7-3770 there is an old version of CUDA and the NVIDIA driver, not up-to-date.
//And the latest version of the cub library were incompatible with my CUDA version,
//because there occurred the following error:
//version of cub cub-1.14.0 is necessary to get rid of #include <cuda/std/utility> error
//<cuda/std/...> is an NVIDIA C++ standard library and bacame avalaible only since CUDA 11.0+
//Before, there was no NVIDIA C++ standard library.
//
//To get rid of this error, I had to install and od version of the cub library: cub-1.14.0 from:
// https://github.com/NVIDIA/cub/releases/tag/1.14.0

//But there also occurred an error, because THRUST_NS_QUALIFIER was not defined.
//The workaround to fix this error is:
#define THRUST_NS_QUALIFIER thrust //cub version is much older than thrust version//this is necessary to fix this point


#include <iostream>
#include <cmath>
#include <random>
#include <vector>

#include <stdio.h>
#include <algorithm>
#include <chrono>

#include <cub/config.cuh>
#include <cub/util_type.cuh>
#include <cub/util_allocator.cuh>
#include <cub/util_namespace.cuh>
#include <cub/version.cuh>
#include <cub/device/device_radix_sort.cuh>

using std::cout;
using std::endl;

//HOW TO COMPILE THIS CODE on CPU Intel Core i7-3770 with GPU NVIDIA GeForce GTX 650 Ti:
// $ nvcc -arch=sm_30 -O3 cuda_cub_radix_sort_pairs.cu -o cuda_cub_radix_sort_pairs -I cub-1.14.0/
//
//
//
//HOW TO COMPILE THIS CODE on the partition 'gpu' of the VNIIA's cluster VKPP with GPU NVIDIA Tesla V100-SXM2-32GB:
// $nvcc -arch=sm_70 -O3 cuda_cub_radix_sort_pairs.cu -o cuda_cub_radix_sort_pairs
// (on modern versions of nvidia-cuda-toolkit and nvidia_hpc_sdk the library 'cub' is already included and is contained in these packages nvidia-cuda-toolkit / nvidia_hpc_sdk, so you do not need to specify the include path manually (-I cub-1.14.0/) and copy the source code of the library (the directory cub-1.14.0/) any more)

constexpr int N=20'000'000;

struct P{
  int ir;
  int id;
  double3 r;
  double3 p;
  __device__ __host__ P():ir(-100),id(-1),r{},p{}{}
  __device__ __host__ P(int _ir, int _id):ir(_ir),id(_id),r{},p{}{}
  __device__ __host__
  friend bool operator==(const P & lhs, const P & rhs)
  {
    return (lhs.ir==rhs.ir);
  }
  friend bool operator!=(const P & lhs, const P & rhs)
  {
    return (lhs.ir!=rhs.ir);
  }
};

int main()
{
  P* h_points=new P[N];
  int* h_keys=new int[N];
  for(int i=0; i<N; i++)
  {
    h_points[i]={rand()%5-1,-9};
    h_keys[i]=h_points[i].ir;
  }
/*
  cout<<"Before sort:"<<endl;
  for(int i=0; i<N; ++i) cout<<h_points[i].ir<<" ";
  cout<<endl;
*/
  //copy to std::vector for sequential sort and comparison:
  std::vector<P> seq_vector(N);
  for(int i=0; i<N; ++i) seq_vector[i]=h_points[i];
  //
  P* d_points;
  int* d_keys;
  auto t1=std::chrono::steady_clock::now();
  cudaMalloc(&d_points, N*sizeof(P));
  cudaMalloc(&d_keys, N*sizeof(int));
  //
  cudaMemcpy(d_keys, h_keys, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_points, h_points, N*sizeof(P), cudaMemcpyHostToDevice);
  //
  void* d_temp_storage=nullptr;
  size_t temp_storage_bytes;
  //*
  auto t2=std::chrono::steady_clock::now();
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                  d_keys, d_keys,
                                  d_points, d_points,
                                  N);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                  d_keys, d_keys,
                                  d_points, d_points,
                                  N);
  //without this comman the code works ~50 times faster
  //and the results of the parallel radix sort match the results of the sequential sort
  //But it is correct to wait for all the threads to finish here.
  //IT IS MORE CORRECT TO PUT THIS COMMAND HERE
  //IT IS A RULE OF A GOOD TONE
  //IT IS WRITTEN IN THE CUDA HANDBOOK AND PROGRAMMING GUIDE
  //BUT PUTTING THE BELOW COMMAND SLOWS DOWN THE CODE EXECUTION DRAMATICALLY
  //
  //YOU CAN TRY TO COMMENT IT,
  //BUT SHOULD THOROUGHLY CHECK FOR THE CORRECTNESS OF THE RESULTS!
  cudaDeviceSynchronize();
  auto t3=std::chrono::steady_clock::now();
  cout<<"time spent on copying the data from host to device="<<std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()<<" us"<<endl;
  cout<<"time spent on parallel radix sorting on GPU using radix sorting algorithm="<<std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count()<<" us"<<endl;
  cout<<"overall time spent on parallel radix sorting on GPU using radix sorting algorithm="<<std::chrono::duration_cast<std::chrono::microseconds>(t3-t1).count()<<" us"<<endl;
  //*/
  cudaMemcpy(h_points, d_points, N*sizeof(P), cudaMemcpyDeviceToHost);
/*
  cout<<"After parallel radix sort:"<<endl;
  for(int i=0; i<N; ++i) cout<<h_points[i].ir<<" ";
  cout<<endl;
*/
  auto t4=std::chrono::steady_clock::now();
  //here is the sequential sort and comparison:
  std::sort(seq_vector.begin(), seq_vector.end(),
            [] (P & a, P & b)
            {
                return a.ir<b.ir;
            });
  auto t5=std::chrono::steady_clock::now();
  cout<<"time spent on std::sort on CPU using standard sorting algorithm="<<std::chrono::duration_cast<std::chrono::microseconds>(t5-t4).count()<<" us"<<endl;
/*
  cout<<"After sequential std::sort:"<<endl;
  for(int i=0; i<N; ++i) cout<<seq_vector[i].ir<<" ";
  cout<<endl;
*/
  //comparison for the match:
  bool MATCH=true;
  for(int i=0; i<N; ++i)
    if(h_points[i].ir != seq_vector[i].ir)
    {
      cout<<h_points[i].ir<<" | "<<seq_vector[i].ir<<endl;
      MATCH=false;
    }
  if(MATCH)
    cout<<"The results of the parallel radix sort on GPU and the sequential std::sort on CPU MATCH!!!"<<endl;
  else
    cout<<"***The results of the parallel radix sort on GPU and the sequential std::sort on CPU DO NOT MATCH!!!"<<endl;
  //
  delete [] h_points;
  cudaFree(d_points);
  cudaFree(d_temp_storage);
  //
  return 0;
}
