#include <iostream>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <chrono>
#include "unistd.h"

//////////////////////////////////////////////////////////////////////////////////
//HOW TO LAUNCH THIS .cu FILE on CPU//////////////////////////////////////////////
//(using Thrust for the host and no CUDA functions)///////////////////////////////
// $ nvcc -std=c++17 -O3 --extended-lambda -Xcompiler -fopenmp cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_CPU.cu -o cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_CPU
//This is the compile line for CPU.
// on NVIDIA Tesla V100-SXM2-32GB
// CUDA Version: 12.2
//////////////////////////////////////////////////////////////////////////////////

using std::cout;
using std::endl;

constexpr int N=20'000'000;
constexpr int Nbin=1024*4;

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

void print_ir_on_gpu(const thrust::host_vector<P> & vector,
                     int b);

thrust::host_vector<int> init(Nbin);
thrust::host_vector<int> fin(Nbin);
//
thrust::host_vector<int> pointer_minus1(Nbin);
thrust::host_vector<int> pointer0(Nbin);
thrust::host_vector<int> pointer1(Nbin);
thrust::host_vector<int> pointer2(Nbin);
thrust::host_vector<int> pointer3(Nbin);
thrust::host_vector<int> count_minus1(Nbin);
thrust::host_vector<int> count0(Nbin);
thrust::host_vector<int> count1(Nbin);
thrust::host_vector<int> count2(Nbin);
thrust::host_vector<int> count3(Nbin);

//sorting:
void mysort_Nthreads(thrust::host_vector<P> & parray,
                     thrust::host_vector<P> & parray2)
{
  //
  thrust::host_vector<int> COUNTbin(5*Nbin);
  //memset(COUNTbin,0,sizeof(COUNTbin));
  thrust::fill(COUNTbin.begin(), COUNTbin.begin()+5*Nbin, 0);
  thrust::host_vector<int> OFFSET(5*Nbin);
  thrust::host_vector<int> COUNT(5);
  //memset(COUNT,0,sizeof(COUNT));
  thrust::fill(COUNT.begin(), COUNT.begin()+5, 0);
  thrust::host_vector<int> COUNTERbin(5*Nbin);
  //memset(COUNTERbin,0,sizeof(COUNTERbin));
  thrust::fill(COUNTERbin.begin(), COUNTERbin.begin()+5*Nbin, 0);
  //
///////////////////////////////////////////////////////////////////////////////  
//1. BEGIN OF Create bins and fiil in the borders of the bins://///////////////
///////////////////////////////////////////////////////////////////////////////
  const int dL = N / Nbin;
  const int DL = dL + 1;
  const int n = Nbin - N%Nbin;
  int* init_pointer = thrust::raw_pointer_cast(init.data());
  int* fin_pointer = thrust::raw_pointer_cast(fin.data());
  thrust::for_each(
      thrust::host,
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(Nbin),
      [=] __device__ __host__ (int b)
      {
          int init=0;
          if(b<n)
              init=b*dL;
          else if(b==n)
              init=n*dL;
          else if(b>n)
              init=n*dL+DL*(b-n);
      ////printf("b=%d   init=%d\n", b, init);
          init_pointer[b]=init;
      }
  );
  //
  thrust::for_each(
      thrust::host,
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(Nbin),
      [=] __device__ __host__ (int b)
      {
          int fin=0;
          if(b<n)
              fin=(b+1)*dL;
          else if(b==n)
              fin=n*dL+DL;
          else if(b>n)
              fin=n*dL+DL*(b-n+1);
      ////printf("b=%d   fin=%d\n", b, fin);
          fin_pointer[b]=fin;
      }
  );
/////////////////////////////////////////////////////////////////////////////  
//1. END OF Create bins and fiil in the borders of the bins//////////////////
/////////////////////////////////////////////////////////////////////////////
      
//////////////////////////////////////////////////////////
//2. BEGIN OF sort particles in each bin://///////////////
//////////////////////////////////////////////////////////
  P* parray_pointer = thrust::raw_pointer_cast(parray.data());
  int* COUNTbin_pointer = thrust::raw_pointer_cast(COUNTbin.data());
#pragma omp parallel for
  for(int b=0; b<Nbin; ++b)
  {
      thrust::sort(
                   thrust::host,
                   parray.begin()+init[b],
                   parray.begin()+fin[b],
                   [] __device__ __host__ (P & a, P & b)
                   {
                     return a.ir<b.ir;
                   }
      );
  }
////////////////////////////////////////////////////////
//2. END OF sort particles in each bin./////////////////
////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
//3. BEGIN OF count of particles of each ir in each bin://///////////////
/////////////////////////////////////////////////////////////////////////
  thrust::fill(count_minus1.begin(), count_minus1.end(), 0);
  thrust::fill(count0.begin(),       count0.end(),       0);
  thrust::fill(count1.begin(),       count1.end(),       0);
  thrust::fill(count2.begin(),       count2.end(),       0);
  thrust::fill(count3.begin(),       count3.end(),       0);
  //
  int* count_minus1_pointer=thrust::raw_pointer_cast(count_minus1.data());
  int* count0_pointer=thrust::raw_pointer_cast(count0.data());
  int* count1_pointer=thrust::raw_pointer_cast(count1.data());
  int* count2_pointer=thrust::raw_pointer_cast(count2.data());
  int* count3_pointer=thrust::raw_pointer_cast(count3.data());
  //
  //
  thrust::for_each(
      thrust::host,
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(Nbin),
      [=] __device__ __host__ (int b)
      {
          for(int j=init_pointer[b]; j<fin_pointer[b]; ++j)
          {
            const P & pp=parray_pointer[j];
            if(-1 == pp.ir) count_minus1_pointer[b]++;
            else if(0  == pp.ir) count0_pointer[b]++;
            else if(1  == pp.ir) count1_pointer[b]++;
            else if(2  == pp.ir) count2_pointer[b]++;
            else if(3  == pp.ir) count3_pointer[b]++;
          }
      }
  );
/////////////////////////////////////////////////////////////////////////
//3. END OF count of particles of each ir in each bin.///////////////////
/////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
//4. BEGIN OF MAKE OFFSETS for writing sorted array://///////////////////
/////////////////////////////////////////////////////////////////////////
/*
  for(int b=0; b<Nbin; ++b)
  {
    print_ir_on_gpu(parray, b);
    cout<<endl;
  }
*/
  /*
  for(int b=0; b<Nbin; ++b)
  {
    cout<<"Bin #"<<b<<endl;
    cout<<"init["<<b<<"]="<<init[b]<<"   fin["<<b<<"]="<<fin[b]<<endl;
    cout<<"c-1="<<count_minus1_pointer[b]
        <<" c0="<<count0[b]
        <<" c1="<<count1[b]
        <<" c2="<<count2[b]
        <<" c3="<<count3[b]
        <<endl;
    cout<<endl;
  }
  */

  int* pointer_minus1_pointer = thrust::raw_pointer_cast(pointer_minus1.data());
  int* pointer0_pointer = thrust::raw_pointer_cast(pointer0.data());
  int* pointer1_pointer = thrust::raw_pointer_cast(pointer1.data());
  int* pointer2_pointer = thrust::raw_pointer_cast(pointer2.data());
  int* pointer3_pointer = thrust::raw_pointer_cast(pointer3.data());
  thrust::for_each(
    thrust::host,
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(1),
        [=] __device__ __host__ (int LITTER) -> void
        {
            pointer_minus1_pointer[0]=pointer0_pointer[0]=pointer1_pointer[0]=pointer2_pointer[0]=pointer3_pointer[0];
            for(int b=0; b<Nbin-1; ++b)
            {
              pointer_minus1_pointer[b+1]=pointer_minus1_pointer[b]+count_minus1_pointer[b];
              pointer0_pointer[b+1]=pointer0_pointer[b]+count0_pointer[b];
              pointer1_pointer[b+1]=pointer1_pointer[b]+count1_pointer[b];
              pointer2_pointer[b+1]=pointer2_pointer[b]+count2_pointer[b];
              pointer3_pointer[b+1]=pointer3_pointer[b]+count3_pointer[b];
            }
        }  
  );

  thrust::host_vector<int> count_minus1_host_copy(N),
                           count0_host_copy(N),
                           count1_host_copy(N),
                           count2_host_copy(N),
                           count3_host_copy(N);
  thrust::copy(count_minus1.begin(), count_minus1.end(), count_minus1_host_copy.begin());
  thrust::copy(count0.begin(), count0.end(), count0_host_copy.begin());
  thrust::copy(count1.begin(), count1.end(), count1_host_copy.begin());
  thrust::copy(count2.begin(), count2.end(), count2_host_copy.begin());
  thrust::copy(count3.begin(), count3.end(), count3_host_copy.begin());
  int COUNT_MINUS_1=0, COUNT0=0, COUNT1=0, COUNT2=0, COUNT3=0;
  for(int b=0; b<Nbin; ++b)
  {
    COUNT_MINUS_1 += count_minus1_host_copy[b];
    COUNT0 += count0_host_copy[b];
    COUNT1 += count1_host_copy[b];
    COUNT2 += count2_host_copy[b];
    COUNT3 += count3_host_copy[b];
  }

  /*
  cout<<"Check out pointers:"<<endl;
  for(int b=0; b<Nbin; ++b)
  {
    cout<<"Bin #"<<b<<":"<<endl;
    cout<<pointer_minus1_pointer[b]<<endl;
    cout<<pointer0_pointer[b]<<endl;
    cout<<pointer1_pointer[b]<<endl;
    cout<<pointer2_pointer[b]<<endl;
    cout<<pointer3_pointer[b]<<endl;
  }
  cout<<endl;
  cout<<COUNT_MINUS_1<<"   "<<COUNT0<<"   "<<COUNT1<<"   "<<COUNT2<<"   "<<COUNT3<<endl;
  */
  
/////////////////////////////////////////////////////////////////////////
//4. END OF MAKE OFFSETS for writing sorted array.///////////////////////
/////////////////////////////////////////////////////////////////////////

  thrust::host_vector<int> init_host_copy(N), fin_host_copy(N);
  thrust::copy(init.begin(), init.end(), init_host_copy.begin());
  thrust::copy(fin.begin(), fin.end(), fin_host_copy.begin());
  //
#pragma omp parallel for
  for(int b=0; b<Nbin; ++b)
  {
      const int shift1=count_minus1_host_copy[b];
      //copy -1:
      thrust::copy(
              parray.begin()+init_host_copy[b],
              parray.begin()+init_host_copy[b]+shift1,
              parray2.begin()+pointer_minus1_pointer[b]);
      const int shift2=shift1+count0_host_copy[b];
      //copy 0:
      thrust::copy(
              parray.begin()+init_host_copy[b]+shift1,
              parray.begin()+init_host_copy[b]+shift2,
              parray2.begin()+COUNT_MINUS_1+pointer0_pointer[b]);
      const int shift3=shift2+count1_host_copy[b];
      //copy 1:
      thrust::copy(
              parray.begin()+init_host_copy[b]+shift2,
              parray.begin()+init_host_copy[b]+shift3,
              parray2.begin()+COUNT_MINUS_1+COUNT0+pointer1_pointer[b]);
      const int shift4=shift3+count2_host_copy[b];
      //copy 2:
      thrust::copy(
              parray.begin()+init_host_copy[b]+shift3,
              parray.begin()+init_host_copy[b]+shift4,
              parray2.begin()+COUNT_MINUS_1+COUNT0+COUNT1+pointer2_pointer[b]);
      const int shift5=shift4+count3_host_copy[b];
      //copy 3:
      thrust::copy(
              parray.begin()+init_host_copy[b]+shift4,
              parray.begin()+init_host_copy[b]+shift5,
              parray2.begin()+COUNT_MINUS_1+COUNT0+COUNT1+COUNT2+pointer3_pointer[b]);
  }

}

int main()
{
  thrust::host_vector<P> particles(N);
  for(int i=0; i<N; ++i)
  {
    particles[i].ir = rand()%5-1;
  }
  thrust::host_vector<P> particles_sequential = particles;
/*
  for(int i=0; i<N; ++i)
    cout<<particles[i].ir<<" ";
  cout<<endl;
*/
  thrust::host_vector<P> particles__dev_in(N);
  thrust::copy(particles.begin(), particles.end(), particles__dev_in.begin());
  thrust::host_vector<P> particles__dev_out(N);
  auto t1=std::chrono::steady_clock::now();
  mysort_Nthreads(particles__dev_in, particles__dev_out);
  auto t2=std::chrono::steady_clock::now();
  cout<<"parallel on CPU sort T="<<std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()<<" ms"<<endl;
  
  //cout<<"CHECKING:"<<endl;
  thrust::host_vector<P> particles_check(N);
  thrust::copy(particles__dev_out.begin(), particles__dev_out.end(), particles_check.begin());
  //for(int i=0; i<N; ++i) cout<<particles_check[i].ir<<" ";
  //cout<<endl;


  t1=std::chrono::steady_clock::now();
  //CHECK FOR MATCHING SEQUENTIAL RESULTS:"<<endl;
  std::sort(particles_sequential.begin(), particles_sequential.end(), [] (P & a, P & b)
            {
              return a.ir<b.ir;
            });
  t2=std::chrono::steady_clock::now();
  cout<<"sequential on CPU sort T="<<std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()<<" ms"<<endl;

  bool MATCH = true;
  for(int i=0; i<N; ++i)
    if(particles_check[i] != particles_sequential[i])
      MATCH =false;
  
  if(MATCH)
    cout<<"The results MATCH sequential code!"<<endl;
  else
    cout<<"***The results DO NOT MATCH sequential code!"<<endl;
  return 0;
}


void print_ir_on_gpu(const thrust::host_vector<P> & vector, int b)
{
  const P* vector_pointer = thrust::raw_pointer_cast(vector.data());
  thrust::host_vector<int> init_host_copy(N), fin_host_copy(N);
  thrust::copy(init.begin(), init.end(), init_host_copy.begin());
  thrust::copy(fin.begin(), fin.end(), fin_host_copy.begin());
  //
  thrust::for_each(
    thrust::host,
    thrust::make_counting_iterator(init_host_copy[b]),
    thrust::make_counting_iterator(fin_host_copy[b]),
        [=] __device__ __host__ (int j) -> void {
            P pp=vector_pointer[j];
            printf("%d ", pp.ir);
        }  
  );  
}
