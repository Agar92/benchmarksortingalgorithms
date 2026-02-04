#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include "Array2D.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <chrono>

using namespace std;

constexpr int N=5'000'000;

struct P{
  int ir;
  int id;
  double3 r;
  double3 p;
  __device__ __host__ P():ir(-1),id(-1),r{},p{}{}
  __device__ __host__ P(int _ir, int _id):ir(_ir),id(_id),r{},p{}{}
};
int main()
{
  thrust::host_vector<P> particles(N);
  for(int i=0; i<N; ++i)
  {
    particles[i].ir = rand()%5-1;
  }
/*
  for(int i=0; i<N; ++i)
    cout<<particles[i].ir<<" ";
  cout<<endl;
*/
  ///thrust::device_vector<P> particles__dev=particles;
  thrust::device_vector<P> particles__dev(N);
  thrust::copy(particles.begin(), particles.end(), particles__dev.begin());
  auto t1=std::chrono::steady_clock::now();
  
  thrust::sort(thrust::device, particles__dev.begin(), particles__dev.end(),
               [] __device__ __host__ (P & a, P & b){return a.ir>b.ir;});

  auto t2=std::chrono::steady_clock::now();

  cout<<"T="<<std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()<<" ms"<<endl;

  //thrust::host_vector<P> particles_check=particles__dev;
  thrust::host_vector<P> particles_check(N);
  thrust::copy(particles__dev.begin(), particles__dev.end(), particles_check.begin());
  
//*
  cout<<"particles_check:"<<endl;
  int cnt3=0, cnt2=0, cnt1=0, cnt0=0, cntm1=0;
  int GLOBAL_COUNTER=0;
  for(int i=0; i<N; ++i)
  {
  //cout<<particles_check[i].ir<<" ";
    if(3 == particles_check[i].ir)  cnt3++;
    if(2 == particles_check[i].ir)  cnt2++;
    if(1 == particles_check[i].ir)  cnt1++;
    if(0 == particles_check[i].ir)  cnt0++;
    if(-1 == particles_check[i].ir) cntm1++;
    GLOBAL_COUNTER++;
  }
  cout<<endl;
  cout<<cnt3<<" "<<cnt2<<" "<<cnt1<<" "<<cnt0<<" "<<cntm1<<endl;
  cout<<"GLOBAL_COUNTER="<<GLOBAL_COUNTER<<endl;
//*/

//*  
  cnt3=cnt2=cnt1=cnt0=cntm1=0;
  for(int i=0; i<N; ++i)
  {
  //cout<<particles[i].ir<<" ";
    if(3 == particles_check[i].ir) cnt3++;
    if(2 == particles_check[i].ir) cnt2++;
    if(1 == particles_check[i].ir) cnt1++;
    if(0 == particles_check[i].ir) cnt0++;
    if(-1 == particles_check[i].ir) cntm1++;
  }
  cout<<endl;
  cout<<cnt3<<" "<<cnt2<<" "<<cnt1<<" "<<cnt0<<" "<<cntm1<<endl;
//*/  
  return 0;
}