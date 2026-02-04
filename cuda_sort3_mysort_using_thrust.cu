#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include "Array2D.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <chrono>
#include "unistd.h"

//////////////////////////////////////////////////////////////////////////////////
//HOW TO LAUNCH THIS FILE/////////////////////////////////////////////////////////
// $ nvcc -arch=sm_30 --expt-extended-lambda -std=c++14 -rdc=true -O3 --use_fast_math cuda_sort3_mysort_using_thrust.cu -o cuda_sort3_mysort_using_thrust
// on NVIDIA GeForce GTX 650 Ti
// CUDA Version: 11.4
//////////////////////////////////////////////////////////////////////////////////

using std::cout;
using std::endl;

constexpr int N=5'000'000;
constexpr int Nbin=1024;

struct P{
  int ir;
  int id;
  double3 r;
  double3 p;
  __device__ __host__ P():ir(-100),id(-1),r{},p{}{}
  __device__ __host__ P(int _ir, int _id):ir(_ir),id(_id),r{},p{}{}
};


//sorting:
void mysort_Nthreads(thrust::device_vector<P> & parray,
                     thrust::device_vector<P> & parray2)
{
  //
  thrust::device_vector<int> COUNTbin(5*Nbin);
  //memset(COUNTbin,0,sizeof(COUNTbin));
  thrust::fill(COUNTbin.begin(), COUNTbin.begin()+5*Nbin, 0);
  thrust::device_vector<int> OFFSET(5*Nbin);
  thrust::device_vector<int> COUNT(5);
  //memset(COUNT,0,sizeof(COUNT));
  thrust::fill(COUNT.begin(), COUNT.begin()+5, 0);
  thrust::device_vector<int> COUNTERbin(5*Nbin);
  //memset(COUNTERbin,0,sizeof(COUNTERbin));
  thrust::fill(COUNTERbin.begin(), COUNTERbin.begin()+5*Nbin, 0);
  //
  //Create bins:
  thrust::device_vector<int> init(Nbin);
  thrust::device_vector<int> fin(Nbin);
  const int dL = N / Nbin;
  const int DL = dL + 1;
  const int n = Nbin - N%Nbin;
  int* init_pointer = thrust::raw_pointer_cast(init.data());
  int* fin_pointer = thrust::raw_pointer_cast(fin.data());
  thrust::for_each(
      thrust::device,
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(Nbin),
      [=] __device__ (int b){
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
  thrust::for_each(
      thrust::device,
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(Nbin),
      [=] __device__ (int b){
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
  //sleep(1);
  //
//#pragma omp parallel for
//for(int b=0; b<Nbin; b++)    
//  for(int j=init[b]; j<fin[b]; j++)
//    COUNTbin[(3-parray[j].ir)+b*5]++;
  
  P* parray_pointer = thrust::raw_pointer_cast(parray.data());
  int* COUNTbin_pointer = thrust::raw_pointer_cast(COUNTbin.data());

  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(Nbin),
        [=] __device__ (int b) -> void {
        ////printf("#1 init=%d fin=%d\n", init_pointer[b], fin_pointer[b]);
            for(int k=init_pointer[b]; k<fin_pointer[b]; k++)
            {
                const P pp=parray_pointer[k];
                COUNTbin_pointer[(3-pp.ir)*Nbin+b]++;
            ////printf("#2 k=%d b=%d ir=%d\n", k, b, pp.ir);
            }
        }  
  );
  //

/*
////BEGIN OF THRUST::PRINT//
  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(Nbin),
        [=] __device__ (int b) -> void {            
            for(int j=0; j<5; ++j)
                printf("#111 b=%d j=%d C[%d][%d]=%d\n",
                        b, j, j, b, COUNTbin_pointer[j*Nbin+b]);
        }  
  );  
////END OF THRUST::PRINT//
*/

//#pragma omp parallel for
//for(int b=0; b<Nbin; b++)
//  for(int j=0; j<5; j++)
//    COUNT[j] += COUNTbin[b*5+j];
  int* COUNT_pointer = thrust::raw_pointer_cast(COUNT.data());
  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(5),
        [=] __device__ (int i) -> void {
            for(int b=0; b<Nbin; ++b)
                COUNT_pointer[i] += COUNTbin_pointer[b+i*Nbin];
        }  
  );

/*
////BEGIN OF THRUST::PRINT//
  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(5),
        [=] __device__ (int j) -> void {            
            printf("#121 ir=%d C[%d]=%d\n",
                    3-j, j, COUNT_pointer[j]);
        }  
  );  
////END OF THRUST::PRINT//
*/

//cout<<"STEP #0"<<endl;
  
//OFFSET[0]=0;
//for(int j=1; j<5; ++j)
//  OFFSET[j*Nbin+0]=OFFSET[j-1*Nbin+0]+COUNT[j-1];

//OFFSET[0][0]=0;
//for(int j=1; j<5; ++j)
//    OFFSET[j][0]=OFFSET[j-1][0]+COUNT[j-1];

  int* OFFSET_pointer = thrust::raw_pointer_cast(OFFSET.data());
  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(1),
        [=] __device__ (int LITTER) -> void {
            OFFSET_pointer[0] = 0;///OFFSET[0]=0;
            for(int j=1; j<5; ++j)
            OFFSET_pointer[j*Nbin] =
                OFFSET_pointer[(j-1)*Nbin+0]+
                COUNT_pointer[j-1];
        }  
  );

/*
////BEGIN OF THRUST::PRINT//
  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(5),
        [=] __device__ (int j) -> void {
            for(int b=0; b<Nbin; b++)
                printf("#1#@#1 OFFSET[%d][%d]=%d\n",
                        j, b, OFFSET_pointer[j*Nbin+b]);
        }  
  );
////END OF THRUST::PRINT//
*/

//cout<<"STEP #1"<<endl;

//#pragma omp parallel for
//for(int b=1; b<Nbin; b++)
//  for(int j=0; j<5; j++)
//    OFFSET[j+b]=COUNTbin[j+(b-1)*5]+OFFSET[j+(b-1)*5];

/*
  for(int j=0; j<5; j++)
      for(int b=1; b<Nbin; b++)
          OFFSET[j][b]=COUNTbin[j][b-1]+OFFSET[j][b-1];
*/
  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(5),
        [=] __device__ (int j) -> void {
            for(int b=1; b<Nbin; b++)
            {
                OFFSET_pointer[j*Nbin+b] =
                    COUNTbin_pointer[j*Nbin+(b-1)] +
                    OFFSET_pointer[j*Nbin+(b-1)];
            }
        }  
  );

/*
////BEGIN OF THRUST::PRINT//
  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(5),
        [=] __device__ (int j) -> void {
            for(int b=0; b<Nbin; b++)
                printf("#131 OFFSET[%d][%d]=%d\n",
                        j, b, OFFSET_pointer[j*Nbin+b]);
        }  
  );  
////END OF THRUST::PRINT//
*/

  int* COUNTERbin_pointer = thrust::raw_pointer_cast(COUNTERbin.data());

//cout<<"STEP #2"<<endl;

//#pragma omp parallel for
//for(int b=0; b<Nbin; b++)    
//{
//  for(int j=init[b]; j<fin[b]; j++)
//  {
//    const int ir=3-parr[j].ir;
//    parr2[ OFFSET[ir][b] + COUNTERbin[ir][b]++ ]=parr[j];
//  }
//}
  P* parray2_pointer = thrust::raw_pointer_cast(parray2.data());
  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(Nbin),
        [=] __device__ (int b) -> void {
            for(int k=init_pointer[b]; k<fin_pointer[b]; k++)
            {
                const P pp=parray_pointer[k];
                int ir=3-pp.ir;
                parray2_pointer[ OFFSET_pointer[ir*Nbin+b] +
                                 COUNTERbin_pointer[ir*Nbin+b]++ ] =
                                 parray_pointer[k];
            }
        }  
  );
}


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
  thrust::device_vector<P> particles__dev(N);
  thrust::copy(particles.begin(), particles.end(), particles__dev.begin());
  auto t1=std::chrono::steady_clock::now();
  

  thrust::device_vector<P> particles__dev2(N);
  mysort_Nthreads(particles__dev, particles__dev2);


  auto t2=std::chrono::steady_clock::now();

  cout<<"T="<<std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()<<" ms"<<endl;

  //thrust::host_vector<P> particles_check=particles__dev;
  thrust::host_vector<P> particles_check(N);
  thrust::copy(particles__dev2.begin(), particles__dev2.end(), particles_check.begin());
  
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