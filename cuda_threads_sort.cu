#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <chrono>

#include <cuda_runtime.h>

//Compiler flags:
//$ nvcc -O3 --use_fast_math cuda_sort1_cuda_threads_without_copying_to_temporary_arrays.cu -o cuda_sort1_cuda_threads_without_copying_to_temporary_arrays



constexpr int Nbins=1024;



constexpr int N=1'000'000;
constexpr int GL=N;
constexpr int M=GL/Nbins+1;
constexpr int GL1=M-1;


constexpr int nStreams=16;
constexpr int NumberOfThreadsPerBlock=256;
constexpr int NbinsPerStream=Nbins/nStreams;
constexpr int NumberOfBlocks=
              (NbinsPerStream+NumberOfThreadsPerBlock-1)/
               NumberOfThreadsPerBlock;

struct P{
  int ir;
  int id;
  double3 r;
  double3 p;
  P():ir(-1),id(-1),r{},p{}{}
  P(int _ir, int _id):ir(_ir),id(_id),r{},p{}{}
  operator unsigned int() const
  {
    return static_cast<unsigned int> (ir) ^ 0x80000000;
  }
  bool operator>(const P & rhs) const
  {
    return (ir>rhs.ir);
  }
  bool operator<(const P & rhs) const
  {
    return (ir<rhs.ir);
  }
  bool operator==(const P & rhs) const
  {
    return (ir==rhs.ir);
  }
  bool operator!=(const P & rhs) const
  {
    return (ir!=rhs.ir);
  }
};

template <typename T>
__device__ void swap1(T& a, T& b){T c(a); a=b; b=c;}

template <typename T>
__device__ void swap2(T a, T b){T c(a); a=b; b=c;}

template <typename T>
__device__ void thread_memcpy(T* dst, const T* src, int N)
{
  for(int j=0; j<N; ++j)
    dst[j]=src[j];
}


__global__ void initialize(
                int* ind01, int* ind23, int Nbins,
                int* init, int* fin,
                int* count01, int* count23,
                int* count_minus1, int* count0, int* count1, int* count2, int* count3,
                int* POSITION3, int* POSITION2, int* POSITION1, int* POSITION0, int* POSITION_MINUS1, int* POSITION23)
{
  __const__ int dL=N/Nbins;
  __const__ int DL=dL+1;
  __const__ int n=Nbins-N%Nbins;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int i = idx;
//1) Find borders of bins:
  if(i < Nbins)
  {
    POSITION_MINUS1[i]=POSITION0[i]=POSITION1[i]=POSITION2[i]=POSITION3[i]=POSITION23[i]=0;
    count01[i]=GL1;
    count23[i]=0;
    count0[i]=GL1;
    count1[i]=0;
    count2[i]=GL1;
    count3[i]=0;
    //43   4
    //dL=43/4=10
    //DL=11
    //extra=3
    if(i<n)
    {
      init[i]=dL*i;
      fin [i]=dL*(i+1);
    }
    else if(i == n)
    {
      init[i]=dL*n;
      fin [i]=n*dL+DL;
    }
    else//if(i>n)
    {
      init[i]=n*dL+DL*(i-n);
      fin [i]=n*dL+DL*(i-n+1);
    }
  }
}

__global__ void kernel(int StreamIndex,
                       int* init, int* fin,
                       int* ind01, int* ind23,
                       P* particles__dev,
                       int* mini, int* ii0, int* ii1, int* ii3, int* ii23,
                       int* POSITION_MINUS1, int* POSITION0, int* POSITION1, int* POSITION2, int* POSITION3, int* POSITION23,
                       int* _POSITION_MINUS1_, int* _POSITION0_, int* _POSITION1_, int* _POSITION2_, int* _POSITION3_, int* _POSITION23_,
                       int* count01, int* count23, int* count_minus1, int* count0, int* count1, int* count2, int* count3,
                       int* pointer_minus1, int* pointer0, int* pointer1, int* pointer2, int* pointer3,
                       int* StreamBinInit_d, int* StreamBinEnd_d
                      )
{
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const int kernel_offset = StreamBinInit_d[StreamIndex];
  const int i = kernel_offset + idx;
  const bool ThreadCondition=
    StreamBinInit_d[StreamIndex] <= i &&
    i < StreamBinEnd_d[StreamIndex] &&
    i < Nbins;

  __const__ int dL=N/Nbins;
  __const__ int DL=dL+1;
  __const__ int n=Nbins-N%Nbins;
  //1) Find borders of bins:
  if(ThreadCondition)
  {
    POSITION_MINUS1[i]=POSITION0[i]=POSITION1[i]=POSITION2[i]=POSITION3[i]=POSITION23[i]=0;
    count01[i]=GL1;
    count23[i]=0;
    count0[i]=GL1;
    count1[i]=0;
    count2[i]=GL1;
    count3[i]=0;
    //43   4
    //dL=43/4=10
    //DL=11
    //extra=3
    if(i<n)
    {
      init[i]=dL*i;
      fin [i]=dL*(i+1);
    }
    else if(i == n)
    {
      init[i]=dL*n;
      fin [i]=n*dL+DL;
    }
    else//if(i>n)
    {
      init[i]=n*dL+DL*(i-n);
      fin [i]=n*dL+DL*(i-n+1);
    }
  }
//__syncthreads();
//2) Find counts of ir=-1,0,1 and ir=2,3:
  if(ThreadCondition)
  {
    for(int j=init[i]; j<fin[i]; ++j)
    {
      if(particles__dev[j].ir<2)
      {
        ind23[i*M+count01[i]]=j;
        count01[i] = count01[i] - 1;
      }
      else
      {
        ind23[i*M+count23[i]]=j;
        count23[i] = count23[i] + 1;
      }
    }
  }
//__syncthreads();
//3) Divide ir=2,3 from ir=-1,0,1.
//   Find counts of ir=-1,0 and ir=1 and ir=2 and ir=3:
//2 3 3 2 3 -1 0 1 1
  if(ThreadCondition)
  {
    ii23[i]=count23[i]-1;
    mini[i]=GL1-count01[i];
    if(count23[i]<mini[i]) mini[i]=count23[i];
    int js=0;
//#pragma omp simd reduction(+:js)
    for(int j=0; j<mini[i]; ++j)
      if (ind23[i*M+ii23[i] - j] > ind23[i*M+GL1 - j]) ++js;
//#pragma omp simd
    for(int j=0; j<js; ++j) swap1(particles__dev[ind23[i*M+ii23[i]-j]],particles__dev[ind23[i*M+GL1-j]]);
    for(int j=init[i]; j<fin[i]; ++j)
    {
      if     (particles__dev[j].ir==-1 ||
              particles__dev[j].ir==0) ind01[i*M+count0[i]--]=j;
      else if(particles__dev[j].ir==1) ind01[i*M+count1[i]++]=j;
      else if(particles__dev[j].ir==2) ind23[i*M+count2[i]--]=j;
      else                             ind23[i*M+count3[i]++]=j;
    }
  }
//__syncthreads();
//4) Divide ir=1 from ir=-1,0.
//   Divide ir=3 from ir=2.
//3 3 3 2 2 1 1 -1 0
  //#pragma omp parallel for
  if(ThreadCondition)    
  {
    ii1[i]=count1[i]-1;
    mini[i]=GL1-count0[i];
    if(count1[i]<mini[i]) mini[i]=count1[i];
    int js=0;
    //#pragma omp simd reduction(+:js)
    for(int j=0; j<mini[i]; ++j)
      if (ind01[i*M+ii1[i] - j] > ind01[i*M+GL1 - j]) ++js;
    //#pragma omp simd
    for(int j=0; j<js; ++j) swap1(particles__dev[ind01[i*M+ii1[i]-j]],particles__dev[ind01[i*M+GL1-j]]);
    ii3[i]=count3[i]-1;
    mini[i]=GL1-count2[i];
    if(count3[i]<mini[i]) mini[i]=count3[i];
    js=0;
    //#pragma omp simd reduction(+:js)
    for(int j=0; j<mini[i]; ++j)
      if (ind23[i*M+ii3[i] - j] > ind23[i*M+GL1 - j]) ++js;
    //#pragma omp simd
    for(int j=0; j<js; ++j) swap1(particles__dev[ind23[i*M+ii3[i]-j]],particles__dev[ind23[i*M+GL1-j]]);
  }
//__syncthreads();
//5) Divide ir=0 from ir=-1:
//3 3 3 2 2 1 1 0 -1  
  //#pragma omp parallel for
  if(ThreadCondition)   
  {
    const int save_value_of_count0_b=count0[i];
    count0[i]=0;
    count_minus1[i]=GL1;
    const int c3=count3[i];
    const int c2=GL1-count2[i];
    const int c1=count1[i];
    for(int j=init[i]+c3+c2+c1; j<fin[i]; ++j)
    {
      if     (particles__dev[j].ir== 0) ind01[i*M+count0[i]++]=j;
      else if(particles__dev[j].ir==-1) ind01[i*M+count_minus1[i]--]=j;
    }
    count_minus1[i]=GL1-count_minus1[i];
    ii0[i]=count0[i]-1;
    mini[i]=count_minus1[i];
    if(count0[i]<mini[i]) mini[i]=count0[i];
    int js=0;
    //#pragma omp simd reduction(+:js)
    for(int j=0; j<mini[i]; ++j)
      if (ind01[i*M+ii0[i] - j] > ind01[i*M+GL1 - j]) ++js;
    //#pragma omp simd
    for(int j=0; j<js; ++j)
      swap1(particles__dev[ind01[i*M+ii0[i]-j]],particles__dev[ind01[i*M+GL1-j]]);
    count0[i]=save_value_of_count0_b;
  }
//__syncthreads();
  //#pragma omp parallel for reduction(+:POSITION_MINUS1,POSITION0,POSITION1,POSITION2,POSITION3,POSITION23)
  if(ThreadCondition)
  {
    count0[i]=GL1-count0[i] - count_minus1[i];
    count2[i]=GL1-count2[i];
    POSITION_MINUS1[i]+=count_minus1[i];
    POSITION0[i]+=count0[i];
    POSITION1[i]+=count1[i];
    POSITION2[i]+=count2[i];
    POSITION3[i]+=count3[i];
    POSITION23[i]+=count23[i];
  }
//__syncthreads();
  if(ThreadCondition)
  {
    atomicAdd(_POSITION_MINUS1_, POSITION_MINUS1[i]);
    atomicAdd(_POSITION0_,       POSITION0[i]);
    atomicAdd(_POSITION1_,       POSITION1[i]);
    atomicAdd(_POSITION2_,       POSITION2[i]);
    atomicAdd(_POSITION3_,       POSITION3[i]);
    atomicAdd(_POSITION23_,      POSITION23[i]);
  }
//__syncthreads();
}

__global__ void calculate_offsets(
                int* count_minus1, int* count0,
                int* count1, int* count2, int* count3,
                int* pointer_minus1, int* pointer0,
                int* pointer1, int* pointer2, int* pointer3
                )
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(0 == i)
  {
    pointer0[0]=pointer1[0]=pointer2[0]=pointer3[0]=0;
    pointer_minus1[0]=0;
    for(int b=0; b<Nbins-1; ++b)
    {
      pointer_minus1[b+1]=pointer_minus1[b]+count_minus1[b];
      pointer0[b+1]=pointer0[b]+count0[b];
      pointer1[b+1]=pointer1[b]+count1[b];
      pointer2[b+1]=pointer2[b]+count2[b];
      pointer3[b+1]=pointer3[b]+count3[b];
    }
  }
//__syncthreads();
}

__global__ void memcpy(int StreamIndex,
                       P* particles__dev,
                       P* particles__output__dev,
                       int* init, int* fin,
                       int* count_minus1, int* count0, int* count1, int* count2, int* count3,
                       int* pointer_minus1, int* pointer0, int* pointer1, int* pointer2, int* pointer3,
                       int* _POSITION_MINUS1_, int* _POSITION0_, int* _POSITION1_, int* _POSITION2_, int* _POSITION3_, int* _POSITION23_,
                       int* StreamBinInit_d, int* StreamBinEnd_d
                       )
{
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const int kernel_offset = StreamIndex * NbinsPerStream;
  const int i = kernel_offset + idx;
  const bool ThreadCondition=
    StreamBinInit_d[StreamIndex] <= i &&
    i < StreamBinEnd_d[StreamIndex] &&
    i < Nbins;
                       
  if(ThreadCondition)
  {
    thread_memcpy(&particles__output__dev[*_POSITION3_ + *_POSITION2_ + *_POSITION1_ +*_POSITION0_ + pointer_minus1[i]],&particles__dev[init[i]+count3[i]+count2[i]+count1[i]+count0[i]],count_minus1[i]);
    thread_memcpy(&particles__output__dev[*_POSITION3_ + *_POSITION2_ + *_POSITION1_ + pointer0[i]],&particles__dev[init[i]+count3[i]+count2[i]+count1[i]],count0[i]);
    thread_memcpy(&particles__output__dev[*_POSITION3_ + *_POSITION2_ + pointer1[i]],&particles__dev[init[i]+count3[i]+count2[i]],count1[i]);
    thread_memcpy(&particles__output__dev[*_POSITION3_ + pointer2[i]],&particles__dev[init[i]+count3[i]],count2[i]);
    thread_memcpy(&particles__output__dev[0 +            pointer3[i]],&particles__dev[init[i]],count3[i]);
  }
//__syncthreads();
}

int main()
{
//THERE MUST BE Nbins >= nStreams !:
  assert(Nbins >= nStreams && "Nbins < nStreams");
//BEGIN OF KERNEL OFFSET CALCULATION://///////////////////////////////////////////
  int n=Nbins % nStreams;
  int* StreamBinInit_h=(int*)malloc(nStreams*sizeof(int));
  int* StreamBinEnd_h =(int*)malloc(nStreams*sizeof(int));
  StreamBinInit_h[0]=0;
  for(int stream=0; stream<nStreams-1; ++stream)
    StreamBinInit_h[stream+1]=
        StreamBinInit_h[stream]+
        ((stream < n)?NbinsPerStream+1:NbinsPerStream);
  for(int stream=0; stream<nStreams-1; ++stream)
    StreamBinEnd_h[stream]=StreamBinInit_h[stream+1];
  StreamBinEnd_h[nStreams-1]=Nbins;

  int* StreamBinInit_d, *StreamBinEnd_d;
  cudaMalloc(&StreamBinInit_d, nStreams*sizeof(int));
  cudaMalloc(&StreamBinEnd_d, nStreams*sizeof(int));
  cudaMemcpy(StreamBinInit_d, StreamBinInit_h, nStreams*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(StreamBinEnd_d,  StreamBinEnd_h,  nStreams*sizeof(int), cudaMemcpyHostToDevice);
//for(int i=0; i<nStreams; ++i)
//  std::cout<<StreamBinInit_h[i]<<"   "<<StreamBinEnd_h[i]<<std::endl;
//END OF KERNEL OFFSET CALCULATION//////////////////////////////////////////////////

  int* init;
  int* fin ;
  int* count01;
  int* count23;
  int* count_minus1;
  int* count0;
  int* count1;
  int* count2;
  int* count3;
  int* mini;
  int* ii0;//not for ir=-1, but for ir=0, because ir=0 should stand before ir=-1 in sorted array particles 
  int* ii1;
  int* ii3;
  int* ii23;
  int* POSITION3;
  int* POSITION2;
  int* POSITION1;
  int* POSITION0;
  int* POSITION_MINUS1;
  int* POSITION23;
  int* pointer_minus1;
  int* pointer0;
  int* pointer1;
  int* pointer2;
  int* pointer3;
  cudaMalloc(&init, Nbins*sizeof(int));
  cudaMalloc(&fin,  Nbins*sizeof(int));
  cudaMalloc(&count01, Nbins*sizeof(int));
  cudaMalloc(&count23, Nbins*sizeof(int));
  cudaMalloc(&count_minus1, Nbins*sizeof(int));
  cudaMalloc(&count0, Nbins*sizeof(int));
  cudaMalloc(&count1, Nbins*sizeof(int));
  cudaMalloc(&count2, Nbins*sizeof(int));
  cudaMalloc(&count3, Nbins*sizeof(int));
  cudaMalloc(&mini, Nbins*sizeof(int));
  cudaMalloc(&ii0, Nbins*sizeof(int));
  cudaMalloc(&ii1, Nbins*sizeof(int));
  cudaMalloc(&ii3, Nbins*sizeof(int));
  cudaMalloc(&ii23, Nbins*sizeof(int));
  cudaMalloc(&POSITION3, Nbins*sizeof(int));
  cudaMalloc(&POSITION2, Nbins*sizeof(int));
  cudaMalloc(&POSITION1, Nbins*sizeof(int));
  cudaMalloc(&POSITION0, Nbins*sizeof(int));
  cudaMalloc(&POSITION_MINUS1, Nbins*sizeof(int));
  cudaMalloc(&POSITION23, Nbins*sizeof(int));
  cudaMalloc(&pointer_minus1, Nbins*sizeof(int));
  cudaMalloc(&pointer0, Nbins*sizeof(int));
  cudaMalloc(&pointer1, Nbins*sizeof(int));
  cudaMalloc(&pointer2, Nbins*sizeof(int));
  cudaMalloc(&pointer3, Nbins*sizeof(int));

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
  P* particles;
  P* particles__dev;
  P* particles__output__dev;
  cudaMalloc(&particles__dev,  GL*sizeof(P));
  cudaMalloc(&particles__output__dev,  GL*sizeof(P));
  particles=new P[GL];
  std::vector<P> vec_seq(GL);
  srand(time(NULL));
  for(int i=0; i<GL; ++i)
  {
    particles[i].ir = rand()%5-1;
    vec_seq[i].ir=particles[i].ir;
  }
/*
  for(int i=0; i<GL; ++i)
    std::cout<<particles[i].ir<<" ";
  std::cout<<std::endl;
*/
  cudaMemcpy(particles__dev, particles, sizeof(P)*GL, cudaMemcpyHostToDevice);
  int* ind01;
  int* ind23;
  cudaMalloc((void**)&ind01, sizeof(int)*Nbins*M);
  cudaMalloc((void**)&ind23, sizeof(int)*Nbins*M);
  cudaMemset(ind01, 0, sizeof(int)*Nbins*M);
  cudaMemset(ind23, 0, sizeof(int)*Nbins*M);
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
  int* _POSITION_MINUS1_, * _POSITION0_, * _POSITION1_, * _POSITION2_, * _POSITION3_, * _POSITION23_;
  int PPOSITION_MINUS1, PPOSITION0, PPOSITION1, PPOSITION2, PPOSITION3, PPOSITION23;
  cudaMalloc(&_POSITION_MINUS1_,  sizeof(int));
  cudaMalloc(&_POSITION0_,  sizeof(int));
  cudaMalloc(&_POSITION1_,  sizeof(int));
  cudaMalloc(&_POSITION2_,  sizeof(int));
  cudaMalloc(&_POSITION3_,  sizeof(int));
  cudaMalloc(&_POSITION23_, sizeof(int));
  cudaMemset(_POSITION_MINUS1_,  0, sizeof(int));
  cudaMemset(_POSITION0_,  0, sizeof(int));
  cudaMemset(_POSITION1_,  0, sizeof(int));
  cudaMemset(_POSITION2_,  0, sizeof(int));
  cudaMemset(_POSITION3_,  0, sizeof(int));
  cudaMemset(_POSITION23_, 0, sizeof(int));
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/*
  initialize<<<(Nbins+NumberOfThreadsPerBlock-1)//NumberOfThreadsPerBlock,
                NumberOfThreadsPerBlock>>>(
                  ind01, ind23, Nbins,
                  init, fin,
                  count01, count23, count_minus1, count0, count1, count2, count3,
                  POSITION3, POSITION2, POSITION1, POSITION0, POSITION_MINUS1, POSITION23
               );
  cudaDeviceSynchronize();
*/

  auto t1=std::chrono::steady_clock::now();
  cudaStream_t streams[nStreams];
  for(int i=0; i<nStreams; ++i) cudaStreamCreate(&streams[i]);
  for(int i=0; i<nStreams; ++i)
  {
    kernel<<<NumberOfBlocks,
             NumberOfThreadsPerBlock,
             0,
             streams[i]>>>(i,
                           init, fin,
                           ind01, ind23,
                           particles__dev,
                           mini, ii0, ii1, ii3, ii23,
                           POSITION_MINUS1,POSITION0,POSITION1,POSITION2,POSITION3,POSITION23,
                           _POSITION_MINUS1_,_POSITION0_,_POSITION1_,_POSITION2_,_POSITION3_,_POSITION23_,
                           count01, count23, count_minus1, count0, count1, count2, count3,
                           pointer_minus1, pointer0, pointer1, pointer2, pointer3,
                           StreamBinInit_d, StreamBinEnd_d
                          );
////cudaStreamSynchronize(streams[i]);
  }

  calculate_offsets<<<(Nbins+NumberOfThreadsPerBlock-1)/NumberOfThreadsPerBlock,
                       NumberOfThreadsPerBlock>>>(
                          count_minus1, count0, count1, count2, count3,
                          pointer_minus1, pointer0, pointer1, pointer2, pointer3
                        );

  for(int i=0; i<nStreams; ++i)
  {
    memcpy<<<NumberOfBlocks,
             NumberOfThreadsPerBlock,
             0,
             streams[i]>>>(i,
                           particles__dev,
                           particles__output__dev,
                           init, fin,
                           count_minus1, count0, count1, count2, count3,
                           pointer_minus1, pointer0, pointer1, pointer2, pointer3,
                           _POSITION_MINUS1_,_POSITION0_,_POSITION1_,_POSITION2_,_POSITION3_,_POSITION23_,
                           StreamBinInit_d, StreamBinEnd_d
                          );
////cudaStreamSynchronize(streams[i]);
  }

  for(int i=0; i<nStreams; ++i) cudaStreamDestroy(streams[i]);

////////////////////////////////////////////////////////////////////////////

  cudaMemcpy(&PPOSITION_MINUS1, _POSITION_MINUS1_, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&PPOSITION0,  _POSITION0_, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&PPOSITION1,  _POSITION1_, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&PPOSITION2,  _POSITION2_, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&PPOSITION3,  _POSITION3_, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&PPOSITION23, _POSITION23_, sizeof(int), cudaMemcpyDeviceToHost);
//cudaDeviceSynchronize();

  auto t2=std::chrono::steady_clock::now();
  std::cout<<"time spent on CUDA Threads binwise sort on GPU="
           <<std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()<<" us"
           <<std::endl;

  P particles_check[GL];
  cudaMemcpy(particles_check, particles__output__dev, GL*sizeof(P), cudaMemcpyDeviceToHost);
//*
  std::cout<<"particles_check:"<<std::endl;
  int cnt3=0, cnt2=0, cnt1=0, cnt0=0, cntm1=0;
  int GLOBAL_COUNTER=0;
  for(int i=0; i<GL; ++i)
  {
  //std::cout<<particles_check[i].ir<<" ";
    if(3 == particles_check[i].ir) cnt3++;
    if(2 == particles_check[i].ir) cnt2++;
    if(1 == particles_check[i].ir) cnt1++;
    if(0 == particles_check[i].ir) cnt0++;
    if(-1 == particles_check[i].ir) cntm1++;
    GLOBAL_COUNTER++;
  }
  std::cout<<std::endl;
  std::cout<<cnt3<<" "<<cnt2<<" "<<cnt1<<" "<<cnt0<<" "<<cntm1<<std::endl;
  std::cout<<PPOSITION3<<" "<<PPOSITION2<<" "<<PPOSITION1<<" "<<PPOSITION0<<" "<<PPOSITION_MINUS1<<std::endl;
  std::cout<<"GLOBAL_COUNTER="<<GLOBAL_COUNTER<<std::endl;
//*/

  std::cout<<"Nbins="<<Nbins<<" N="<<N
           <<" GL="<<GL<<" M="<<M
           <<" GL1="<<GL1
           <<" nStreams="<<nStreams
           <<std::endl;

  std::cout<<"Nbins="<<Nbins
           <<" nStreams="<<nStreams
           <<" NbinsPerStream="<<NbinsPerStream
           <<std::endl;

//SEQUENTIAL SORT CHECK:
  t1=std::chrono::steady_clock::now();
  std::sort(vec_seq.begin(), vec_seq.end(), std::greater<P>());
  t2=std::chrono::steady_clock::now();
  std::cout<<"time spent on std::::sort="
           <<std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()<<" us"
           <<std::endl;
  bool MATCH=true;
  for(int i=0; i<GL; ++i)
  {
    if( particles_check[i].ir != vec_seq[i].ir )
        MATCH = false;
  }

  if(MATCH)
    std::cout<<"+ THE RESULTS MATCH!!!"<<std::endl;
  else
    std::cout<<"x THE RESULTS DO NOT MATCH!!!"<<std::endl;

  return 0;
}
