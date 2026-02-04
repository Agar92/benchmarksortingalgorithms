#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

constexpr int N=140'000;
//constexpr int Nbin=4;
constexpr int Nbin=1024;
constexpr int GL=N;
constexpr int M=GL/Nbin+1;//10'000;
constexpr int GL1 = M - 1;

const int ThreadsPerBlock=1024;

struct P{
  int ir;
  int id;
  double3 r;
  double3 p;
  P():ir(-1),id(-1),r{},p{}{}
  P(int _ir, int _id):ir(_ir),id(_id),r{},p{}{}
};

P* particles;
P arr_minus1[GL];
P arr0[GL];
P arr1[GL];
P arr2[GL];
P arr3[GL];

int sizep=sizeof(P);

template <typename T>
__device__ void swap1(T& a, T& b){T c(a); a=b; b=c;}

template <typename T>
__device__ void swap2(T a, T b){T c(a); a=b; b=c;}


template <typename T>
__device__ void thread_memcpy(T* dst, const T* src, int N)
{
  #pragma unroll
  for(int j=0; j<N; ++j)
    dst[j]=src[j];
}

__global__ void fun(int* ind01, int* ind23, int N,
                    P* particles__dev,
                    P* arr_minus1__dev, P* arr0__dev, P* arr1__dev, P* arr2__dev, P* arr3__dev,
                    int* _POSITION_MINUS1_, int* _POSITION0_, int* _POSITION1_, int* _POSITION2_, int* _POSITION3_, int* _POSITION23_,
                    int* init,
                    int* fin ,
                    int* count01,     
                    int* count23,
                    int* count_minus1,
                    int* count0,
                    int* count1,
                    int* count2,
                    int* count3,
                    int* mini,
                    int* ii0,//not for ir=-1, but for ir=0, because ir=0 should stand before ir=-1 in sorted array particles 
                    int* ii1,
                    int* ii3,
                    int* ii23,
                    int* POSITION3,
                    int* POSITION2,
                    int* POSITION1,
                    int* POSITION0,
                    int* POSITION_MINUS1,
                    int* POSITION23,
                    int* pointer_minus1,
                    int* pointer0,
                    int* pointer1,
                    int* pointer2,
                    int* pointer3
                    )
{
  __const__ int dL=N/Nbin;
  __const__ int DL=dL+1;
  __const__ int n=Nbin-N%Nbin;

  int i = blockIdx.x*blockDim.x+threadIdx.x;
//1) Find borders of bins:
  if(i < Nbin)
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
    if(i == n)
    {
      init[i]=dL*n;
      fin [i]=n*dL+DL;
    }
    if(i>n)
    {
      init[i]=n*dL+DL*(i-n);
      fin [i]=n*dL+DL*(i-n+1);
    }
////printf("#%d b=%d dL=%d DL=%d n=%d Nbin=%d init[%d]=%d   fin[%d]=%d\n",
////        i, b, dL, DL, n, Nbin, i, init[i], i, fin[i]);
  }
  ///printf("STEP #1 bin #%d\n", i);
  __syncthreads();

/*
  if(i < Nbin)
  {
    if(0 == i)
    {
      printf("bin #%d particles:\n", i);
      for(int j=init[i]; j<fin[i]; ++j)
        printf("%d ", particles__dev[j].ir);
      printf("\n");
    }
    if(1 == i)
    {
      printf("bin #%d particles:\n", i);
      for(int j=init[i]; j<fin[i]; ++j)
        printf("%d ", particles__dev[j].ir);
      printf("\n");
    }
    if(2 == i)
    {
      printf("bin #%d particles:\n", i);
      for(int j=init[i]; j<fin[i]; ++j)
        printf("%d ", particles__dev[j].ir);
      printf("\n");
    }
    if(3 == i)
    {
      printf("bin #%d particles:\n", i);
      for(int j=init[i]; j<fin[i]; ++j)
        printf("%d ", particles__dev[j].ir);
      printf("\n");
    }
  }
*/

  ///printf("STEP #2 bin #%d\n", i);
//2) Find counts of ir=-1,0,1 and ir=2,3:
  if(i < Nbin)
  {
    ///printf("III\n");
    for(int j=init[i]; j<fin[i]; ++j)
    {
      ///printf("J1J %d\n", particles__dev[j].ir);
      if(particles__dev[j].ir<2)
      {
        ///printf("A\n");
        ind23[i*M+count01[i]]=j;
        count01[i] = count01[i] - 1;
      }
      else
      {
        ///printf("B\n");
        ind23[i*M+count23[i]]=j;
        ///printf("B2\n");
        count23[i] = count23[i] + 1;
      }
      ///printf("J2J %d\n", (particles__dev)[j].ir);
    }
  }
  ///printf("STEP #3 bin #%d\n", i);
  __syncthreads();
  ///printf("STEP #4 bin #%d\n", i);
//3) Divide ir=2,3 from ir=-1,0,1.
//   Find counts of ir=-1,0 and ir=1 and ir=2 and ir=3:
//2 3 3 2 3 -1 0 1 1
  if(i<Nbin)
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
  __syncthreads();
  ///printf("STEP #5 bin #%d\n", i);
//4) Divide ir=1 from ir=-1,0.
//   Divide ir=3 from ir=2.
//3 3 3 2 2 1 1 -1 0
  //#pragma omp parallel for
  if(i<Nbin)    
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
  __syncthreads();
  ///printf("STEP #6 bin #%d\n", i);
//5) Divide ir=0 from ir=-1:
//3 3 3 2 2 1 1 0 -1  
  //#pragma omp parallel for
  if(i<Nbin)   
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
  ///printf("STEP #7 bin #%d\n", i);
  __syncthreads();
  //#pragma omp parallel for reduction(+:POSITION_MINUS1,POSITION0,POSITION1,POSITION2,POSITION3,POSITION23)
  if(i<Nbin)
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
  __syncthreads();
  //*
  if(i<Nbin)
  {
    atomicAdd(_POSITION_MINUS1_, POSITION_MINUS1[i]);
    atomicAdd(_POSITION0_,       POSITION0[i]);
    atomicAdd(_POSITION1_,       POSITION1[i]);
    atomicAdd(_POSITION2_,       POSITION2[i]);
    atomicAdd(_POSITION3_,       POSITION3[i]);
    atomicAdd(_POSITION23_,      POSITION23[i]);
    }
  //*/
  /*
  if(0 == i)
  {
    for(int j=0; j<Nbin; ++j)
    {
      *_POSITION_MINUS1_ = *_POSITION_MINUS1_ + POSITION_MINUS1[j];
      *_POSITION0_       = *_POSITION0_ + POSITION0[j];
      *_POSITION1_       = *_POSITION1_ + POSITION1[j];
      *_POSITION2_       = *_POSITION2_ + POSITION2[j];
      *_POSITION3_       = *_POSITION3_ + POSITION3[j];
      *_POSITION23_      = *_POSITION23_ + POSITION23[j];
    }
  }
  */

/*
  /////printf("PP #%d:   %d   %d   %d\n", i, _POSITION0_, _POSITION1_, _POSITION2_, _POSITION3_);
  ///printf("STEP #8 bin #%d\n", i);
  if(i < Nbin)
  {
    if(0 == i)
    {
      printf("bin #%d:   %d   %d   %d   %d   %d Total=%d\n\n\n",
              i, POSITION_MINUS1[i], POSITION0[i], POSITION1[i], POSITION2[i], POSITION3[i],
              POSITION_MINUS1[i]+POSITION0[i]+POSITION1[i]+POSITION2[i]+POSITION3[i]);
      printf("bin #%d particles:\n", i);
      for(int j=init[i]; j<fin[i]; ++j)
        printf("%d|%d ", j, particles__dev[j].ir);
      printf("\n");
    }
    if(1 == i)
    {
      printf("bin #%d:   %d   %d   %d   %d   %d Total=%d\n\n\n",
              i, POSITION_MINUS1[i], POSITION0[i], POSITION1[i], POSITION2[i], POSITION3[i],
              POSITION_MINUS1[i]+POSITION0[i]+POSITION1[i]+POSITION2[i]+POSITION3[i]);
      printf("bin #%d particles:\n", i);
      for(int j=init[i]; j<fin[i]; ++j)
        printf("%d|%d ", j, particles__dev[j].ir);
      printf("\n");
    }
    if(2 == i)
    {
      printf("bin #%d:   %d   %d   %d   %d   %d Total=%d\n\n\n",
              i, POSITION_MINUS1[i], POSITION0[i], POSITION1[i], POSITION2[i], POSITION3[i],
              POSITION_MINUS1[i]+POSITION0[i]+POSITION1[i]+POSITION2[i]+POSITION3[i]);
      printf("bin #%d particles:\n", i);
      for(int j=init[i]; j<fin[i]; ++j)
        printf("%d|%d ", j, particles__dev[j].ir);
      printf("\n");
    }
    if(3 == i)
    {
      printf("bin #%d:   %d   %d   %d   %d   %d Total=%d\n\n\n",
              i, POSITION_MINUS1[i], POSITION0[i], POSITION1[i], POSITION2[i], POSITION3[i],
              POSITION_MINUS1[i]+POSITION0[i]+POSITION1[i]+POSITION2[i]+POSITION3[i]);
      printf("bin #%d particles:\n", i);
      for(int j=init[i]; j<fin[i]; ++j)
        printf("%d|%d ", j, particles__dev[j].ir);
      printf("\n");
    }
  }
*/

  if(0 == i)
  {
    pointer0[0]=pointer1[0]=pointer2[0]=pointer3[0]=0;
    pointer_minus1[0]=0;
    for(int b=0; b<Nbin-1; ++b)
    {
      pointer_minus1[b+1]=pointer_minus1[b]+count_minus1[b];
      pointer0[b+1]=pointer0[b]+count0[b];
      pointer1[b+1]=pointer1[b]+count1[b];
      pointer2[b+1]=pointer2[b]+count2[b];
      pointer3[b+1]=pointer3[b]+count3[b];
    }
  }
  ///printf("STEP #10\n");
  __syncthreads();
  ///printf("STEP #11\n");
  if(i<Nbin)
  {
    ///printf("STEP #111\n");
    thread_memcpy(&arr_minus1__dev[pointer_minus1[i]],&particles__dev[init[i]+count3[i]+count2[i]+count1[i]+count0[i]],count_minus1[i]);
    ///printf("STEP #222\n");
    thread_memcpy(&arr0__dev[pointer0[i]],&particles__dev[init[i]+count3[i]+count2[i]+count1[i]],count0[i]);
    thread_memcpy(&arr1__dev[pointer1[i]],&particles__dev[init[i]+count3[i]+count2[i]],count1[i]);
    thread_memcpy(&arr2__dev[pointer2[i]],&particles__dev[init[i]+count3[i]],count2[i]);
    thread_memcpy(&arr3__dev[pointer3[i]],&particles__dev[init[i]],count3[i]);
  }
  ///printf("STEP #12\n");
/*
  if(i < Nbin)
  {
    if(0 == i)
    {
      printf("bin #%d:   array arr_minus1 size=%d\n\n\n",
              i, pointer_minus1[Nbin-1]);
      for(int j=pointer_minus1[0]; j<pointer_minus1[Nbin-1]; ++j)
        printf("%d ", arr_minus1__dev[j].ir);
      printf("\n");
    }
    ///printf("STEP #13\n");
  }
*/
  __syncthreads();
}

__device__ double d_D[N];

int main()
{
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
  cudaMalloc(&init, Nbin*sizeof(int));
  cudaMalloc(&fin,  Nbin*sizeof(int));
  cudaMalloc(&count01, Nbin*sizeof(int));
  cudaMalloc(&count23, Nbin*sizeof(int));
  cudaMalloc(&count_minus1, Nbin*sizeof(int));
  cudaMalloc(&count0, Nbin*sizeof(int));
  cudaMalloc(&count1, Nbin*sizeof(int));
  cudaMalloc(&count2, Nbin*sizeof(int));
  cudaMalloc(&count3, Nbin*sizeof(int));
  cudaMalloc(&mini, Nbin*sizeof(int));
  cudaMalloc(&ii0, Nbin*sizeof(int));
  cudaMalloc(&ii1, Nbin*sizeof(int));
  cudaMalloc(&ii3, Nbin*sizeof(int));
  cudaMalloc(&ii23, Nbin*sizeof(int));
  cudaMalloc(&POSITION3, Nbin*sizeof(int));
  cudaMalloc(&POSITION2, Nbin*sizeof(int));
  cudaMalloc(&POSITION1, Nbin*sizeof(int));
  cudaMalloc(&POSITION0, Nbin*sizeof(int));
  cudaMalloc(&POSITION_MINUS1, Nbin*sizeof(int));
  cudaMalloc(&POSITION23, Nbin*sizeof(int));
  cudaMalloc(&pointer_minus1, Nbin*sizeof(int));
  cudaMalloc(&pointer0, Nbin*sizeof(int));
  cudaMalloc(&pointer1, Nbin*sizeof(int));
  cudaMalloc(&pointer2, Nbin*sizeof(int));
  cudaMalloc(&pointer3, Nbin*sizeof(int));
  

  P* particles__dev;
  P* arr_minus1__dev;
  P* arr0__dev;
  P* arr1__dev;
  P* arr2__dev;
  P* arr3__dev;
  cudaMalloc(&particles__dev,    GL*sizeof(P));
  cudaMalloc(&arr_minus1__dev, GL*sizeof(P));
  cudaMalloc(&arr0__dev,       GL*sizeof(P));
  cudaMalloc(&arr1__dev,       GL*sizeof(P));
  cudaMalloc(&arr2__dev,       GL*sizeof(P));
  cudaMalloc(&arr3__dev,       GL*sizeof(P));
  particles=new P[GL];
  srand(time(NULL));
  for(int i=0; i<GL; ++i)
  {
    particles[i].ir = rand()%5-1;
  }
/*
  for(int i=0; i<GL; ++i)
    cout<<particles[i].ir<<" ";
  cout<<endl;
*/  
  //*/
  cudaMemcpy(particles__dev, particles, sizeof(P)*GL, cudaMemcpyHostToDevice);
  //
  int* ind01;
  int* ind23;
  cudaMalloc((void**)&ind01, sizeof(int)*Nbin*M);
  cudaMalloc((void**)&ind23, sizeof(int)*Nbin*M);
  cudaMemset(ind01, 0, sizeof(int)*Nbin*M);
  cudaMemset(ind23, 0, sizeof(int)*Nbin*M);
  //
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
  //
  auto t1=std::chrono::steady_clock::now();
  fun<<<(N+ThreadsPerBlock-1)/ThreadsPerBlock,ThreadsPerBlock>>>(ind01, ind23, N,
               particles__dev,
               arr_minus1__dev, arr0__dev, arr1__dev, arr2__dev, arr3__dev,
               _POSITION_MINUS1_,_POSITION0_,_POSITION1_,_POSITION2_,_POSITION3_,_POSITION23_,
               init,
               fin ,
               count01,     
               count23,
               count_minus1,
               count0,
               count1,
               count2,
               count3,
               mini,
               ii0,//not for ir=-1, but for ir=0, because ir=0 should stand before ir=-1 in sorted array particles 
               ii1,
               ii3,
               ii23,
               POSITION3,
               POSITION2,
               POSITION1,
               POSITION0,
               POSITION_MINUS1,
               POSITION23,
               pointer_minus1,
               pointer0,
               pointer1,
               pointer2,
               pointer3
               );
  cudaDeviceSynchronize();
  auto t2=std::chrono::steady_clock::now();
  cout<<"T="<<std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()<<" us"<<endl;
  //*
  cudaMemcpy(&PPOSITION_MINUS1, _POSITION_MINUS1_, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&PPOSITION0,  _POSITION0_, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&PPOSITION1,  _POSITION1_, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&PPOSITION2,  _POSITION2_, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&PPOSITION3,  _POSITION3_, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&PPOSITION23, _POSITION23_, sizeof(int), cudaMemcpyDeviceToHost);
  //§ã§Ý§Ú§ñ§ß§Ú§Ö §ñ§ë§Ú§Ü§à§Ó §Õ§Ý§ñ 1, 2, 3 §Ó §Þ§Ñ§ã§ã§Ú§Ó particles
  cudaMemcpy(&particles__dev[0],&arr3__dev[0],PPOSITION3*sizep, cudaMemcpyDeviceToDevice);
  cudaMemcpy(&particles__dev[PPOSITION3],&arr2__dev[0],PPOSITION2*sizep, cudaMemcpyDeviceToDevice);
  cudaMemcpy(&particles__dev[PPOSITION23],&arr1__dev[0],PPOSITION1*sizep, cudaMemcpyDeviceToDevice);
  cudaMemcpy(&particles__dev[PPOSITION23+PPOSITION1],&arr0__dev[0],PPOSITION0*sizep, cudaMemcpyDeviceToDevice);
  cudaMemcpy(&particles__dev[PPOSITION23+PPOSITION1+PPOSITION0],&arr_minus1__dev[0], PPOSITION_MINUS1*sizep, cudaMemcpyDeviceToDevice);

  P particles_check[GL];
  
  //cudaMemcpy(, particles__dev, GL*sizeof(P), cudaMemcpyDeviceToHost);
  
  cudaMemcpy(particles_check, particles__dev, GL*sizeof(P), cudaMemcpyDeviceToHost);
//*
  cout<<"particles_check:"<<endl;
  int cnt3=0, cnt2=0, cnt1=0, cnt0=0, cntm1=0;
  int GLOBAL_COUNTER=0;
  for(int i=0; i<GL; ++i)
  {
  //cout<<particles_check[i].ir<<" ";
    if(3 == particles_check[i].ir) cnt3++;
    if(2 == particles_check[i].ir) cnt2++;
    if(1 == particles_check[i].ir) cnt1++;
    if(0 == particles_check[i].ir) cnt0++;
    if(-1 == particles_check[i].ir) cntm1++;
    GLOBAL_COUNTER++;
  }
  cout<<endl;
  cout<<cnt3<<" "<<cnt2<<" "<<cnt1<<" "<<cnt0<<" "<<cntm1<<endl;
  cout<<PPOSITION3<<" "<<PPOSITION2<<" "<<PPOSITION1<<" "<<PPOSITION0<<" "<<PPOSITION_MINUS1<<endl;
  cout<<"GLOBAL_COUNTER="<<GLOBAL_COUNTER<<endl;
//*/

//*  
  cnt3=cnt2=cnt1=cnt0=cntm1=0;
  for(int i=0; i<GL; ++i)
  {
  //cout<<particles[i].ir<<" ";
    if(3 == particles[i].ir) cnt3++;
    if(2 == particles[i].ir) cnt2++;
    if(1 == particles[i].ir) cnt1++;
    if(0 == particles[i].ir) cnt0++;
    if(-1 == particles[i].ir) cntm1++;
  }
  cout<<endl;
  cout<<cnt3<<" "<<cnt2<<" "<<cnt1<<" "<<cnt0<<" "<<cntm1<<endl;
//*/  
  //
  //
  delete [] particles;
  return 0;
}
