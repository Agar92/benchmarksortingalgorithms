#include <iostream>
#include <cmath>
#include <chrono>
#include <cstring>
#include <vector>
#include <algorithm>

#include "globals.h"
#include "particle.h"
#include "extra.h"

using namespace std;

inline const int N=20'000'000;

//1st buffer (input)
P particles[N];

//initialization
void initialize()
{
  for(int i=0; i<N; ++i)
  {
    particles[i].ir=(rand()%5-1);
    particles[i].id=i;
    //we let r and p fields of the structure P be initialized
    //using default P constructor:
    //r(0.0,0.0,0.0)
    //and
    //p(0.0,0.0,0.0)
  }
}

void TPT3_sort_CPU(P * particles)
{
  unsigned int dL=LIFE/Nbin;
  unsigned int DL=dL+1;
  
  unsigned int const n = Nbin - LIFE % Nbin;
    
  POSITION_MINUS1=POSITION0=POSITION1=POSITION2=POSITION3=POSITION23=0;
#ifdef OPENACC
#pragma acc parallel loop gang vector copy(GL1,dL,DL,n,count01,count23,count0,count1,count2,count3,init,fin)
#else
#pragma omp parallel for simd
#pragma distribute_point
#endif
    for(int b=0; b<Nbin; ++b)    
    {
      count01[b]=GL1;
      count23[b]=0;
      count0[b]=GL1;
      count1[b]=0;
      count2[b]=GL1;
      count3[b]=0;
      if(b<n)
      {
        init[b]=b*dL;
        fin[b]=(b+1)*dL;
      }
      else if(b==n)
      {
        init[b]=n*dL;
        fin[b]=n*dL+DL;
      }
      else if(b>n)
      {
        init[b]=n*dL+DL*(b-n);
        fin[b]=n*dL+DL*(b-n+1);
      }
      //cout<<"b="<<b<<" init["<<b<<"]="<<init[b]<<" fin["<<b<<"]="<<fin[b]<<endl;
    }

#ifdef OPENACC
#pragma acc parallel loop gang vector copy(count01,count23,init,fin) present(particles,ind23)
#endif
#ifndef OPENACC
#pragma omp parallel for
#endif
    for(int b=0; b<Nbin; ++b)
    {
#ifdef OPENACC
#else
#endif
      for(int i=init[b]; i<fin[b]; ++i)
      {
        if(particles[i].ir<2) ind23[b][count01[b]--]=i;
        else                  ind23[b][count23[b]++]=i;
      }
    }
    
#ifdef OPENACC
#pragma acc parallel loop gang copy(count0,count1,count2,count3,count01,count23,mini,ii23,init,fin) present(particles,ind01,ind23)
#else
#pragma omp parallel for
#pragma distribute_point
#endif
    for(int b=0; b<Nbin; ++b)    
    {
      ii23[b]=count23[b]-1;
      mini[b]=GL1-count01[b];
      if(count23[b]<mini[b]) mini[b]=count23[b];
      int js=0;
#ifdef OPENACC
#pragma acc loop vector reduction(+:js)
#else
#pragma omp simd reduction(+:js)
#endif
      for(int j=0; j<mini[b]; ++j)
        if (ind23[b][ii23[b] - j] > ind23[b][GL1 - j]) ++js;
#ifdef OPENACC
#pragma acc loop vector
#else
#pragma omp simd
#endif
      for(int j=0; j<js; ++j) std::swap(particles[ind23[b][ii23[b]-j]],particles[ind23[b][GL1-j]]);

      for(int i=init[b]; i<fin[b]; ++i)
      {
        if     (particles[i].ir==-1 ||
                particles[i].ir==0) ind01[b][count0[b]--]=i;
        else if(particles[i].ir==1) ind01[b][count1[b]++]=i;
        else if(particles[i].ir==2) ind23[b][count2[b]--]=i;
        else                        ind23[b][count3[b]++]=i;
      }
    }
#ifdef OPENACC
#pragma acc parallel loop gang copy(count0,count1,count2,count3,mini,ii1,ii3,GL1) present(particles,ind01,ind23)
#else
#pragma omp parallel for
#pragma distribute_point
#endif
    for(int b=0; b<Nbin; ++b)    
    {
      ii1[b]=count1[b]-1;
      mini[b]=GL1-count0[b];
      if(count1[b]<mini[b]) mini[b]=count1[b];
      int js=0;
#ifdef OPENACC
#pragma acc loop vector reduction(+:js)
#else
#pragma omp simd reduction(+:js)
#endif
      for(int j=0; j<mini[b]; ++j)
        if (ind01[b][ii1[b] - j] > ind01[b][GL1 - j]) ++js;
#ifdef OPENACC
#pragma acc loop vector
#else
#pragma omp simd
#endif
      for(int j=0; j<js; ++j) std::swap(particles[ind01[b][ii1[b]-j]],particles[ind01[b][GL1-j]]);
      ii3[b]=count3[b]-1;
      mini[b]=GL1-count2[b];
      if(count3[b]<mini[b]) mini[b]=count3[b];
      js=0;
#ifdef OPENACC
#pragma acc loop vector reduction(+:js)
#else
#pragma omp simd reduction(+:js)
#endif
      for(int j=0; j<mini[b]; ++j)
        if (ind23[b][ii3[b] - j] > ind23[b][GL1 - j]) ++js;
#ifdef OPENACC
#pragma acc loop vector
#else
#pragma omp simd
#endif
      for(int j=0; j<js; ++j) std::swap(particles[ind23[b][ii3[b]-j]],particles[ind23[b][GL1-j]]);
//#if defined(NF) && defined(TIMELIMIT)
      //sort particles with ir=-1 and ir=0
      //ir=-1 should be in front of ir=0:
      //
      //here we save the value of count0[b] to use it below in the lines 1964-1971:
      const int save_value_of_count0_b=count0[b];
      int N0=GL1-count0[b];
      count0[b]=0;
      count_minus1[b]=GL1;
      const int c3=count3[b];
      const int c2=GL1-count2[b];
      const int c1=count1[b];
      for(int i=init[b]+c3+c2+c1; i<fin[b]; ++i)
      {
        if     (particles[i].ir== 0) ind01[b][count0[b]++]=i;
        else if(particles[i].ir==-1) ind01[b][count_minus1[b]--]=i;
      }
      const int c0=count0[b];
      count_minus1[b]=GL1-count_minus1[b];
      ii0[b]=count0[b]-1;
      mini[b]=count_minus1[b];//GL1-count_minus1[b];
      if(count0[b]<mini[b]) mini[b]=count0[b];
      js=0;
#ifdef OPENACC
#pragma acc loop vector reduction(+:js)
#else
#pragma omp simd reduction(+:js)
#endif
      for(int j=0; j<mini[b]; ++j)
        if (ind01[b][ii0[b] - j] > ind01[b][GL1 - j]) ++js;
#ifdef OPENACC
#pragma acc loop vector
#else
#pragma omp simd
#endif
      for(int j=0; j<js; ++j)
        std::swap(particles[ind01[b][ii0[b]-j]],particles[ind01[b][GL1-j]]);
      //here we recover the original value of count0[b] to use it below in the lines 1964-1971:
      count0[b]=save_value_of_count0_b;// - count_minus1[b];
//#endif//#if defined(NF) && defined(TIMELIMIT)      
    }

  // Reorder the pointers limits
#ifdef OPENACC
#pragma acc parallel loop gang vector reduction(+:POSITION_MINUS1,POSITION0,POSITION1,POSITION2,POSITION3,POSITION23)
#else
#pragma omp parallel for simd reduction(+:POSITION_MINUS1,POSITION0,POSITION1,POSITION2,POSITION3,POSITION23)
#endif
  for(int b=0; b<Nbin; ++b)
  {
    count0[b]=GL1-count0[b] - count_minus1[b];
    count2[b]=GL1-count2[b];
    POSITION_MINUS1+=count_minus1[b];
    POSITION0+=count0[b];
    POSITION1+=count1[b];
    POSITION2+=count2[b];
    POSITION3+=count3[b];
    POSITION23+=count23[b];
  }

  std::cout<<"POSITION_MINUS1="<<POSITION_MINUS1
           <<" POSITION0="<<POSITION0
           <<" POSITION1="<<POSITION1
           <<" POSITION2="<<POSITION2
           <<" POSITION3="<<POSITION3
           <<std::endl;
  
//§Ù§Õ§Ö§ã§î §Õ§à§Ý§Ø§ß§à §Ú§Õ§ä§Ú §ã§Ý§Ú§ñ§ß§Ú§Ö §Þ§Ú§ß§Ú §ñ§ë§Ú§Ü§à§Ó §Ó §ñ§ë§Ú§Ü§Ú §Õ§Ý§ñ 0, 1, 2, 3, §å§Õ§Ñ§Ý§Ö§ß§Ú§Ö 0, §Ú §á§Ö§â§Ö§Ü§Ý§Ñ§Õ§í§Ó§Ñ§ß§Ú§Ö 3, 2, 1 §Ó §Ú§ã§ç§à§Õ§ß§í§Û §ñ§ë§Ú§Ü
  pointer_minus1[0]=pointer0[0]=pointer1[0]=pointer2[0]=pointer3[0]=0;
#ifdef OPENACC
#pragma acc serial loop /*num_gangs(1) vector_length(1)*/ copy(pointer1,pointer2,pointer3)
#endif
  for(int b=0; b<Nbin-1; ++b)
  {
    pointer_minus1[b+1]=pointer_minus1[b]+count_minus1[b];
    pointer0[b+1]=pointer0[b]+count0[b];
    pointer1[b+1]=pointer1[b]+count1[b];
    pointer2[b+1]=pointer2[b]+count2[b];
    pointer3[b+1]=pointer3[b]+count3[b];
  }

  //DO NOT parallelize or vectorize - undefined behavior
  for(int b=0; b<Nbin; ++b)
  {
    memcpy(&arr_minus1[pointer_minus1[b]],&particles[init[b]+count3[b]+count2[b]+count1[b]+count0[b]],count_minus1[b]*sizep);
    memcpy(&arr0[pointer0[b]],&particles[init[b]+count3[b]+count2[b]+count1[b]],count0[b]*sizep);
    memcpy(&arr1[pointer1[b]],&particles[init[b]+count3[b]+count2[b]],count1[b]*sizep);
    memcpy(&arr2[pointer2[b]],&particles[init[b]+count3[b]],count2[b]*sizep);
    memcpy(&arr3[pointer3[b]],&particles[init[b]],count3[b]*sizep);
  }
  /*
  cout<<"arr3:"<<endl;
  for(int i=0; i<POSITION3; ++i)
    cout<<arr3[i].ir<<" ";
  cout<<endl;
  */
  memcpy(&particles[0],&arr3[0],POSITION3*sizep);
  memcpy(&particles[POSITION3],&arr2[0],POSITION2*sizep);
  memcpy(&particles[POSITION23],&arr1[0],POSITION1*sizep);
  memcpy(&particles[POSITION23+POSITION1],&arr0[0],POSITION0*sizep);
  memcpy(&particles[POSITION23+POSITION1+POSITION0],&arr_minus1[0],POSITION_MINUS1*sizep);
 
}//End of Compressor




int main()
{
  //initialization:
  initialize();
  std::vector<P> vec_seq(N);
  std::memcpy(vec_seq.data(), particles, sizeof(P)*N);
  //
  const auto begin=std::chrono::steady_clock::now();
  TPT3_sort_CPU(particles);
  const auto end=std::chrono::steady_clock::now();
  const auto time_elapsed=
    std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
  cout<<"T="<<"\033[31m"<<time_elapsed<<"\033[34m"<<" ms"<<endl;
  cout<<"\033[37m"<<"Nbin="<<Nbin<<endl;
  //
  /*
  for(int i=0; i<N; ++i)
    cout<<particles[i].ir<<" ";
  cout<<endl;
  */
  //
  //SEQUENTIAL SORT CHECK:
  auto t1=std::chrono::steady_clock::now();
  std::sort(vec_seq.begin(), vec_seq.end(), [](const P & l, const P & r)
                                            {
                                              return l.ir>r.ir;
                                            });
  /*
  cout<<endl<<endl<<endl;
  for(int i=0; i<N; ++i)
    cout<<vec_seq[i].ir<<" ";
  cout<<endl;
  */
  auto t2=std::chrono::steady_clock::now();
  std::cout<<"time spent on std::sort="
           <<std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()<<" ms"
           <<std::endl;
  bool MATCH=true;
  for(int i=0; i<N; ++i)
  {
    if( particles[i].ir != vec_seq[i].ir )
      MATCH = false;
  }

  if(MATCH)
    std::cout<<"+ THE RESULTS MATCH!!!"<<std::endl;
  else
    std::cout<<"x THE RESULTS DO NOT MATCH!!!"<<std::endl;
  
  return 0;
}
