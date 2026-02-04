#pragma once

#include "globals.h"
#include "particle.h"

//
//The sorting algorithm how it is implemented in TPT3
//


//
//
//
int ind01[Nbin][BLt] __attribute__((aligned(64)));
int ind23[Nbin][BLt] __attribute__((aligned(64)));
int ARR_MINUS1_COUNTER=0;
P arr_minus1[GL] __attribute__((aligned(64)));
P arr0[GL] __attribute__((aligned(64)));
P arr1[GL] __attribute__((aligned(64)));
P arr2[GL] __attribute__((aligned(64)));
P arr3[GL] __attribute__((aligned(64)));
//
unsigned int POSITION3;
unsigned int POSITION2;
unsigned int POSITION1;
unsigned int POSITION0;
unsigned int POSITION_MINUS1;
unsigned int POSITION23;
unsigned int sizep=sizeof(P);
//  
int mini[Nbin];
int count01[Nbin];
int count23[Nbin];
int count_minus1[Nbin];
int count0[Nbin];
int count1[Nbin];
int count2[Nbin];
int pos2[Nbin];
int count3[Nbin];
int ii0[Nbin];//not for ir=-1, but for ir=0, because ir=0 should stand before ir=-1 in sorted array particles 
int ii1[Nbin];
int ii3[Nbin];
int ii23[Nbin];
//
int init[Nbin];
int fin[Nbin];
//
int pointer_minus1[Nbin];
int pointer0[Nbin];
int pointer1[Nbin];
int pointer2[Nbin];
int pointer3[Nbin];
//
//
//


//TPT3 sort:
template <size_t Nt>
void TPT3_sort(P (&particles)[Nt])
{
//1) Find borders of bins:
  const int dL=LIFE/Nbin;
  const int DL=dL+1;
  const int n=Nbin-LIFE%Nbin;
  POSITION_MINUS1=POSITION0=POSITION1=POSITION2=POSITION3=POSITION23=0;
#pragma omp parallel for
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
  }
//2) Find counts of ir=-1,0,1 and ir=2,3:
#pragma omp parallel for
  for(int b=0; b<Nbin; ++b)
  {
    for(int i=init[b]; i<fin[b]; ++i)
    {
      if(particles[i].ir<2) ind23[b][count01[b]--]=i;
      else                  ind23[b][count23[b]++]=i;
    }
  }
//3) Divide ir=2,3 from ir=-1,0,1.
//   Find counts of ir=-1,0 and ir=1 and ir=2 and ir=3:
//2 3 3 2 3 -1 0 1 1
#pragma omp parallel for
  for(int b=0; b<Nbin; ++b)    
  {
    ii23[b]=count23[b]-1;
    mini[b]=GL1-count01[b];
    if(count23[b]<mini[b]) mini[b]=count23[b];
    int js=0;
#pragma omp simd reduction(+:js)
    for(int j=0; j<mini[b]; ++j)
      if (ind23[b][ii23[b] - j] > ind23[b][GL1 - j]) ++js;
#pragma omp simd
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
//4) Divide ir=1 from ir=-1,0.
//   Divide ir=3 from ir=2.
//3 3 3 2 2 1 1 -1 0
#pragma omp parallel for
  for(int b=0; b<Nbin; ++b)    
  {
    ii1[b]=count1[b]-1;
    mini[b]=GL1-count0[b];
    if(count1[b]<mini[b]) mini[b]=count1[b];
    int js=0;
#pragma omp simd reduction(+:js)
    for(int j=0; j<mini[b]; ++j)
      if (ind01[b][ii1[b] - j] > ind01[b][GL1 - j]) ++js;
#pragma omp simd
    for(int j=0; j<js; ++j) std::swap(particles[ind01[b][ii1[b]-j]],particles[ind01[b][GL1-j]]);
    ii3[b]=count3[b]-1;
    mini[b]=GL1-count2[b];
    if(count3[b]<mini[b]) mini[b]=count3[b];
    js=0;
#pragma omp simd reduction(+:js)
    for(int j=0; j<mini[b]; ++j)
      if (ind23[b][ii3[b] - j] > ind23[b][GL1 - j]) ++js;
#pragma omp simd
    for(int j=0; j<js; ++j) std::swap(particles[ind23[b][ii3[b]-j]],particles[ind23[b][GL1-j]]);
  }
//5) Divide ir=0 from ir=-1:
//3 3 3 2 2 1 1 0 -1  
#pragma omp parallel for
  for(int b=0; b<Nbin; ++b)    
  {
    const int save_value_of_count0_b=count0[b];
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
    count_minus1[b]=GL1-count_minus1[b];
    ii0[b]=count0[b]-1;
    mini[b]=count_minus1[b];
    if(count0[b]<mini[b]) mini[b]=count0[b];
    int js=0;
#pragma omp simd reduction(+:js)
    for(int j=0; j<mini[b]; ++j)
      if (ind01[b][ii0[b] - j] > ind01[b][GL1 - j]) ++js;
#pragma omp simd
    for(int j=0; j<js; ++j)
      std::swap(particles[ind01[b][ii0[b]-j]],particles[ind01[b][GL1-j]]);
    count0[b]=save_value_of_count0_b;
  }
#pragma omp parallel for reduction(+:POSITION_MINUS1,POSITION0,POSITION1,POSITION2,POSITION3,POSITION23)
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
//§Ù§Õ§Ö§ã§î §Õ§à§Ý§Ø§ß§à §Ú§Õ§ä§Ú §ã§Ý§Ú§ñ§ß§Ú§Ö §Þ§Ú§ß§Ú §ñ§ë§Ú§Ü§à§Ó §Ó §ñ§ë§Ú§Ü§Ú §Õ§Ý§ñ 0, 1, 2, 3, §å§Õ§Ñ§Ý§Ö§ß§Ú§Ö 0, §Ú §á§Ö§â§Ö§Ü§Ý§Ñ§Õ§í§Ó§Ñ§ß§Ú§Ö 3, 2, 1 §Ó §Ú§ã§ç§à§Õ§ß§í§Û §ñ§ë§Ú§Ü
  pointer0[0]=pointer1[0]=pointer2[0]=pointer3[0]=0;
  pointer_minus1[0]=ARR_MINUS1_COUNTER;
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
  // §ã§Ý§Ú§ñ§ß§Ú§Ö §ñ§ë§Ú§Ü§à§Ó §Õ§Ý§ñ 1, 2, 3 §Ó §Þ§Ñ§ã§ã§Ú§Ó particles
  memcpy(&particles[0],&arr3[0],POSITION3*sizep);
  memcpy(&particles[POSITION3],&arr2[0],POSITION2*sizep);
  memcpy(&particles[POSITION23],&arr1[0],POSITION1*sizep);
  memcpy(&particles[POSITION23+POSITION1],&arr0[0],POSITION0*sizep);
  memcpy(&particles[POSITION23+POSITION1+POSITION0],&arr_minus1[0],
         POSITION_MINUS1*sizep);  
}
