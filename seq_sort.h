#pragma once

#include "particle.h"

//
//A sequential sorting algorithm
//It works less than 3 times slower than
//mysort_Nthreads(...) in benchmark.cpp
//But is You want to use it,
//do not compile it with O1/O2/O3 compiler options
//Somewhy adding these options leads to a sementation faul
//!!! Compile mysort(...) ONLY WITH -O0 compiler option!!!
//

inline const int N=20'000'000;

template <size_t Nt>
void mysort(P (&parray)[Nt])
{
  int POINTER_WRITE=LIFE;
  int j=0;
  //
  int POSITION_MINUS_1=0;
  int POSITION0=0;
  int POSITION1=0;
  int POSITION2=0;
  int POSITION3=0;
  int POSITION23=0;
  for(int i=0; i<LIFE; ++i)
  {
    if(-1==parray[i].ir) POSITION_MINUS_1++;
    else if( 0==parray[i].ir) POSITION0++;
    else if( 1==parray[i].ir) POSITION1++;
    else if( 2==parray[i].ir) POSITION2++;
    else if( 3==parray[i].ir) POSITION3++;
  }
  POSITION23=POSITION2+POSITION3;
//1) sort ir=2,3 from ir=-1,0,1:  
  for(int i=0; i<LIFE; ++i)
  {
    if(parray[i].ir<2)
      parray[POINTER_WRITE++]=parray[i];
    else
      parray[j++]=parray[i];
  }
  memcpy(&parray[j],&parray[LIFE],sizeof(P)*
         (POSITION_MINUS_1+POSITION0+POSITION1));
//2) sort ir=2 from ir=3:
  POINTER_WRITE=LIFE;
  j=0;
  for(int i=0; i<POSITION23; ++i)
  {
    if(2==parray[i].ir)
      parray[POINTER_WRITE++]=parray[i];
    else if(3==parray[i].ir)
      parray[j++]=parray[i];
  }
  memcpy(&parray[j],&parray[LIFE],sizeof(P)*POSITION2);
//3) sort ir=1 from ir=0,-1:
  POINTER_WRITE=LIFE;
  j=POSITION23;
  for(int i=POSITION23; i<LIFE; ++i)
  {
    if(parray[i].ir<1)
      parray[POINTER_WRITE++]=parray[i];
    else
      parray[j++]=parray[i];
  }
  memcpy(&parray[j],&parray[LIFE],sizeof(P)*
         (POSITION_MINUS_1+POSITION0));
//4) sort ir=0 from ir=-1:
  POINTER_WRITE=LIFE;
  ///const int POSITION1=CNT2;
  j=POSITION23+POSITION1;
  for(int i=POSITION23+POSITION1; i<LIFE; ++i)
  {
    if(-1==parray[i].ir)
      parray[POINTER_WRITE++]=parray[i];
    else if(0==parray[i].ir)
      parray[j++]=parray[i];
  }
  memcpy(&parray[j],&parray[LIFE],sizeof(P)*POSITION_MINUS_1);
}
