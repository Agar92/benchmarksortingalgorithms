#include <iostream>
#include <random>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <cassert>
#include <omp.h>
#include <cassert>

#include "particle.h"
#include "extra.h"
#include "seq_sort.h"


// README:
// How to compile this piece of code:
// $g++ -O3 -Wall -std=c++17 -mcmodel=large -fopenmp benchmark.cpp -o benchmark
// Run it:
// $./benchmark


using namespace std;

using seq_sort::N;

template <int Nt>
void print(int (&arr)[Nt], const char * msg="")
{
  assert(Nt<100000 &&
         "Nt is too big to output so many array elements!");
  if(strlen(msg) > 0) cout<<msg<<endl;
  for(int i=0; i<Nt; ++i) cout<<arr[i]<<" ";
  cout<<endl;
}

//1st buffer (input)
P parr[N];
//2nd buffer (output)
P parr2[N];

//initialization
void initialize()
{
  for(int i=0; i<LIFE; ++i)
  {
    parr[i].ir=(rand()%5-1);
    parr[i].id=i;
    //we let r and p fields of the structure P be initialized
    //using default P constructor:
    //r(0.0,0.0,0.0)
    //and
    //p(0.0,0.0,0.0)
  }
}

//sorting:
template <size_t Nt>
void mysort_Nthreads(P (&parray)[Nt])
{
  int COUNTbin[5][Nbin];
  memset(COUNTbin,0,sizeof(COUNTbin));
  int OFFSET[5][Nbin];
  int COUNT[5];
  memset(COUNT,0,sizeof(COUNT));
  int COUNTERbin[5][Nbin];
  memset(COUNTERbin,0,sizeof(COUNTERbin));
  //
  //Create bins:
  int init[Nbin];
  int fin[Nbin];
  const int dL=LIFE/Nbin;
  const int DL=dL+1;
  const int n=Nbin-LIFE%Nbin;
#pragma omp parallel for
  for(int b=0; b<Nbin; b++)    
  {
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
  //
#pragma omp parallel for
  for(int b=0; b<Nbin; b++)    
    for(int j=init[b]; j<fin[b]; j++)
      COUNTbin[(3-parray[j].ir)][b]++;
  
#pragma omp parallel for
  for(int j=0; j<5; j++)
    for(int b=0; b<Nbin; b++)
      COUNT[j] += COUNTbin[j][b];
  OFFSET[0][0]=0;
  for(int j=1; j<5; ++j)
    OFFSET[j][0]=OFFSET[j-1][0]+COUNT[j-1];
#pragma omp parallel for
  for(int j=0; j<5; j++)
    for(int b=1; b<Nbin; b++)
      OFFSET[j][b]=COUNTbin[j][b-1]+OFFSET[j][b-1];
  
#pragma omp parallel for
  for(int b=0; b<Nbin; b++)    
  {
    for(int j=init[b]; j<fin[b]; j++)
    {
      const int ir=3-parr[j].ir;
      parr2[ OFFSET[ir][b] + COUNTERbin[ir][b]++ ]=parr[j];
    }
  }
}


int main()
{
  //initialization:
  initialize();
  
  const int SORTING_ALGO=1;
  if(0 == SORTING_ALGO)
    cout<<"***You chose std::sort sorting algorithm***"<<endl;
  if(1 == SORTING_ALGO)
    cout<<"***You chose mysort_Nthreads sorting algorithm***"<<endl;
  if(2 == SORTING_ALGO)
    cout<<"***You chose mysort sorting algorithm from seq_sort.h***"<<endl;
  if(3 == SORTING_ALGO)
    cout<<"***You chose TPT3_sort sorting algorithm from extra.h***"<<endl;  
  const auto begin=std::chrono::steady_clock::now();
  //sorting:
  if(0==SORTING_ALGO)
  {
    /*
    //Simple sequential (1 thread) bubble sort:
    for(int i=0; i<LIFE; ++i)
    {
      for(int j=0; j<LIFE-1-i; ++j)
      {
        if(parr[j].ir<parr[j+1].ir ||
           (parr[j].ir==parr[j+1].ir
            && parr[j].id>parr[j+1].id))
          std::swap(parr[j],parr[j+1]);
      }
    }
    */
    std::sort(parr,parr+LIFE,[](const P & l, const P & r)->bool
                             {
                               return l.ir>r.ir;// ||
                               //(l.ir==r.ir && l.id<r.id);
                             });
  }
  if(1==SORTING_ALGO) mysort_Nthreads(parr);
  if(2==SORTING_ALGO) seq_sort::mysort(parr);
  if(3==SORTING_ALGO) TPT3_sort(parr);

  const auto end=std::chrono::steady_clock::now();
  const auto time_elapsed=
    std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
  cout<<"T="<<"\033[31m"<<time_elapsed<<"\033[34m"<<" ms"<<endl;
  cout<<"\033[37m"<<"Nbin="<<Nbin<<endl;
  
  return 0;
}
