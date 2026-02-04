#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <array>

#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <execution>

//////////////////////////////////////////////////////////////////////////
//TEST stpar nvc++ GPU offloading:
//////////////////////////////////////////////////////////////////////////

// $ nvc++ -stdpar=gpu test_nvc++_stdpar_gpu_sorting.cpp -o test_nvc++_stdpar_gpu_sorting

constexpr int N=20'000'000;

using std::cout;
using std::endl;

//typedef std::array<double, 3> double3;

struct P{
  int ir;
  int id;
  double3 r;
  double3 p;
  P():ir(-100),id(-1),r{},p{}{}
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

struct KeyExtractor
{
  unsigned char operator()(const P & p, size_t offset) const
  {
    unsigned int key = static_cast<unsigned int> (p.ir) ^ 0x80000000;
    const unsigned char * bytes = reinterpret_cast<const unsigned char*>(&key);
    if(offset < sizeof(key))
      return bytes[offset];
    return 0;
  }
  typedef unsigned char result_type;
};



int main()
{
  std::vector<P> particles(N);
  for(int i=0; i<N; ++i) particles[i].ir=rand()%5-1;
  std::vector<P> seq_vector=particles;
/*
  cout<<"Check the initial array:"<<endl;
  for(int i=0; i<N; ++i) cout<<v[i]<<" ";
  cout<<endl;
*/
  auto t1=std::chrono::steady_clock::now();
  std::sort(std::execution::par, particles.begin(), particles.end());
  auto t2=std::chrono::steady_clock::now();
  cout<<"time spent on stpar sort on GPU="<<std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()<<" ms"<<endl;
/*
  cout<<"After sort:"<<endl;
  for(int i=0; i<particles.size(); ++i) cout<<particles[i].ir<<" ";
  cout<<endl;
*/
  //Sequential sort:
  t1=std::chrono::steady_clock::now();
  std::sort(seq_vector.begin(), seq_vector.end());
  t2=std::chrono::steady_clock::now();
  cout<<"time spent on std::::sort="<<std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()<<" ms"<<endl;
  //
  if(seq_vector == particles)
    cout<<"+ SORTING RESULTS MATCH!!!"<<endl;
  else
    cout<<"x SORTING RESULTS DO NOT MATCH!!!"<<endl;

  return 0;
}
