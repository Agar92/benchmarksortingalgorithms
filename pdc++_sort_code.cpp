#include <iostream>
#include <cmath>
#include <random>
#include <vector>

#include <stdio.h>
#include <algorithm>
#include <chrono>



#include <sycl/sycl.hpp>
#include <oneapi/dpl/experimental/kernel_templates>

namespace kt=oneapi::dpl::experimental::kt;

//////////////////////////////////////////////////////////////////////////
//Works only with Boost 1.89
//Tested that does not work with Boost version <=1.74 
//////////////////////////////////////////////////////////////////////////
//TEST boost::sort::spreadsort:
//////////////////////////////////////////////////////////////////////////

constexpr int N=10;

using std::cout;
using std::endl;

typedef std::array<double, 3> double3;

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

  //THIS (just below) somewhy fails at N > 10000:
  //KeyExtractor extractor;
  //boost::sort::spreadsort::integer_sort(particles.begin(), particles.end(), extractor);
  auto t1=std::chrono::steady_clock::now();
  //boost::sort::spreadsort::integer_sort(particles.begin(), particles.end(),
  //                                      [](const P & p, size_t offset) -> unsigned int {return static_cast<unsigned int> (p.ir) ^ 0x80000000;});\

  sycl::queue q{sycl::gpu_selector_v};
  //std::uint32_t* keys = sycl::malloc_shared<std::uint32_t>(N, q);
  std::vector<std::uint32_t> host_keys(N);
  std::uint32_t* device_keys =sycl::malloc_device<std::uint32_t>(N, q);
  q.memcpy(device_keys, host_keys.data(), N*sizeof(std::uint32_t)).wait();
  auto e = kt::gpu::esimd::radix_sort<false, 8>(q, device_keys, device_keys+N, kt::kernel_param<32, 32>{});
  //e.wait();
 
  auto t2=std::chrono::steady_clock::now();
  cout<<"time spent on boost::sort::spreadsort::integer_sort="<<std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()<<" ms"<<endl;
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
  return 0;
}
