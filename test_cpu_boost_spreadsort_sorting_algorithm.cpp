#include <iostream>
#include <cmath>
#include <random>
#include <vector>

#include <stdio.h>
#include <algorithm>
#include <chrono>



#include <boost/sort/spreadsort/spreadsort.hpp>

//HOW TO INSTALL Boost?
//Download .tar.gz file and untar it.
//cd to its directory
//run $./bootstrap.sh --prefix=/path/to/the/installation/folder/of/Boost --without-libraries=graph_parallel,mpi
//run $./b2 install
//If you have root rights and specified system directories for installation
//(e. g. /usr/local/ or /opt/), you need to run this command as a superuser:
//run $sudo ./b2 install
//Finally, you need to export in ~/.bashrc the following environment variables:
//export BOOST_ROOT=/cluster/users/70-gaa/boost/boost-1.89.0
//export LD_LIBRARY_PATH=/cluster/users/70-gaa/boost/boost-1.89/lib:$LD_LIBRARY_PATH
//this variable contains the path to Boost headers it is responsible for compiling code the code using Boost from the command line without manually specifying -I /path/to/Boost
//export CPATH=$BOOST_ROOT/include:$CPATH
//And, finally $ source ~/.bashrc
//Now you can compile the code using Boost in the command line without
//manually specifying -I or -l or -L.


//////////////////////////////////////////////////////////////////////////
////Works only with Boost 1.89
////Tested that does not work with Boost version <=1.74 
//////////////////////////////////////////////////////////////////////////
//TEST boost::sort::spreadsort:
//////////////////////////////////////////////////////////////////////////

// $ g++ -std=c++17 -O3 -march=native -mtune=native -mcmodel=large test_cpu_boost_spreadsort_sorting_algorithm.cpp -o test_cpu_boost_spreadsort_sorting_algorithm

constexpr int N=20'000'000;

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
  cout<<BOOST_VERSION<<endl;
  cout<<BOOST_LIB_VERSION<<endl;
  std::vector<P> particles(N);
  for(int i=0; i<N; ++i) particles[i].ir=rand()%5-1;
  std::vector<P> seq_vector=particles;
/*
  cout<<"Check the initial array:"<<endl;
  for(int i=0; i<N; ++i) cout<<v[i]<<" ";
  cout<<endl;
*/
/*
  //THIS 3-ARGUMENT VERSION DOES NOT COMPILE!!!
  boost::sort::spreadsort::spreadsort(v.begin(), v.end(),[](int x){
                                                           return static_cast<unsigned int> (x) ^ 0x80000000;
                                                         });
  //THIS 3-ARGUMENT VERSION DOES NOT COMPILE!!!
  KeyExtractor extractor;
  boost::sort::spreadsort::spreadsort(v.begin(), v.end(), extractor);
*/
  //ONLY THE 2-ARGUMENT VERSION COMPILES!!!:
  //boost::sort::spreadsort::integer_sort(particles.begin(), particles.end());


  
  //THIS (just below) somewhy fails at N > 10000:
  //KeyExtractor extractor;
  //boost::sort::spreadsort::integer_sort(particles.begin(), particles.end(), extractor);
  auto t1=std::chrono::steady_clock::now();
  boost::sort::spreadsort::integer_sort(particles.begin(), particles.end(),
                                        [](const P & p, size_t offset) -> unsigned int {return static_cast<unsigned int> (p.ir) ^ 0x80000000;});
  
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
