/**
 * Created by Kenneth Chiguichon.
 * Full details can be found in the readme.txt file, but feel free to share, use, or modify any or all 
 * code found in this file.
 * However credit must be given to the original author (Kenneth Chiguichon).
 * Note: Must compile with -std=c++11 or higher flag enabled
 */
#include <iostream>
#include <random>
#include <time.h>
#include <string.h>

#include <vector>
#include <array>
#include <algorithm>
#include <chrono>

// $ g++ -std=c++17 -O0 radixsort.cpp -o radixsort

constexpr int N=20'000'000;
constexpr int minNum=-1;
constexpr int maxNum=3;



using namespace std;

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

bool arrayIsSorted(P*,int);
P* randomIntArray(int,int,int,mt19937&);
void printArray(P*,int);
void printArray(P* a, int N, const char * msg);
void radixSort(P*, int);

#ifndef CLOCKS_PER_SEC
#define CLOCKS_PER_SEC (clock_t)1000
#endif //CLOCKS_PER_SEC

int main()
{
/*
    int N, minNum, maxNum;
    cout << "Enter a size: ";
    cin >> N;
    cout << "\nEnter a minimum: ";
    cin >> minNum;
    cout << "\nEnter a maximum: ";
    cin >> maxNum;
*/
    mt19937 seed(time(0));
    P* a = randomIntArray(N,minNum,maxNum,seed);
    //
    std::vector<P> seq_a(N);
    for(int i=0; i<N; ++i) seq_a[i]=a[i]; 
    //
    clock_t startTime, endTime;
    startTime = clock();
////printArray(a, N, "BEFORE SORT:");
    auto t1=std::chrono::steady_clock::now();
    radixSort(a,N);
    auto t2=std::chrono::steady_clock::now();
////printArray(a, N, "AFTER SORT:");
    cout<<"T spent on radix sort on CPU="<<std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()<<" us"<<endl;
    endTime = clock();
    cout <<"Radix Sort took " << (((float) endTime)-((float)startTime)) / CLOCKS_PER_SEC << " second(s)" << endl;
    cout << "a[0] = " << a[0].ir << "\na[(N-1)/2] = " << a[(N-1)>>1].ir <<"\na[N-1] = " << a[N-1].ir << endl;
    cout << boolalpha;
    cout << "No elements out of place for array \"a\": " << arrayIsSorted(a,N) << endl;
    //
    //Sequential sort:
    //
    t1=std::chrono::steady_clock::now();
    std::sort(seq_a.begin(), seq_a.end());
    t2=std::chrono::steady_clock::now();
    cout<<"T seq="<<std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()<<" ms"<<endl;
    //
    //Check for matching of the results: 
    //
    bool MATCH=true;
    for(int i=0; i<N; ++i)
      if(a[i] != seq_a[i])
        MATCH=false;
    if(MATCH)
      cout<<"+ The results MATCH!!!"<<endl;
    else
      cout<<"x The results DO NOT MATCH!!!"<<endl;
    //
    //
    //
    delete[] a;
    return 0;
}

bool arrayIsSorted(P* a,int N){
    for (int i = 0; i < N-1; ++i) if (a[i].ir > a[i + 1].ir) return false;
    return true;
}


P *randomIntArray(int n, int minNum, int maxNum, mt19937 &seed){
    P* a = new P[n];
    uniform_int_distribution<int> genRand(minNum,maxNum);
    for (int i = 0; i<n; ++i) a[i].ir = genRand(seed);
    return a;
}

void printArray(P* a, int N){
    cout << "[";
    int i=0;
    while(i < N-1) cout << a[i++].ir << ", ";
    if(i < N) cout << a[i].ir;
    cout << "]";
    cout<<endl;
}

void printArray(P* a, int N, const char * msg){
    cout<<msg<<endl;
    cout << "[";
    int i=0;
    while(i < N-1) cout << a[i++].ir << ", ";
    if(i < N) cout << a[i].ir;
    cout << "]";
    cout<<endl;
}

void radixSort(P* a, int N){
    const int INT_BIT_SIZE = sizeof(int)<<3, RADIX = 0x100, MASK = RADIX-1, MASK_BIT_LENGTH = 8;
    P* result = new P[N]();
    int* buckets = new int[RADIX](), *startIndex = new int[RADIX](), *temp = nullptr;
    int flag = 0, key = 0;
    bool hasNeg = false;
    while (flag < INT_BIT_SIZE){
        for (int i = 0; i < N; ++i) {
            key = (a[i].ir & (MASK << flag)) >> flag;
            if(key < 0){
                key += MASK;
                hasNeg = true;
            }
            ++buckets[key];
        }
        startIndex[0] = 0;
        for (int j = 1; j < RADIX; ++j) startIndex[j] = startIndex[j - 1] + buckets[j - 1];
        for (int i = N-1; i >= 0; --i){
            key = (a[i].ir & (MASK << flag)) >> flag;
            if(key < 0) key += MASK;
            result[startIndex[key] + --buckets[key]].ir = a[i].ir;
        }
        memcpy(a,result,N*sizeof(P));
        flag += MASK_BIT_LENGTH;
    }
    if(hasNeg){
        int indexOfNeg = 0;
        for (int i = 0; i < N; i++) {
            if(a[i].ir < 0) {
                indexOfNeg = i;
                break;
            }
        }
        memcpy(a,result+indexOfNeg,(N-indexOfNeg)*sizeof(P));
        memcpy(a+(N-indexOfNeg),result,indexOfNeg*sizeof(P));
    }
    delete[] result;
    delete[] buckets;
    delete[] startIndex;
}
