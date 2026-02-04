#pragma once

//
//Dummy structure P defining a particle
//

struct double3{
  float x, y, z;
};

struct P{
  int ir;
  int id;
  double3 r;
  double3 p;
  P():ir(-1),id(-1),r{},p{}{}
  P(int _ir, int _id):ir(_ir),id(_id),r{},p{}{}
};
