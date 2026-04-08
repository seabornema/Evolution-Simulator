#ifndef ISING_H
#define ISING_H




#include <random>
#include <cmath>
//#include <algorithm>
#include <vector>



int zuper_modulus(int a,int b);

float delta_H(const std::vector<std::vector<float>> &input_matrix, int x, int y);

void evolve_model(std::vector<std::vector<float>> &input_matrix,float Beta,int steps);


void initialize_model(std::vector<std::vector<float>> &input_matrix);
#endif
