#ifndef ISING_H
#define ISING_H




#include <random>
#include <cmath>
#include <algorithm>
#include <vector>



int zuper_modulus(int a,int b);

float delta_H(const std::vector<float> &input_matrix, int x, int y,int n);

void evolve_model(std::vector<float> &input_matrix,float Beta,int steps,int n);

void regrow_grass(std::vector<float>& lattice, float regrowth_rate);

void grow_berries(std::vector<float> &input_matrix,int size,int n);
void initialize_model(std::vector<float> &input_matrix);
#endif
