#include <vector>
#include <random>
#include <header/ising.h>


int zuper_modulus(int a,int b) {
    int temp = a % b;
    if(temp < 0) {
        return temp+b;
    }else {
        return temp;
    }
}


float delta_H(const std::vector<std::vector<float>> &input_matrix, int x, int y) {
    int n = input_matrix.size();
    double nearest_neighbors = 0;

    nearest_neighbors += input_matrix[zuper_modulus(x-1, n)][y]; 
    nearest_neighbors += input_matrix[zuper_modulus(x+1, n)][y]; 
    nearest_neighbors += input_matrix[x][zuper_modulus(y-1, n)]; 
    nearest_neighbors += input_matrix[x][zuper_modulus(y+1, n)]; 

    return 2.0 * input_matrix[x][y] * nearest_neighbors;
}

void evolve_model(std::vector<std::vector<float>> &input_matrix,float Beta,int steps){
  for(int i = 0; i < steps; i++){
  int n = input_matrix.size();

    int x = rand()%n;
    int y= rand()%n;
    float temp = (float)(rand()) / RAND_MAX;
        
    float H = delta_H(input_matrix,x,y);
    float probability = std::min(1.0f,std::exp(-Beta*H));
    if(probability >=temp) {
        input_matrix[x][y] *= -1.0f;
    }
  }

}


void initialize_model(std::vector<std::vector<float>> &input_matrix) {
    for (auto &row : input_matrix) {
        for (auto &d : row) {
            float r = (float) rand() / RAND_MAX;  
            d = (r < 0.5 ? -1.0 : 1.0);             
        }
    }
}

