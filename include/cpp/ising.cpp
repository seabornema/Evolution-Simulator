#include <vector>
#include <random>
#include <header/ising.h>
#include <algorithm>

int zuper_modulus(int a,int b) {
    int temp = a % b;
    if(temp < 0) {
        return temp+b;
    }else {
        return temp;
    }
}


float delta_H(const std::vector<float> &input_matrix, int x, int y,int n) {
    double nearest_neighbors = 0;

    nearest_neighbors += input_matrix[(n*y)+zuper_modulus(x-1, n)]; 
    nearest_neighbors += input_matrix[(n*y)+zuper_modulus(x+1, n)]; 
    nearest_neighbors += input_matrix[x+(n*zuper_modulus(y-1, n))]; 
    nearest_neighbors += input_matrix[x+(n*zuper_modulus(y+1, n))]; 

    return 2.0 * input_matrix[x+(n*y)] * nearest_neighbors;
}

void evolve_model(std::vector<float> &input_matrix,float Beta,int steps,int n){
  for(int i = 0; i < steps; i++){

    int x = rand()%n;
    int y= rand()%(n);
    float temp = (float)(rand()) / RAND_MAX;
        
    float H = delta_H(input_matrix,x,y,n);
    float probability = std::min(1.0f,std::exp(-Beta*H));
    if(probability >=temp) {
        input_matrix[x+(n*y)] *= -1.0f;
    }
  }

}

void regrow_grass(std::vector<float>& lattice, float regrowth_rate) {
    for(int i = 0; i < (int)lattice.size(); i++) {
      float c = lattice[i];
        if(lattice[i] < 1.0f && c != -1.0f) {
            lattice[i] = std::min(1.0f, c + (1-c)*regrowth_rate);
        }
    }
}

void grow_berries(std::vector<float> &input_matrix,int size,int n){
  for(int i=0;i<n;i++){
    int x = rand()%(size*size);
    while (input_matrix[x] < 0.0f) {
      x = rand()%(size*size);
    }
        input_matrix[x] = 50.0f ;
  }
}
void initialize_model(std::vector<float> &input_matrix) {
        for (auto &d : input_matrix) {
            float r = (float) rand() / RAND_MAX;  
            d = (r < 0.5 ? -1.0 : 1.0);             
        }
}

