#ifndef CREATURE_CLASS_H
#define CREATURE_CLASS_H

#include <vector>
#include <array>
#include <glm/glm.hpp>
#include <cmath>

class Brain {
  public:
    std::vector<float> neurons; //matrix data

    Brain(std::vector<float> input_neurons);
    Brain();
};

class Eye {
  public: 
    float length;
    float angle_offset;
  
    Eye(float length, float angle_offset);
  glm::vec4 get_color(){
    return {0.0f,0.0f,0.0f,0.0f};
  }
};

class Creature
{
    public:

      
      Brain brain;
      std::array<float,2> position;
      std::array<float,2> velocity = {0.0f,0.0f};
        float mass;
        float radius = 2.0*sqrt(mass/M_PI);
        float rotation;
  
        float world_size;
        float tile_size;

        glm::vec3 col;
        std::vector<Eye> eyes;

        Creature(std::array<float,2> position,std::array<float,2> velocity,float mass,float rotation,glm::vec3 col,std::vector<Eye> eyes,Brain brain);
            
        
        void evolve_position(float dt,float linear_force,float angular_force);
        
        void eat(std::vector<std::vector<float>>& lattice);
        std::vector<float> get_eye_arrays();
     //   std::vector<float> get_input();


    float energy = 100.0f;
    float eating_efficiency = 1.0f;


};

#endif

