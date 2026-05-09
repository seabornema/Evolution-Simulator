#ifndef CREATURE_CLASS_H
#define CREATURE_CLASS_H

#include <vector>
#include <array>
#include <glm/glm.hpp>
#include <cmath>
#include <header/camera.h>

#include <iostream>
#include <algorithm>
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
    //inputs: rotation, energy,vx,vy, color of tile at pos, color of tiles at eyes
    //outputs: eat, move, rotate, reproduce
    public:
      Brain brain;
      std::array<float,2> position;
      std::array<float,2> velocity = {0.0f,0.0f};

        int brain_mutation_rate = 50;
        float mass_0_mutation_rate = 0.05f;
        float mass_1_mutation_rate = 0.05f;

        float offspring_energy_mutation_rate = 0.05f;

        float eye_mutation_rate  =0.05f;
        float color_mutation_strength = 0.2f;
         
        float water_penalty= 2.0f;
        float eating_efficiency = 0.30f;
        float eating_energy_cost = 0.15f;
        float mass_to_energy_const = 8.0f;

        float movement_inefficiency = 0.05f;

        float dt = 0.1f;
        float drag_coefficient = 2.0;
        float initial_energy=20.0f;

        float ambient_energy_loss_const = 0.25f;
      

        float mass_0;
        float mass_1;
  
        float offspring_energy; //percent of energy the offspring gets, not accounting for loss

        float radius; 
        float rotation;
  
        float world_size;
        float tile_size;

        glm::vec3 col;
        std::vector<Eye> eyes;

        float effective_mass();
        Creature(std::array<float,2> position,std::array<float,2> velocity,float mass_0,float mass_1,float rotation,glm::vec3 col,std::vector<Eye> eyes,Brain brain,float offspring_energy);
            

        void time_evolve(std::vector<float> inputs,std::vector<Creature> &creatures,std::vector<float> &lattice,int n);

        
        void evolve_position(float linear_force, float angular_force,float lattice_color);
        void eat(float &current);
        void die(float &current);
        void reproduce(std::vector<Creature> &creatures);
        float random_range(float min, float max);


        void write_position_arrays(std::vector<float>& destination, int& starting_point,std::vector<int> &index_arr,int index,int &index_start);
        void write_gl_position_arrays(std::vector<float>& destination, int& starting_point, Camera &camera);
        
    float energy= initial_energy;


};

#endif

