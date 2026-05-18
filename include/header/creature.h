#ifndef CREATURE_CLASS_H
#define CREATURE_CLASS_H

#include <vector>
#include <array>
#include <glm/glm.hpp>
#include <cmath>
#include <header/config.h>
#include <header/camera.h>

#include <iostream>
#include <algorithm>



struct Vec2 {
    float x, y;
 
    Vec2(float x = 0.0f, float y = 0.0f) : x(x), y(y) {}
 
    Vec2 operator+(const Vec2& rhs) const { return {x + rhs.x, y + rhs.y}; }
    Vec2& operator+=(const Vec2& rhs) { x += rhs.x; y += rhs.y; return *this; }
 
    Vec2 operator*(float s) const { return {x * s, y * s}; }
    Vec2& operator*=(float s) { x *= s; y *= s; return *this; }
 
    float length()  const { return std::sqrt(x * x + y * y); }
    float lengthSq() const { return x * x + y * y; }
 
};
 

class Brain {
  public:
    std::vector<float> neurons; //matrix data
    std::vector<float> cost_function; 
                              

    Brain(std::vector<float> input_neurons,std::vector<float> cost_function);
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
      Vec2 pos; 
      Vec2 velo;

      

        const CreatureConfig* config;

        float mass_0;
        float mass_1;
  
        float offspring_energy; //percent of energy the offspring gets, not accounting for loss

        float radius; 
        float rotation;
        float angular_frequency= 0.0f;
  
        float world_size;
        float tile_size;

        glm::vec3 col;
        std::vector<Eye> eyes;

        float effective_mass();
        Creature(std::array<float,2> position,std::array<float,2> velocity,float mass_0,float mass_1,float rotation,glm::vec3 col,std::vector<Eye> eyes,Brain brain,float offspring_energy,const CreatureConfig* config);
            

        void time_evolve(std::vector<float> inputs,std::vector<Creature> &creatures,std::vector<float> &lattice,int n);

        
        void evolve_position(float linear_force, float angular_force,float lattice_color);
        void eat(float &current);
        void die(float &current);
        void reproduce(std::vector<Creature> &creatures);
        float random_range(float min, float max);


        void write_position_arrays(std::vector<float>& destination, int& starting_point,std::vector<int> &index_arr,int index,int &index_start); 
        void collide(Creature &opponent);
        void write_gl_position_arrays(std::vector<float>& destination, int& starting_point, Camera &camera);
    float energy;


};

#endif

