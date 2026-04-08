#include <header/creature.h>
#include <array>
#include <glm/glm.hpp>
#include <cmath>



Brain::Brain(std::vector<float> input_neurons) {
  Brain::neurons = input_neurons;
}
Brain::Brain() {
  Brain::neurons = {};
}

Eye::Eye(float length,float angle_offset) {
  Eye::length = length;
  Eye::angle_offset = angle_offset;

}

Creature::Creature(std::array<float,2> position,std::array<float,2> velocity,float mass,float rotation,glm::vec3 col,std::vector<Eye> eyes,Brain brain){
  Creature::position = position;
  Creature::velocity = velocity;
  Creature::mass = mass;
  Creature::rotation = rotation;
  Creature::col = col;
  Creature::eyes = eyes;
  Creature::brain = brain;
}

void Creature::evolve_position(float dt,float linear_force,float angular_force){

  rotation += angular_force*dt;

  velocity[0] += std::cos(rotation)*(linear_force/mass)*dt;
  velocity[1] += std::sin(rotation)*(linear_force/mass)*dt;
 if (position[0] < 0 || position[0] > world_size) {
    velocity[0] *= -1;
}

if (position[1] < 0 || position[1] > world_size) {
    velocity[1] *= -1;
}


  position[0] += velocity[0]*dt;
  position[1] += velocity[1]*dt;

}
void Creature::eat(std::vector<std::vector<float>>& lattice) {
    int x = (int)std::floor(position[0] / tile_size);
    int y = (int)std::floor(position[1] / tile_size);

    if (x < 0) x = 0;
    if (x >= lattice.size()) x = lattice.size() - 1;

    if (y < 0) y = 0;
    if (y >= lattice[0].size()) y = lattice[0].size() - 1;

    energy += lattice[x][y]*0.1f; 
    lattice[y][x] *= 0.9f;


}

std::vector<float> Creature::get_eye_arrays() {
  std::vector<float> temp;
  for(Eye E : eyes){
    temp.push_back(position[0]);
    temp.push_back(position[1]);
  
    temp.insert(temp.end(),{0.0f,1.0f,1.0f,1.0f});


    temp.push_back(position[0]+E.length*std::cos(rotation+E.angle_offset));
    temp.push_back(position[1]+E.length*std::sin(rotation+E.angle_offset));
    temp.insert(temp.end(),{0.0f,1.0f,1.0f,1.0f});
  }
  return temp;

}

//std::vector<float> get_input() {
//  std::vector<float> temp = {saturation,rotation,};
//}
