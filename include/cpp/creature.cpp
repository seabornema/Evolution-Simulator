#include <header/creature.h>
#include <array>
#include <glm/glm.hpp>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

#include <header/camera.h>

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

float Creature::effective_mass(){
  return mass_0 + (energy/initial_energy)*mass_1;
}


Creature::Creature(std::array<float,2> position,std::array<float,2> velocity,float mass_0,float mass_1,float rotation,glm::vec3 col,std::vector<Eye> eyes,Brain brain,float offspring_energy){
  Creature::position = position;
  Creature::velocity = velocity;
  Creature::mass_0 = mass_0;
  Creature::mass_1 = mass_1;
  Creature::rotation = rotation;
  Creature::col = col;
  Creature::eyes = eyes;
  Creature::brain = brain;
  Creature::offspring_energy = offspring_energy;

  Creature::radius = 2.0*sqrt((effective_mass()/M_PI));
}

void Creature::time_evolve(std::vector<float> inputs,std::vector<Creature> &creatures,std::vector<float> &lattice,int n) {
  //inputs have the order: forward_movement, rotational_movement, should eat?, should reproduce?
  //
  int x = std::clamp((int)std::floor(position[0] / tile_size), 0, n - 1);
  int y = std::clamp((int)std::floor(position[1] / tile_size), 0, n - 1);
  float &current = lattice[x+ (n*y)];


  energy -= (current == -1.0f) ?   (ambient_energy_loss_const/effective_mass())*dt*water_penalty : (ambient_energy_loss_const/effective_mass())*dt;


  Creature::radius = 2.0*sqrt(effective_mass()/M_PI);

  evolve_position(inputs[0],inputs[1],current);
  if(inputs[2] > 0.5f) {
  eat(current);
  }
  if(inputs[3] > 0.99f && energy > 60) {
  reproduce(creatures);
  }
  if(energy < 0) {
  die(current);
  }

}

float Creature::random_range(float min, float max) {
  return min + (max - min) * ((float)rand() / RAND_MAX);
}
void Creature::reproduce(std::vector<Creature> &creatures){
   Creature temp_creature({position[0],position[1]} //position
                           ,{0.0f,0.0f} //velocity
                           ,mass_0 //const mass
                           ,mass_1//linear mass const
                           ,0.0f  //rotation
                           ,col   //color 
                           ,eyes  //eyes
                            ,brain //brian
                            ,offspring_energy); 

    temp_creature.tile_size = tile_size;
    temp_creature.world_size = world_size;
    
    if(mass_0_mutation_rate > random_range(0.f,1.0f)){
    temp_creature.mass_0 = std::min(0.5f,mass_0+random_range(-1.0f,1.0f));
    }
    if(mass_1_mutation_rate > random_range(0.f,1.0f)){
    temp_creature.mass_1 = std::min(0.0f,mass_0+random_range(-0.3f,0.3f));
    }
    if(offspring_energy_mutation_rate > random_range(0.f,1.0f)){
    temp_creature.offspring_energy = std::clamp(offspring_energy+random_range(-0.1f,0.1f),0.0f,1.0f);
    }

    temp_creature.energy = 0.5f*energy*offspring_energy;
    energy *= 0.5f*(1.0f-offspring_energy);
    energy -= temp_creature.mass_0*mass_to_energy_const;

    int num_mutations = rand()%brain_mutation_rate;
    for(int i =0;i<num_mutations;i++){
      temp_creature.brain.neurons[rand()%(temp_creature.brain.neurons.size()-1)] +=  random_range(-1.0f,1.0f);
    }
    if(eye_mutation_rate > random_range(0.0f,1.0f) && temp_creature.eyes.size() < 6){
     
        temp_creature.col.x = std::clamp(col.r + random_range(-color_mutation_strength, color_mutation_strength), 0.0f, 1.0f);
        temp_creature.col.y = std::clamp(col.g + random_range(-color_mutation_strength, color_mutation_strength), 0.0f, 1.0f);
        temp_creature.col.z = std::clamp(col.b + random_range(-color_mutation_strength, color_mutation_strength), 0.0f, 1.0f);

      if(random_range(0.0f,1.0f) > 0.5){

        temp_creature.eyes.push_back(Eye(random_range(0.5f,5.0f),random_range(0.0f,2*M_PI)));
      } else if(eyes.size() > 1){
        temp_creature.eyes.erase(temp_creature.eyes.begin());

      }
    }


  creatures.push_back(temp_creature);
}


void Creature::evolve_position(float linear_force, float angular_force,float lattice_color) {
    float magnitude = (velocity[0]*velocity[0] + velocity[1]*velocity[1]);
    energy -= linear_force*sqrt(magnitude)*dt; 
    rotation += angular_force * dt;
    float iswater = 0.0f;
    if(lattice_color == -1.0f){
    iswater =1.0f;
    }
    float drag_prefactor = (drag_coefficient+(iswater*water_penalty))*radius*radius*magnitude;
    float pref = (dt/effective_mass());
    velocity[0] += pref*(std::cos(rotation) * (linear_force) - drag_prefactor*velocity[0]);
    velocity[1] += pref*(std::sin(rotation) * (linear_force) - drag_prefactor*velocity[1]);
    position[0] += velocity[0] * dt;
    position[1] += velocity[1] * dt;

    // Periodic boundaries
    position[0] = std::fmod(position[0], world_size);
    position[1] = std::fmod(position[1], world_size);
    if (position[0] < 0) position[0] += world_size;
    if (position[1] < 0) position[1] += world_size;
}
void Creature::eat(float &current) {


    if(current== -1.0f){
      energy -= eating_energy_cost*dt;
      return;
    }
    energy += (current*eating_efficiency - eating_energy_cost)*dt;

    current -= current*dt;
}
void Creature::die(float &current) {
  if(current != -1.0){
    current += mass_0*0.0f;
}
}

void Creature::write_position_arrays(std::vector<float>& destination, int& starting_point,std::vector<int> &index_arr,int index,int &index_start) {
  int i = 0;
  float xpos = position[0];
  float ypos = position[1];
  destination[starting_point] = xpos; 
  destination[starting_point+1] = ypos; 
  destination[starting_point+2] = xpos; 
  destination[starting_point+3] = ypos; 
  index_arr[index_start] = index;
  starting_point += 4;
  index_start++;
  for(Eye &E : eyes){
    destination[starting_point+i] = (xpos);
    destination[starting_point+i+1] =  (ypos);

    destination[starting_point+i+2] =xpos+E.length*std::cos(rotation+E.angle_offset); 
    destination[starting_point+i+3] = ypos+E.length*std::sin(rotation+E.angle_offset);
    index_arr[index_start+(i/4)] = index;
    i +=4;
  }
index_start += (i/4);
starting_point += i;
}

void Creature::write_gl_position_arrays(std::vector<float>& destination, int& starting_point, Camera &camera) {
  int i = 0;


  float cam_x = -camera.Position.x / camera.Zoom;
    float cam_y = -camera.Position.y / camera.Zoom;

    float dx = position[0]- cam_x;
    float dy = position[1] - cam_y;
    dx = dx - world_size * std::round(dx / world_size);
    dy = dy - world_size * std::round(dy / world_size);

    float xpos = cam_x + dx;
    float ypos = cam_y + dy;

  for(Eye &E : eyes){
    destination[starting_point+i] = (xpos);
    destination[starting_point+i+1] =  (ypos);

    destination[starting_point+i+2] =xpos+E.length*std::cos(rotation+E.angle_offset); 
    destination[starting_point+i+3] = ypos+E.length*std::sin(rotation+E.angle_offset);
    i +=4;
  }
starting_point += i;
}


