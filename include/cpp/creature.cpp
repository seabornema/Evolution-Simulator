#include <header/creature.h>
#include <array>
#include <glm/glm.hpp>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <header/config.h>

#include <header/camera.h>

const float moment_of_intertia_prefactor = 0.25f*M_PI;
Brain::Brain(std::vector<float> input_neurons,std::vector<float> cost_function) {
  Brain::neurons = input_neurons;
  Brain::cost_function = cost_function;
}
Brain::Brain() {
  Brain::neurons = {};
  Brain::cost_function = {};
}

Eye::Eye(float length,float angle_offset) {
  Eye::length = length;
  Eye::angle_offset = angle_offset;

}

float Creature::effective_mass(){
  return mass_0 + (energy/config->initial_energy)*mass_1;
}


Creature::Creature(std::array<float,2> position,std::array<float,2> velocity,float mass_0,float mass_1,
    float rotation,glm::vec3 col,std::vector<Eye> eyes,Brain brain,float offspring_energy,const CreatureConfig* config){
  Creature::pos = Vec2(position[0],position[1]);
  Creature::velo = Vec2(velocity[0],velocity[1]);
  Creature::mass_0 = mass_0;
  Creature::mass_1 = mass_1;
  Creature::rotation = rotation;
  Creature::col = col;
  Creature::eyes = eyes;
  Creature::brain = brain;
  Creature::offspring_energy = offspring_energy;
  Creature::config = config;

  Creature::radius = 2.0*sqrt((effective_mass()/M_PI));
}

void Creature::time_evolve(std::vector<float> inputs,std::vector<Creature> &creatures,std::vector<float> &lattice,int n) {
  //inputs have the order: forward_movement, rotational_movement, should eat?, should reproduce?
  //
  int x = std::clamp((int)std::floor(pos.x / tile_size), 0, n - 1);
  int y = std::clamp((int)std::floor(pos.y / tile_size), 0, n - 1);
  float &current = lattice[x+ (n*y)];
  if(energy < 0) {
  die(current);
  return;
  }


  energy -= (current == -1.0f) ?   (config->ambient_energy_loss_const/effective_mass())*config->dt*config->water_penalty*config->evil_water_penalty : (config->ambient_energy_loss_const/effective_mass())*config->dt;
  


  Creature::radius = 2.0*sqrt(effective_mass()/M_PI);

  evolve_position(inputs[0],inputs[1],current);
  if(inputs[2] > 0.5f) {
  eat(current);
  }
  if(inputs[3] > 0.70f && energy > 50) {
  reproduce(creatures);
  }
}



float Creature::random_range(float min, float max) {
  return min + (max - min) * ((float)rand() / RAND_MAX);
}
void Creature::reproduce(std::vector<Creature> &creatures){
  //int x = std::clamp((int)std::floor(position[0] / tile_size), 0, n - 1);
 // int y = std::clamp((int)std::floor(position[1] / tile_size), 0, n - 1);
   Creature temp_creature({std::fmod(pos.x- 3*velo.x,world_size),std::fmod(pos.y- 3*velo.y,world_size)} //position
                           ,{0.0f,0.0f} //velocity
                           ,mass_0 //const mass
                           ,mass_1//linear mass const
                           ,0.0f  //rotation
                           ,col   //color 
                           ,eyes  //eyes
                            ,brain //brian
                            ,offspring_energy
                            ,config);

    temp_creature.tile_size = tile_size;
    temp_creature.world_size = world_size;
    
    if(config->mass_0_mutation_rate > random_range(0.0f,1.0f)){
    temp_creature.mass_0 = std::max(0.5f,mass_0+random_range(-1.0f,1.0f));
    }
    if(config->mass_1_mutation_rate > random_range(0.f,1.0f)){
    temp_creature.mass_1 = std::max(0.1f,mass_1+random_range(-0.1f,0.1f));
    }
    if(config->offspring_energy_mutation_rate > random_range(0.f,1.0f)){
    temp_creature.offspring_energy = std::clamp(offspring_energy+random_range(-0.1f,0.1f),0.1f,1.0f);
    }

    temp_creature.energy = energy*offspring_energy;
    energy *= (1.0f-offspring_energy);
    energy -= temp_creature.mass_0*config->mass_to_energy_const;
    int num_mutations = rand()%config->brain_mutation_rate;
    for(int i =0;i<num_mutations;i++){
      temp_creature.brain.cost_function[rand()%(temp_creature.brain.cost_function.size())] +=  random_range(-0.1f,0.1f);
    }

    if(config->eye_mutation_rate > random_range(0.0f,1.0f) && temp_creature.eyes.size() < config->max_eyes){
     float color_mutation_strength = config->color_mutation_strength;
        temp_creature.col.x = std::clamp(col.r + random_range(-color_mutation_strength, color_mutation_strength), 0.0f, 1.0f);
        temp_creature.col.y = std::clamp(col.g + random_range(-color_mutation_strength, color_mutation_strength), 0.0f, 1.0f);
        temp_creature.col.z = std::clamp(col.b + random_range(-color_mutation_strength, color_mutation_strength), 0.0f, 1.0f);

      if(random_range(0.0f,1.0f) > 0.5){

        temp_creature.eyes.push_back(Eye(random_range(0.5f,5.0f),random_range(0.0f,2*M_PI)));
      } else if(eyes.size() > 1){
        temp_creature.eyes.erase(temp_creature.eyes.begin());

      }
    }

 if(energy > 0) {
  creatures.push_back(temp_creature);
 }
}


void Creature::evolve_position(float linear_force, float angular_force,float lattice_color) {
  float dt = config->dt;
    float magnitude = velo.lengthSq();
    energy -= abs(linear_force)*sqrt(magnitude)*dt;
    energy -= abs(angular_force)*abs(angular_frequency)*dt;
    angular_frequency += (angular_force*dt)/(moment_of_intertia_prefactor*std::pow(radius,4)) - (angular_frequency*config->angular_drag_coefficient*config->dt);
    rotation += angular_frequency * dt;
    float iswater = 0.0f;
    if(lattice_color == -1.0f){
    iswater =1.0f;
    }
    float drag_prefactor = (config->drag_coefficient+(iswater*config->water_penalty))*radius*radius*magnitude;
    float pref = (dt/effective_mass());
    velo += Vec2(pref*(std::cos(rotation) * (linear_force) - drag_prefactor*velo.x), pref*(std::sin(rotation) * (linear_force) - drag_prefactor*velo.y));
    pos += velo*dt;

    pos = Vec2(std::fmod(pos.x, world_size),std::fmod(pos.y, world_size));
    if (pos.x < 0) pos.x += world_size;
    if (pos.y < 0) pos.y += world_size;
}
        

void Creature::collide(Creature & opponent)
{
  float W  = world_size;
  float H  = world_size;
    Vec2 delta{ opponent.pos.x - pos.x, opponent.pos.y - pos.y };

    delta.x -= std::round(delta.x / W) * W;
    delta.y -= std::round(delta.y / H) * H;

    float dist2 = delta.x * delta.x + delta.y * delta.y;
    if (dist2 < 0.01f) return;

    float dist = std::sqrt(dist2);
    Vec2 normal{ delta.x / dist, delta.y / dist };

    float min_dist = std::pow(radius + opponent.radius, 6.f);
    float force = config->repulsion_strength / std::max(dist2, min_dist);

    energy -= config->collision_energy_loss_const*force*(opponent.effective_mass()/effective_mass()); 
    opponent.energy -= config->collision_energy_loss_const*force*(effective_mass()/opponent.effective_mass()); 
    velo.x          -= normal.x * force;
    velo.y          -= normal.y * force;
    opponent.velo.x += normal.x * force;
    opponent.velo.y += normal.y * force;
}




void Creature::eat(float &current) {

    if(current== -1.0f){
      energy -= config->eating_energy_cost*config->dt;
      return;
    }
    energy += 8*(current*config->eating_efficiency - config->eating_energy_cost)*config->dt;

    current -= current*config->dt;
}
void Creature::die(float &current) {
  if(current != -1.0){
    current += effective_mass()*config->mass_to_energy_const;
} 
energy = -100;
}


void Creature::write_position_arrays(std::vector<float>& destination, int& starting_point,std::vector<int> &index_arr,int index,int &index_start) {
  int i = 0;
  float xpos = pos.x;
  float ypos = pos.y;
  destination[starting_point] = xpos; 
  destination[starting_point+1] = ypos; 
  index_arr[index_start] = index;
  starting_point += 2;
  index_start++;
  for(Eye &E : eyes){
    destination[starting_point+i] =xpos+E.length*std::cos(rotation+E.angle_offset); 
    destination[starting_point+i+1] = ypos+E.length*std::sin(rotation+E.angle_offset);
    index_arr[index_start+(i/2)] = index;
    i +=2;
  }
index_start += (i/2);
starting_point += i;
}

void Creature::write_gl_position_arrays(std::vector<float>& destination, int& starting_point, Camera &camera) {
  int i = 0;


  float cam_x = -camera.Position.x / camera.Zoom;
    float cam_y = -camera.Position.y / camera.Zoom;

    float dx = pos.x- cam_x;
    float dy = pos.y - cam_y;
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

