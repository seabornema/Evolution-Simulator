#include <random>
#include <vector>
#include <header/creature.h>
#include <header/config.h>

extern int N_layers;
extern int Layer_size;


float random_range(float min, float max) {
  return min + (max - min) * ((float)rand() / RAND_MAX);
}

float gauss_number() {
    static std::mt19937 rng(std::random_device{}());
    static std::normal_distribution<float> nd(0.0, 1.0);
    return nd(rng);
}

Brain random_brain(BrainConfig &brainConfig) {
  int brain_number = brainConfig.brain_number(); 
  std::vector<float> temp(brain_number);
  for(int i =0; i < brain_number; i++) {
    temp[i] = (gauss_number());
  }
  int cost_number = brainConfig.cost_function_size();
  std::vector<float> ctemp(cost_number);
  for(int i=0; i< cost_number ; i++) {
  ctemp[i] = random_range(-1.0,1.0);
  }
  return Brain(move(temp),move(ctemp));

}


Creature random_creature(MainConfig &config) {
   float eye_probability = config.creatureConfig.initial_eye_probability;
   int eyenumber = 1;
   float world_scale = config.worldConfig.world_size(); 
   Creature temp_creature({random_range(0.0f,world_scale),random_range(0.0f,world_scale)} //position
                           ,{0.0f,0.0f} //velocity
                           ,std::max(config.creatureConfig.min_mass,abs(gauss_number())/2.0f) //mass_0
                           ,std::max(config.creatureConfig.min_mass,abs(gauss_number())/2.0f) //mass_1
                           ,0.0f  //rotation
                           ,{random_range(0.0,1.0),random_range(0.0,1.0),random_range(0.0,1.0)}   //color 
                           ,{Eye(random_range(config.creatureConfig.min_eye_distance,config.creatureConfig.max_eye_distance),0.0f)},  //eyes
                            Brain(),   //empty brain
                            random_range(0.1f,1.0f)   //percent of energy the offspring gets
                            ,&config.creatureConfig); //config
  
    while(random_range(0.0f,1.0f) < eye_probability && eyenumber < config.creatureConfig.max_eyes) {
        temp_creature.eyes.push_back(Eye(random_range(config.creatureConfig.min_eye_distance,config.creatureConfig.max_eye_distance),random_range(0.0f,2*M_PI)));
        eye_probability = 0.0;
        eyenumber++;
    }
    temp_creature.brain = random_brain(config.brainConfig);
    temp_creature.world_size = world_scale;
    temp_creature.tile_size = config.worldConfig.tile_size;
    return temp_creature;
}

