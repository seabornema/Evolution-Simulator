#ifndef GLOBALS 
#define GLOBALS 

struct BrainConfig { 
  int N_layers = 2; 
  int layer_size = 16; 
  int input_size = 32;
  int output_size = 4;
  int brain_number() {  
  return (output_size+input_size)*(layer_size) + (N_layers-1)* (int)std::pow(layer_size,2);
  }
};
struct CreatureConfig { 
  //random initial generation
  float min_mass = 0.8f; 
  float max_eye_distance = 3.0f; 
  float min_eye_distance = 0.5f; 
  int max_eyes = 6; 
  float initial_eye_probability = 0.5f;
  float max_speed = 10.0f; 

  //important stats
  int brain_mutation_max = 50;
  float mass_mutation_rate = 0.05f;
  float eye_mutation_rate  = 0.05f;
  float color_mutation_strength = 0.2f;
         
  float water_penalty = 2.0f;
  float eating_inefficiency = 0.087f;
  float movement_inefficiency = 0.05f;

  float deltaT = 0.1f;
  float drag_coefficient = 2.0;
}; 

struct WorldConfig {
  int MaxCreatures = 1000;
  int L = 100; 
  float tile_size = 1.0f; 
  int L_sq(){
    return L*L;
  }
  float world_size(){
    return (float)L*tile_size;
  }
  int initial_creature_count = 200; 
};

struct GraphicsConfig {
  int width = 800;
  int height = 600;

};

struct MainConfig {
  BrainConfig brainConfig;
  CreatureConfig creatureConfig;
  WorldConfig worldConfig;
  GraphicsConfig graphicsConfig;
};
#endif
