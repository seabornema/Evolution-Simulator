#ifndef GLOBALS 
#define GLOBALS 

struct BrainConfig { 
  int N_layers = 2; 
  int layer_size = 8; 
  int input_size = 8;
  int output_size = 5;
  int brain_number() {  
  return (output_size+input_size)*(layer_size) + (N_layers-1)* (int)std::pow(layer_size,2);
  }
  int cost_function_size(){
    return input_size*output_size;
  }
};
struct CreatureConfig { 
  //random initial generation
  float min_mass = 0.8f; 
  float max_eye_distance = 3.0f; 
  float min_eye_distance = 0.5f; 
  int max_eyes = 1; 
  float initial_eye_probability = 0.5f;



  float repulsion_strength = 0;
        int brain_mutation_rate = 8;
        float mass_0_mutation_rate = 0.05f;
        float mass_1_mutation_rate = 0.05f;

        float offspring_energy_mutation_rate = 0.05f;

        float eye_mutation_rate  =0.00f;
        float color_mutation_strength = 0.2f;
         
        float water_penalty= 6.0f;
        float evil_water_penalty = 2.0f;
        float eating_efficiency = 0.8f;
        float eating_energy_cost = 0.3f;
        float mass_to_energy_const = 1.0f;

        float movement_inefficiency = 0.05f;

        float dt = 0.1f;
        float drag_coefficient = 1.0f;
        float angular_drag_coefficient = 1.0f;
        float initial_energy=40.0f;
        float collision_energy_loss_const = 1.0f;


        float ambient_energy_loss_const = 0.01f;




  
}; 

struct WorldConfig {
  float random_spawn_rate = 0.0001;
  int MaxCreatures = 1000;
  int L = 240; 
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
  int width = 80;
  int height = 80;

};

struct MainConfig {
  BrainConfig brainConfig;
  CreatureConfig creatureConfig;
  WorldConfig worldConfig;
  GraphicsConfig graphicsConfig;
};
#endif
