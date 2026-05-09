#ifndef RANDOMIZERS
#define RANDOMIZERS

#include <random>
#include <header/creature.h>
#include <header/config.h>
#include <vector>
float random_range(float min, float max);

float gauss_number();

Brain random_brain(BrainConfig &brainConfign);

Creature random_creature(MainConfig &config);

#endif
