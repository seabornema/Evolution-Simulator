#ifndef CREATURE_CLASS_H
#define CREATURE_CLASS_H

#include <vector>
#include <array>


class Creature
{
    public:
        array<2,float> position;
        array<2,float> velocity = {0.0f,0.0f};
        float mass;
        float rotation;

        Creature(array<2,float> position,array<2,float> velocity = {0.0f,0.0f},float mass,float rotation);
      
        void evolve_position(float dt);
};

#endif

