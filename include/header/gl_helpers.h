#ifndef GLHELP 
#define GLHELP

#include <array>
#include <vector>
#include <header/creature.h>
#include <header/camera.h>
#include <header/config.h>
#include <glm/glm.hpp>

#include <GLFW/glfw3.h>
#include <glad/glad.h>

const float triangle_scale_factor = std::sqrt(3.0)/2.0;

void generate_creature_buffers(unsigned int& body_VAO,unsigned int& body_VBO,unsigned int& eye_VAO, unsigned int& eye_VBO,int N,std::vector<float>& eye_arr);

std::vector<float> creature_body_model_generator(Creature& c,Camera& camera);

void position_array_builder(std::vector<Creature>& creatures, std::vector<float> &eye_arr, std::vector<int> &index_arr);

void gl_position_array_builder(std::vector<Creature>& creatures, std::vector<float> &eye_arr,Camera& camera);

void makeRGBTexture(
    std::vector<unsigned char>& texData,
    std::vector<float>& lattice,
    int width, int height,Camera &camera
    ,MainConfig &config);

std::vector<float> quad_generator(MainConfig &config);

#endif
