#include <array>
#include <vector>
#include <header/creature.h>
#include <glm/glm.hpp>
#include <header/camera.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <header/config.h>

const float triangle_scale_factor = std::sqrt(3.0)/2.0;
void generate_creature_buffers(unsigned int& body_VAO,unsigned int& body_VBO,unsigned int& eye_VAO, unsigned int& eye_VBO,int N,std::vector<float>& eye_arr) {

    glBindVertexArray(body_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, body_VBO);
    glBufferData(GL_ARRAY_BUFFER, 30 * sizeof(float)* N, NULL, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)(8*sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)(9*sizeof(float)));
    glEnableVertexAttribArray(4);

    glBindVertexArray(eye_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, eye_VBO);
    glBufferData(GL_ARRAY_BUFFER, (eye_arr.size()) * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
}


std::vector<float> creature_body_model_generator(Creature& c,Camera& camera){ 
  float l = c.radius;


  float cam_x = -camera.Position.x / camera.Zoom;
    float cam_y = -camera.Position.y / camera.Zoom;

    float dx = c.position[0]- cam_x;
    float dy = c.position[1] - cam_y;
    dx = dx - c.world_size * std::round(dx / c.world_size);
    dy = dy - c.world_size * std::round(dy / c.world_size);



    glm::vec3 Co = c.col;
    std::array<float,2> Cpos = {cam_x+dx,cam_y+dy};
    std::vector<float> temp =
    {Cpos[0] ,Cpos[1] + l,0.0f,Co[0],Co[1],Co[2],
      Cpos[0],Cpos[1],0.5f*c.radius,c.rotation,

     Cpos[0] + l*triangle_scale_factor,Cpos[1] - l*0.5f, 0.0f,Co[0],Co[1],Co[2],
      Cpos[0],Cpos[1],0.5f*c.radius,c.rotation,

     Cpos[0] -l*triangle_scale_factor,Cpos[1] -l*0.5f, 0.0f,Co[0],Co[1],Co[2],
     Cpos[0],Cpos[1],0.5f*c.radius,c.rotation};
    return temp;
}

void position_array_builder(std::vector<Creature>& creatures, std::vector<float> &eye_arr, std::vector<int> &index_arr) {
    int starting_point = 0;
    int index_start = 0;
    int c_size = creatures.size();

    // worst case: 4 floats for body + 6 eyes * 4 floats each = 28 floats per creature
    eye_arr.assign(c_size * 28, 0.0f);
    // worst case: 1 body entry + 6 eye entries = 7 indices per creature  
    index_arr.assign(c_size * 7, 0);

    for(int i = 0; i < c_size; i++){
        creatures[i].write_position_arrays(eye_arr, starting_point, index_arr, i, index_start);
    }

    eye_arr.resize(starting_point);
    index_arr.resize(index_start);
}

void gl_position_array_builder(std::vector<Creature>& creatures, std::vector<float> &eye_arr,Camera& camera) {
    int starting_point = 0;
    int c_size = creatures.size();

    // worst case: 4 floats for body + 6 eyes * 4 floats each = 28 floats per creature
    eye_arr.assign(c_size * 28, 0.0f);

    for(int i = 0; i < c_size; i++){
        creatures[i].write_gl_position_arrays(eye_arr, starting_point,camera);
    }

    eye_arr.resize(starting_point);
}


void makeRGBTexture(
    std::vector<unsigned char>& texData,
    std::vector<float>& lattice,
    int width, int height,
    Camera& camera,
    MainConfig &config
) {
  float tile_size = config.worldConfig.tile_size;
    const int n = (int)std::sqrt((float)lattice.size());
    texData.resize(width * height * 3);
 float px_to_world = 1.0f / camera.Zoom;
float origin_wx = (-camera.Position.x) * px_to_world - (width  * 0.5f) * px_to_world;
float origin_wy = (-camera.Position.y) * px_to_world - (height * 0.5f) * px_to_world;   
for (int py = 0; py < height; ++py) {
        float wy = origin_wy + py * px_to_world;
        int ly = ((int)std::floor(wy / tile_size)) % n;
        if (ly < 0) ly += n;
        int row_base = ly * n;

        unsigned char* row_ptr = texData.data() + py * width * 3;

        for (int px = 0; px < width; ++px) {
            float wx = origin_wx + px * px_to_world;
            int lx = ((int)std::floor(wx / tile_size)) % n;
            if (lx < 0) lx += n;

            float val = lattice[row_base + lx];

            unsigned char r, g, b;
            if (val > 1.0f) {
                r = (unsigned char)std::min(val / 50.0f * 255.0f, 255.0f);
                g = 0;
                b = 50;
            } else if (val > 0.0f) {
                r = 0;
                g = (unsigned char)(val * 255.0f);
                b = 0;
            } else {
                r = 0;
                g = 0;
                b = 255;
            }

            row_ptr[px * 3    ] = r;
            row_ptr[px * 3 + 1] = g;
            row_ptr[px * 3 + 2] = b;
        }
    }
}
std::vector<float> quad_generator(MainConfig &config) {
return {
    // positions          // texture coords
     -1.0f,  1.0f, 0.0f,   0.0f, 1.0f,   // top left 
     -1.0f, -1.0f, 0.0f,   0.0f, 0.0f,   // bottom left
     1.0f, 1.0f, 0.0f,   1.0f, 1.0f,   // top right
     1.0f,  -1.0f, 0.0f,   1.0f, 0.0f    // bottom right
};
}
