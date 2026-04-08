#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <header/shader.h>
#include <header/camera.h>
#include <header/ising.h>
#include <header/creature.h>
#include <header/computeshader.h>

#include <iostream>
#include <vector>
#include <random>
#include <array>
#include <ctime>

using namespace std;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
float zoom_scale = 100;
const int n_creatures = 10;

//tiling 
const float tile_size = 1.0;
const int n = 100;

//neural_network
//
const int N_layers = 1;
const int Layer_size = 1;

const float triangle_scale_factor = sqrt(3)/2;

vector<Creature> creatures;

vector<vector<float>> lattice(n,vector<float>(n));


float random_range(float min, float max) {
  return min + (max - min) * ((float)rand() / RAND_MAX);
}

float gauss_number() {
    static std::mt19937 rng(std::random_device{}());
    static std::normal_distribution<float> nd(0.0, 1.0);
    return nd(rng);
}


Brain random_brain(int eye_number, int N_layers,int layer_size,int output_number) {
  vector<float> temp;
  int brain_number = output_number*(7+(3*eye_number))*pow(layer_size,2+ 2*(N_layers));
  for(int i =0; i < brain_number; i++) {
    temp.push_back(random_range(-1.0,1.0));
  }
  return Brain(temp);
}

vector<Creature> initialize_creatures(int n,float scale) {
    vector<Creature> temp; 
  for(int i =0; i<n;i++) {
   float eye_probability = 0.5f;
   int eyenumber = 1;
   Creature temp_creature({scale*(float)rand()/RAND_MAX,scale*(float)rand()/RAND_MAX} //position
                           ,{0.0f,0.0f} //velocity
                           ,max(0.1f,abs(gauss_number())) //mass
                           ,0.0f  //rotation
                           ,{random_range(0.0,1.0),random_range(0.0,1.0),random_range(0.0,1.0)}   //color 
                           ,{Eye(random_range(0.5f,5.0f),random_range(0.0f,2*M_PI))},  //eyes
                            Brain());  //empty brain
  
    while(random_range(0.0f,1.0f) < eye_probability) {
        temp_creature.eyes.push_back(Eye(random_range(0.5f,5.0f),random_range(0.0f,2*M_PI)));
        eye_probability /= 2.0;
        eyenumber++;
    }
    temp_creature.brain = random_brain(eyenumber,N_layers,Layer_size,4);
    temp_creature.world_size = scale;
    temp_creature.tile_size = 1.0;
    temp.push_back(temp_creature);

  }
  return temp;
}


vector<float> creature_body_model_generator(Creature& c){ 
  float l = c.radius;
    glm::vec3 Co = c.col;
    array<float,2> Cpos = c.position;
  vector<float> temp =
    {Cpos[0] ,Cpos[1] + l,0.0f,Co[0],Co[1],Co[2],
      Cpos[0],Cpos[1],0.5f*c.radius,

     Cpos[0] + l*triangle_scale_factor,Cpos[1] - l*0.5f, 0.0f,Co[0],Co[1],Co[2],
      Cpos[0],Cpos[1],0.5f*c.radius,

     Cpos[0] -l*triangle_scale_factor,Cpos[1] -l*0.5f, 0.0f,Co[0],Co[1],Co[2],
     Cpos[0],Cpos[1],0.5f*c.radius};
    return temp;
}

vector<float> lattice_model_generator(const vector<vector<float>>& lattice, float tile_scale) {
    int n = lattice.size();
    vector<float> temp;

    for (int y = 0; y < n - 1; y++) {
        for (int x = 0; x < n - 1; x++) {

            float c = lattice[y][x];

            float x0 = x * tile_scale;
            float y0 = y * tile_scale;
            float x1 = (x + 1) * tile_scale;
            float y1 = (y + 1) * tile_scale;

            float green = c;
            float blue = (c<0) ? 1.0f : 0.0f;
            temp.insert(temp.end(), {
                x0, y0, 0.0f,  0.0f, green, blue,
                x1, y0, 0.0f,  0.0f, green, blue,
                x0, y1, 0.0f,  0.0f, green, blue
            });
            temp.insert(temp.end(), {
                x1, y0, 0.0f,  0.0f, green, blue,
                x1, y1, 0.0f,  0.0f, green, blue,
                x0, y1, 0.0f,  0.0f, green, blue
            });
        }
    }

    return temp;
}

vector<float> eye_array_builder(vector<Creature>& creatures){
    vector<float> eye_arr;
    for(Creature &C : creatures){
      vector<float> current = C.get_eye_arrays();
      eye_arr.insert(eye_arr.end(),current.begin(),current.end());
    }
    return eye_arr;
}

void generate_creature_buffers(unsigned int& body_VAO,unsigned int& body_VBO,unsigned int& eye_VAO, unsigned int& eye_VBO, vector<Creature>& creatures) {
    int N = creatures.size();

    glBindVertexArray(body_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, body_VBO);
    glBufferData(GL_ARRAY_BUFFER, 27 * sizeof(float)* N, NULL, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(8*sizeof(float)));
    glEnableVertexAttribArray(3);

    glBindVertexArray(eye_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, eye_VBO);
    glBufferData(GL_ARRAY_BUFFER, eye_array_builder(creatures).size() * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
}

int main()
{
  //settings 
  //
glfwSwapInterval(0); //vsync 0=off 1 = on
                     

srand(time(NULL));

   creatures =  initialize_creatures(n_creatures,n*tile_size);

    initialize_model(lattice); 
    evolve_model(lattice, 1.0f,(n*n*n)/2); // generate terrain

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "main", NULL, NULL);
    if (window == NULL)
    {
        cout << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        cout << "Failed to initialize GLAD" << endl;
        return -1;
    }

    //Shaders 
    //
    //
    //
    Shader ourShader("include/shader/default.vert", "include/shader/default.frag");
    Shader CircleShader("include/shader/circle.vert", "include/shader/circle.frag");

    vector<float> vertices = lattice_model_generator(lattice,tile_size); 

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*vertices.size(),vertices.data(),GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);


    unsigned int VBO_circle, VAO_circle;
    glGenVertexArrays(1, &VAO_circle);
    glGenBuffers(1, &VBO_circle);
  
    unsigned int VBO_lines, VAO_lines;
    glGenVertexArrays(1, &VAO_lines);
    glGenBuffers(1, &VBO_lines);

    generate_creature_buffers(VAO_circle,VBO_circle,VAO_lines,VBO_lines,creatures);

    Camera main_camera(1.0f,SCR_WIDTH, SCR_HEIGHT, glm::vec3(0.0f, 0.0f, 0.0f));
  
    //Uniform locations 
    //
    int center_location = glGetUniformLocation(CircleShader.ID,"center"); 
    int radius_location = glGetUniformLocation(CircleShader.ID,"radius");

    //compute Shaders
    //

    ComputeShader computeShaderSource("include/shader/neural_network.glsl");

    //rendering loop!!! 
    //

    double previousTime = glfwGetTime();
    int frameCount = 0;

    while (!glfwWindowShouldClose(window))
    {
      //compute Shaders
      //
      computeShaderSource.use();
      glDispatchCompute(creatures.size(),1,1);
      glMemoryBarrier(GL_ALL_BARRIER_BITS);

      double currentTime = glfwGetTime();
      frameCount++;
      if (currentTime - previousTime >= 1.0) {
        std::cout << "FPS: " << frameCount << std::endl;
        frameCount = 0;
        previousTime = currentTime;
        Creature temp_creature({n*tile_size*(float)rand()/RAND_MAX,n*tile_size*(float)rand()/RAND_MAX} //position
                                ,{1.0f,1.0f} //velocity
                                ,max(0.1f,abs(gauss_number())) //mass
                                ,0.0f  //rotation
                                ,{random_range(0.0,1.0),random_range(0.0,1.0),random_range(0.0,1.0)}   //color 
                                ,{Eye(random_range(0.5f,5.0f),random_range(0.0f,2*M_PI))},  //eyes
                                 Brain());  //empty brain
        creatures.push_back(temp_creature);

    generate_creature_buffers(VAO_circle,VBO_circle,VAO_lines,VBO_lines,creatures);
      }

      processInput(window);

      glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);

      main_camera.Inputs(window);
      main_camera.updateMatrix((float)SCR_WIDTH,(float)SCR_HEIGHT,-0.1f,100.0f);

      //background
      //
      ourShader.use();

      vertices = lattice_model_generator(lattice, tile_size);


      glBindBuffer(GL_ARRAY_BUFFER, VBO);
      glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());

      main_camera.Matrix(ourShader,"camMatrix");

      glBindVertexArray(VAO);
      glDrawArrays(GL_TRIANGLES, 0, vertices.size()/6);
    
      //creatures
      //

      CircleShader.use();

      main_camera.Matrix(CircleShader,"camMatrix");

      //creature loop
      //
      vector<float> creature_body_vertices;
      for(Creature& C : creatures){
        C.evolve_position(0.01,1.0,1.0);
        C.eat(lattice);
      
        vector<float> this_triangle = creature_body_model_generator(C);
        creature_body_vertices.insert(creature_body_vertices.end(),this_triangle.begin(),this_triangle.end());
      }
      glBindBuffer(GL_ARRAY_BUFFER, VBO_circle);
      glBufferSubData(GL_ARRAY_BUFFER, 0, creature_body_vertices.size()*sizeof(float), creature_body_vertices.data());
      glBindVertexArray(VAO_circle);
      glDrawArrays(GL_TRIANGLES, 0, (creature_body_vertices.size()/9));

      //eyes
      //
      ourShader.use();

      main_camera.Matrix(ourShader,"camMatrix");

      vector<float> eye_array = eye_array_builder(creatures);

      glBindBuffer(GL_ARRAY_BUFFER, VBO_lines);
      glBufferSubData(GL_ARRAY_BUFFER, 0, eye_array.size()*sizeof(float), eye_array.data());
      glBindVertexArray(VAO_lines);
      glDrawArrays(GL_LINES, 0, eye_array.size()/6);




      glfwSwapBuffers(window);
      glfwPollEvents();
      }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}
