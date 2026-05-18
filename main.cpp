#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <header/shader.h>
#include <header/camera.h>
#include <header/ising.h>
#include <header/creature.h>
#include <header/randomizers.h>
#include <header/config.h>
#include <header/quadtrees.h>
#include <header/cl_buffers.h>
#include <header/cl_programs.h>
#include <header/gl_helpers.h>

#include <iostream>
#include <vector>
#include <random>
#include <array>
#include <ctime>
#include <string>

using namespace std;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

MainConfig mainConfig;

float zoom_scale = 100;




const char* input_shader_source = "include/shader/get_inputs.cl";
const char* neural_network_shader_source = "include/shader/brain.cl";


vector<Creature> creatures;

vector<float> lattice(mainConfig.worldConfig.L_sq());

vector<Creature> initialize_creatures(MainConfig &config) {
    vector<Creature> temp; 
  for(int i =0; i<config.worldConfig.initial_creature_count;i++) {
    temp.push_back(random_creature(mainConfig));
  }
  return temp;
}


void create_fullpass_output(vector<float> &destination,cl_mem &true_output_buffer,cl_command_queue &queue,int csize){
destination.resize(8*csize);
clEnqueueReadBuffer(queue, true_output_buffer, CL_TRUE, 0, sizeof(float) * destination.size(), destination.data(), 0, NULL, NULL);
}

int main()
{
  int width = mainConfig.graphicsConfig.width;
  int height = mainConfig.graphicsConfig.height;
  int L = mainConfig.worldConfig.L;
  //settings 
  //

srand(time(NULL));

//opencl setup
//

cl_int err;
cl_uint num_devices_returned;
cl_device_id devices[1];//my laptop only has one device,
err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU,1,&devices[0], &num_devices_returned);

cl_context context;
context = clCreateContext(0,1,devices,NULL,NULL,&err);

cl_program input_handler = createclProgram(context,err,input_shader_source); 
err = clBuildProgram(input_handler, 1, devices, NULL, NULL, NULL);

cl_program neural_network_handler = createclProgram(context,err,neural_network_shader_source); 
err = clBuildProgram(neural_network_handler, 1, devices, NULL, NULL, NULL);

cl_kernel input_kernel = clCreateKernel(input_handler,"generate_color_data",&err);
cl_kernel neural_network_kernel = clCreateKernel(neural_network_handler,"forward_pass",&err);

cl_command_queue queue_gpu;
queue_gpu = clCreateCommandQueue(context,devices[0],0,&err);

cl_mem position_input_buffer;
cl_mem lattice_input_buffer;
cl_mem output_color_buffer;

cl_mem weight_buffer;
cl_mem brain_input_buffer;
cl_mem true_output_buffer;
cl_mem cost_function_buffer;

init_opencl_color_buffers(position_input_buffer,output_color_buffer, lattice_input_buffer, context, err,mainConfig);
init_opencl_brain_buffers(brain_input_buffer,true_output_buffer, weight_buffer,cost_function_buffer ,context, err,mainConfig,queue_gpu);

creatures =  initialize_creatures(mainConfig);

initialize_model(lattice); 
evolve_model(lattice, 4.0f,pow(L,3),L); // generate terrain
                                         //
                                         //
                                         //

vector<float> init_position_array;
vector<int> init_index_array;
position_array_builder(creatures,init_position_array,init_index_array);

fullpass(input_kernel,neural_network_kernel,devices[0],err,queue_gpu,position_input_buffer
    ,lattice_input_buffer,output_color_buffer,weight_buffer,brain_input_buffer,cost_function_buffer
    ,true_output_buffer,init_position_array,lattice,init_index_array,creatures,mainConfig);





    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    GLFWwindow* window = glfwCreateWindow(width, height, "main", NULL, NULL);
    if (window == NULL)
    {
        cout << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

glfwSwapInterval(0); //vsync 0=off 1 = on
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        cout << "Failed to initialize GLAD" << endl;
        return -1;
    }

    //Shaders 
    //
    //
    Shader ourShader("include/shader/default.vert", "include/shader/default.frag");
    Shader LinesShader("include/shader/lines.vert", "include/shader/lines.frag");
    Shader CircleShader("include/shader/circle.vert", "include/shader/circle.frag");


    Camera main_camera(1.0f,width, height, glm::vec3(0.0f,0.0f,0.0f));


    vector<float> vertices = quad_generator(mainConfig); 
    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*vertices.size(),vertices.data(),GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    unsigned int texture;
    glGenTextures(1, &texture);   
    glBindTexture(GL_TEXTURE_2D, texture);
    vector<unsigned char> texData(3*width*height);
    makeRGBTexture(texData,lattice,width,height,main_camera,mainConfig);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texData.data());
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    unsigned int VBO_circle, VAO_circle;
    glGenVertexArrays(1, &VAO_circle);
    glGenBuffers(1, &VBO_circle);
  
    unsigned int VBO_lines, VAO_lines;
    glGenVertexArrays(1, &VAO_lines);
    glGenBuffers(1, &VBO_lines);


    generate_creature_buffers(VAO_circle,VBO_circle,VAO_lines,VBO_lines,creatures.size(),init_position_array);

  
    grow_berries(lattice,L,10);
    //rendering loop!!! 
    //

    double previousTime = glfwGetTime();
    int frameCount = 0;
      




    vector<float> position_array;
      vector<int> index_array;

    while (!glfwWindowShouldClose(window))
    {

      quadtree qt = build(creatures.begin(), creatures.end());
      collide_all(qt, creatures.begin(), creatures.end());
      //background
      //
      //
      //
      regrow_grass(lattice,0.00026f); 

      double currentTime = glfwGetTime();
      frameCount++;




      processInput(window);

      glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);

      main_camera.Inputs(window,mainConfig);
      main_camera.updateMatrix((float)width,(float)height,-0.1f,100.0f);




      //background
      //

     
      ourShader.use();
      glActiveTexture(GL_TEXTURE0);
      makeRGBTexture(texData,lattice,width,height,main_camera,mainConfig);
      glBindTexture(GL_TEXTURE_2D, texture);

      main_camera.Matrix(ourShader,"camMatrix");
      glUniform1i(glGetUniformLocation(ourShader.ID, "tex"), 0);
      glBindVertexArray(VAO);


      glBindTexture(GL_TEXTURE_2D, texture);
      glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE,texData.data());


      glDrawArrays(GL_TRIANGLE_STRIP, 0, vertices.size()/5);
      //creatures
      //







      for(int i =creatures.size()-1;i>= 0;i--){
        float ec = creatures[i].energy;
        if(ec <= 0) {
          creatures.erase(creatures.begin()+i);
          continue;
        }

      }
      position_array_builder(creatures,position_array,index_array);
      fullpass(input_kernel,neural_network_kernel,devices[0],err,queue_gpu,position_input_buffer,lattice_input_buffer,output_color_buffer,weight_buffer,brain_input_buffer,cost_function_buffer,
      true_output_buffer,position_array,lattice,index_array,creatures,mainConfig);
      vector<float> instructions;
      create_fullpass_output(instructions,true_output_buffer,queue_gpu,creatures.size());



      vector<float> creature_body_vertices;
      vector<Creature> new_creatures;
      for(int i =0;i<creatures.size();i++) {
        
        int j = (8*i);
        Creature &C = creatures[i];
        vector<float> inputs(4);
        inputs[0] = instructions[j];
        inputs[1] = instructions[j+1];
        inputs[2] = instructions[j+2];
        inputs[3] = instructions[j+3];

        C.time_evolve(inputs,new_creatures,lattice,mainConfig.worldConfig.L);
   
      
        vector<float> this_triangle = creature_body_model_generator(C,main_camera);
        creature_body_vertices.insert(creature_body_vertices.end(),this_triangle.begin(),this_triangle.end());
      }


      CircleShader.use();

      main_camera.Matrix(CircleShader,"camMatrix");


      glBindBuffer(GL_ARRAY_BUFFER, VBO_circle);
      glBufferData(GL_ARRAY_BUFFER, creature_body_vertices.size()*sizeof(float), creature_body_vertices.data(),GL_DYNAMIC_DRAW);
      glBindVertexArray(VAO_circle);
      glDrawArrays(GL_TRIANGLES, 0, (creature_body_vertices.size()/10));
     





      creatures.insert(creatures.end(),new_creatures.begin(),new_creatures.end());
      if(mainConfig.worldConfig.random_spawn_rate > random_range(0,1.0)) {
        cout << "random_spawn" <<endl;
        creatures.push_back(random_creature(mainConfig));
      }









     


      //eyes
      //
      LinesShader.use();

      main_camera.Matrix(LinesShader,"camMatrix");


      vector<float> eye_array;
      gl_position_array_builder(creatures,eye_array,main_camera);
      glBindBuffer(GL_ARRAY_BUFFER, VBO_lines);
      glBindVertexArray(VAO_lines);
      glBufferData(GL_ARRAY_BUFFER, eye_array.size()*sizeof(float), eye_array.data(),GL_DYNAMIC_DRAW);
      glDrawArrays(GL_LINES, 0, eye_array.size()/2);



      //  generate_creature_buffers(VAO_circle, VBO_circle, VAO_lines, VBO_lines, creatures.size(), position_array);

      if (currentTime - previousTime >= 1.0) {
          cout << "FPS: " << frameCount << endl;
          cout<< "Creatures: "<<creatures.size()<<endl;
        frameCount = 0;
        previousTime = currentTime;
      }
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
