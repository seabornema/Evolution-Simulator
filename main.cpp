#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
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
#include <string>

using namespace std;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
float zoom_scale = 100;

const int MAX_CREATURES = 1000;
const int n_creatures =200;

//tiling 
const float tile_size = 1.0;
const int n = 120;

//neural_network
//
const int N_layers = 4;
const int Layer_size = 24;
const int important_brain_number = 32;

const char* input_shader_source = "include/shader/get_inputs.cl";
const char* neural_network_shader_source = "include/shader/brain.cl";

const float triangle_scale_factor = sqrt(3)/2;

vector<Creature> creatures;

vector<float> lattice(n*n);


float random_range(float min, float max) {
  return min + (max - min) * ((float)rand() / RAND_MAX);
}

float gauss_number() {
    static mt19937 rng(random_device{}());
    static normal_distribution<float> nd(0.0, 1.0);
    return nd(rng);
}

auto load_kernel = [](const char* path) {
    ifstream f(path);
    return string(istreambuf_iterator<char>(f),istreambuf_iterator<char>{});
};
Brain random_brain(int N_layers,int layer_size) {
  vector<float> temp;
  int brain_number = (4+32)*(layer_size) + (N_layers-1)* (int)pow(layer_size,2); // 4 outputs per creature, 32 inputs per creature
  temp.resize(brain_number);
  for(int i =0; i < brain_number; i++) {
    temp[i] = (gauss_number());
  }
  return Brain(move(temp));
}

Creature random_creature(float world_scale) {
   float eye_probability = 0.5f;
   int eyenumber = 1;
   Creature temp_creature({world_scale*(float)rand()/RAND_MAX,world_scale*(float)rand()/RAND_MAX} //position
                           ,{0.0f,0.0f} //velocity
                           ,max(0.8f,abs(gauss_number())/2) //mass
                           ,0.0f  //rotation
                           ,{random_range(0.0,1.0),random_range(0.0,1.0),random_range(0.0,1.0)}   //color 
                           ,{Eye(random_range(0.5f,2.0f),random_range(0.0f,2*M_PI))},  //eyes
                            Brain());  //empty brain
  
    while(random_range(0.0f,1.0f) < eye_probability && eyenumber <6) {
        temp_creature.eyes.push_back(Eye(random_range(0.5f,3.0f),random_range(0.0f,2*M_PI)));
        eye_probability /= 2.0;
        eyenumber++;
    }
    temp_creature.brain = random_brain(N_layers,Layer_size);
    temp_creature.world_size = world_scale;
    temp_creature.tile_size = 1.0;
    return temp_creature;
}


vector<Creature> initialize_creatures(int n,float scale) {
    vector<Creature> temp; 
  for(int i =0; i<n;i++) {
    temp.push_back(random_creature(scale));
  }
  return temp;
}


vector<float> creature_body_model_generator(Creature& c){ 
  float l = c.radius;
    glm::vec3 Co = c.col;
    array<float,2> Cpos = c.position;
  vector<float> temp =
    {Cpos[0] ,Cpos[1] + l,0.0f,Co[0],Co[1],Co[2],
      Cpos[0],Cpos[1],0.5f*c.radius,c.rotation,

     Cpos[0] + l*triangle_scale_factor,Cpos[1] - l*0.5f, 0.0f,Co[0],Co[1],Co[2],
      Cpos[0],Cpos[1],0.5f*c.radius,c.rotation,

     Cpos[0] -l*triangle_scale_factor,Cpos[1] -l*0.5f, 0.0f,Co[0],Co[1],Co[2],
     Cpos[0],Cpos[1],0.5f*c.radius,c.rotation};
    return temp;
}

vector<float> lattice_model_generator(const vector<float>& lattice, float tile_scale,int n) {
    vector<float> temp;

    for (int y = 0; y < n - 1; y++) {
        for (int x = 0; x < n - 1; x++) {

            float c = lattice[x + ((n)*y)];

            float x0 = x * tile_scale;
            float y0 = y * tile_scale;
            float x1 = (x + 1) * tile_scale;
            float y1 = (y + 1) * tile_scale;

            if(c > 1.0f) {
            temp.insert(temp.end(), {
                x0, y0, 0.0f,  c/50, 0.0f, 0.4f,
                x1, y0, 0.0f,  c/50, 0.0f, 0.4f,
                x0, y1, 0.0f,  c/50, 0.0f, 0.4f
            });
            temp.insert(temp.end(), {
                x1, y0, 0.0f,  c/50, 0.0f, 0.4f,
                x1, y1, 0.0f,  c/50, 0.0f, 0.4f,
                x0, y1, 0.0f,  c/50, 0.0f, 0.4f
            });
            } else{
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
            });}
        }
    }

    return temp;
}
void position_array_builder(vector<Creature>& creatures, vector<float> &eye_arr, vector<int> &index_arr) {
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

void generate_creature_buffers(unsigned int& body_VAO,unsigned int& body_VBO,unsigned int& eye_VAO, unsigned int& eye_VBO,int N,vector<float>& eye_arr) {

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

void init_opencl_color_buffers(cl_mem &input_buffer, cl_mem &output_buffer,cl_mem &lattice_buffer,cl_context &context,cl_int &err) {
    int input_size  =  (MAX_CREATURES * (28));
    int lattice_input_size = n*n; //n from lattice length
    int output_size =  MAX_CREATURES * (28);

  input_buffer  = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*input_size,  NULL, &err);
  lattice_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*lattice_input_size,  NULL, &err);
  output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*output_size, NULL, &err);
}
void init_opencl_brain_buffers(cl_mem &creature_input_buffer, cl_mem &true_output_buffer,cl_mem &weight_buffer,cl_context &context,cl_int &err) {
    int input_size  =  (MAX_CREATURES * (32));
    int lattice_input_size = MAX_CREATURES*((4+32)*(Layer_size) + (N_layers-1)* (int)pow(Layer_size,2)); 
    int output_size =  MAX_CREATURES * (4);

  creature_input_buffer  = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*input_size,  NULL, &err);
  weight_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*lattice_input_size,  NULL, &err);
  true_output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*output_size, NULL, &err);
}

void create_brain_input_buffer(cl_mem &color_output_buffer, vector<Creature> &creatures, vector<int> &indices, cl_mem &brain_input_buffer, cl_int &err, cl_command_queue &queue) {
    
    int num_creatures = creatures.size();
    int num_entries = indices.size();

    vector<float> result(4 * num_entries);
    err = clEnqueueReadBuffer(queue, color_output_buffer, CL_TRUE, 0, sizeof(float) * result.size(), result.data(), 0, NULL, NULL);

    vector<float> brain_input(32 * num_creatures, 0.0f);

    for(int i = 0; i < num_creatures; i++) {
        Creature &C = creatures[i];
        brain_input[32 * i]     = C.energy;
        brain_input[32 * i + 1] = C.velocity[0];
        brain_input[32 * i + 2] = C.velocity[1];
        brain_input[32 * i + 3] = C.rotation;
    }

    vector<int> eye_counter(num_creatures, -1);

    for(int i = 0; i < num_entries; i++) {
        int creature_idx = indices[i];
        int j = 4 * i;

        if(creature_idx < 0 || creature_idx >= num_creatures) continue; // stale index guard

        eye_counter[creature_idx]++;

        if(eye_counter[creature_idx] == 0) continue; // skip body entry

        int eye_slot = eye_counter[creature_idx] - 1;
        if(eye_slot >= 6) continue;

        int index = 32 * creature_idx + 4 + (eye_slot * 4);
        if(index + 3 >= (int)brain_input.size()) continue;

        brain_input[index]     = result[j];
        brain_input[index + 1] = result[j + 1];
        brain_input[index + 2] = result[j + 2];
        brain_input[index + 3] = result[j + 3];
    }

    err = clEnqueueWriteBuffer(queue, brain_input_buffer, CL_TRUE, 0, sizeof(float) * brain_input.size(), brain_input.data(), 0, NULL, NULL);
}
void upload_weights(cl_command_queue &queue,cl_mem &weights,cl_int &err,vector<Creature> &creatures) {
    int brain_number = (4 + 32) * Layer_size + (N_layers - 1) * (int)pow(Layer_size, 2);

    vector<float> buffer;
    buffer.reserve(creatures.size() * brain_number);

    for (Creature &C : creatures) {
        vector<float>& w = C.brain.neurons;
        buffer.insert(buffer.end(), w.begin(), w.end());
    }
    err= clEnqueueWriteBuffer(queue, weights,CL_TRUE,0,sizeof(float) * buffer.size(),buffer.data(),0,NULL,NULL);
}

void update_opencl_input_buffers(cl_mem& input_buffer,cl_mem &lattice_input_buffer,cl_int &err, cl_command_queue &queue,
                                 vector<float>& positions,vector<float>& lattice) {
    int input_size = ((positions.size()));

    err = clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(float)*input_size, positions.data(), 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, lattice_input_buffer, CL_TRUE, 0, sizeof(float)*n*n, lattice.data(), 0,NULL,NULL);

}


void set_input_kernel(cl_int &err,cl_kernel &kernel,cl_mem &position_input,cl_mem &lattice_input,int creatures_size,int lattice_size,cl_mem &output){
  clSetKernelArg(kernel,0,sizeof(cl_mem),&position_input);
  clSetKernelArg(kernel,1,sizeof(cl_mem),&lattice_input);
  clSetKernelArg(kernel,2,sizeof(int),&creatures_size);
  clSetKernelArg(kernel,3,sizeof(int),&lattice_size);
  clSetKernelArg(kernel,4,sizeof(cl_mem),&output);
}

void set_brain_input_kernel(cl_int &err,cl_kernel &kernel,cl_mem &weights,cl_mem &inputs,cl_mem &output){
  clSetKernelArg(kernel,0,sizeof(cl_mem),&inputs);
  clSetKernelArg(kernel,1,sizeof(cl_mem),&weights);
  clSetKernelArg(kernel,2,sizeof(int),&N_layers);
  clSetKernelArg(kernel,3,sizeof(int),&Layer_size);
  clSetKernelArg(kernel,4,sizeof(int),&important_brain_number);
  clSetKernelArg(kernel,5,sizeof(cl_mem),&output);
}

void generate_colors(cl_mem& input_buffer,cl_mem &lattice_input_buffer,cl_mem &output_buffer,cl_int &err, cl_command_queue &queue,
                                   vector<float>& positions,vector<float>& lattice,cl_kernel &kernel,cl_device_id &device){

  update_opencl_input_buffers(input_buffer,lattice_input_buffer,err,queue,positions,lattice);

  int c_size = positions.size()/4;//this sets the number of "creatures" the gpu sees

  set_input_kernel(err,kernel,input_buffer,lattice_input_buffer,c_size,n,output_buffer);

  size_t global = c_size;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
  clFinish(queue);
}





void fullpass(cl_kernel &kernel,cl_kernel &NN_kernel,cl_device_id &device, cl_int &err, cl_command_queue &queue,
    cl_mem& input_buffer,cl_mem &lattice_input_buffer,cl_mem &output_buffer, cl_mem &weights, cl_mem &brain_input_buffer,cl_mem &true_output_buffer,
    vector<float>& positions,vector<float>& lattice,vector<int> indices){

    generate_colors(input_buffer,lattice_input_buffer,output_buffer,err,queue,positions,lattice,kernel,device);

    upload_weights(queue,weights,err,creatures);
    create_brain_input_buffer(output_buffer,creatures,indices,brain_input_buffer,err,queue);

    set_brain_input_kernel(err,NN_kernel,weights,brain_input_buffer,true_output_buffer);
  size_t global = creatures.size();
  err = clEnqueueNDRangeKernel(queue, NN_kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
  clFinish(queue);
}
 
void create_fullpass_output(vector<float> &destination,cl_mem &true_output_buffer,cl_command_queue &queue,int csize){
destination.resize(4*csize);
clEnqueueReadBuffer(queue, true_output_buffer, CL_TRUE, 0, sizeof(float) * destination.size(), destination.data(), 0, NULL, NULL);

}

int main()
{
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

string input_src = load_kernel(input_shader_source);
const char* input_src_ptr = input_src.c_str();
cl_program input_handler = clCreateProgramWithSource(context, 1, &input_src_ptr, NULL, &err);

err = clBuildProgram(input_handler, 1, devices, NULL, NULL, NULL);

string neural_src = load_kernel(neural_network_shader_source);
const char* neural_src_ptr = neural_src.c_str();
cl_program neural_network_handler = clCreateProgramWithSource(context, 1, &neural_src_ptr, NULL, &err);

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

init_opencl_color_buffers(position_input_buffer,output_color_buffer, lattice_input_buffer, context, err);
init_opencl_brain_buffers(brain_input_buffer,true_output_buffer, weight_buffer, context, err);

creatures =  initialize_creatures(n_creatures,n*tile_size);

initialize_model(lattice); 
evolve_model(lattice, 1.0f,(n*n*n)/2,n); // generate terrain
                                         //
                                         //
                                         //

vector<float> init_position_array;
vector<int> init_index_array;
position_array_builder(creatures,init_position_array,init_index_array);

fullpass(input_kernel,neural_network_kernel,devices[0],err,queue_gpu,position_input_buffer,lattice_input_buffer,output_color_buffer,weight_buffer,brain_input_buffer,true_output_buffer,init_position_array,lattice,init_index_array);





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
    //
    Shader ourShader("include/shader/default.vert", "include/shader/default.frag");
    Shader LinesShader("include/shader/lines.vert", "include/shader/lines.frag");
    Shader CircleShader("include/shader/circle.vert", "include/shader/circle.frag");

    vector<float> vertices = lattice_model_generator(lattice,tile_size,n); 

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


    generate_creature_buffers(VAO_circle,VBO_circle,VAO_lines,VBO_lines,creatures.size(),init_position_array);

    Camera main_camera(1.0f,SCR_WIDTH, SCR_HEIGHT, glm::vec3(0.0f, 0.0f, 0.0f));
  


    //rendering loop!!! 
    //

    double previousTime = glfwGetTime();
    int frameCount = 0;

    grow_berries(lattice,n,50);
    while (!glfwWindowShouldClose(window))
    {
      //compute Shaders
      //
      //
      

regrow_grass(lattice,0.001f); 
      double currentTime = glfwGetTime();
      frameCount++;



      if (currentTime - previousTime >= 1.0) {
          cout << "FPS: " << frameCount << endl;
          cout<< "Creatures: "<<creatures.size()<<endl;
        frameCount = 0;
        previousTime = currentTime;
      }

      processInput(window);

      glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);

      main_camera.Inputs(window);
      main_camera.updateMatrix((float)SCR_WIDTH,(float)SCR_HEIGHT,-0.1f,100.0f);




      //background
      //
      ourShader.use();

      vertices = lattice_model_generator(lattice, tile_size,n);

      glBindBuffer(GL_ARRAY_BUFFER, VBO);
      glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());

      main_camera.Matrix(ourShader,"camMatrix");

      glBindVertexArray(VAO);
      glDrawArrays(GL_TRIANGLES, 0, vertices.size()/6);
      //creatures
      //



      vector<float> position_array;
      vector<int> index_array;
      position_array_builder(creatures,position_array,index_array);
    fullpass(input_kernel,neural_network_kernel,devices[0],err,queue_gpu,position_input_buffer,lattice_input_buffer,output_color_buffer,weight_buffer,brain_input_buffer,
    true_output_buffer,position_array,lattice,index_array);
   vector<float> instructions;
   create_fullpass_output(instructions,true_output_buffer,queue_gpu,creatures.size());

      vector<float> creature_body_vertices;
      vector<Creature> new_creatures;
      int csize = creatures.size();
      for(int i =csize-1;i>= 0;i--){

        int j = (4*i);
        Creature &C = creatures[i];

        vector<float> inputs(4);
        inputs[0] = instructions[j];
        inputs[1] = instructions[j+1];
        inputs[2] = instructions[j+2];
        inputs[3] = instructions[j+3];

        C.time_evolve(inputs,new_creatures,lattice,n);
    
        if(C.energy <= 0) {
          creatures.erase(creatures.begin()+i);
          continue;
        }
      
        vector<float> this_triangle = creature_body_model_generator(C);
        creature_body_vertices.insert(creature_body_vertices.end(),this_triangle.begin(),this_triangle.end());

      }
      creatures.insert(creatures.end(),new_creatures.begin(),new_creatures.end());

      CircleShader.use();

      main_camera.Matrix(CircleShader,"camMatrix");

      //creature loop
      //

      glBindBuffer(GL_ARRAY_BUFFER, VBO_circle);
      glBufferSubData(GL_ARRAY_BUFFER, 0, creature_body_vertices.size()*sizeof(float), creature_body_vertices.data());
      glBindVertexArray(VAO_circle);
      glDrawArrays(GL_TRIANGLES, 0, (creature_body_vertices.size()/10));














     


      //eyes
      //
      LinesShader.use();

      main_camera.Matrix(LinesShader,"camMatrix");


      glBindBuffer(GL_ARRAY_BUFFER, VBO_lines);
      glBufferSubData(GL_ARRAY_BUFFER, 0, position_array.size()*sizeof(float), position_array.data());
      glBindVertexArray(VAO_lines);
      glDrawArrays(GL_LINES, 0, position_array.size()/2);



    generate_creature_buffers(VAO_circle, VBO_circle, VAO_lines, VBO_lines, creatures.size(), position_array);

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
