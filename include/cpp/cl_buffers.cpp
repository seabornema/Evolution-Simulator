#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/opencl.h>
#include <vector>
#include <header/creature.h>
#include <iostream>
#include <header/config.h>

void init_opencl_color_buffers(cl_mem &input_buffer, cl_mem &output_buffer,cl_mem &lattice_buffer,cl_context &context,cl_int &err,MainConfig &config) {
  int MAX_CREATURES = config.worldConfig.MaxCreatures;
  //2 position elements per point, 1 body 1 eye + n eyes = 2+ n*2 max size = 14, 16 for power of 2
//output for 7 points is 14
    int input_size  =  (MAX_CREATURES * (config.brainConfig.input_size));
    int lattice_input_size = config.worldConfig.L_sq();
    int output_size =  MAX_CREATURES * (2*config.creatureConfig.max_eyes);

  input_buffer  = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*input_size,  NULL, &err);
  lattice_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*lattice_input_size,  NULL, &err);
  output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*output_size, NULL, &err);
}

void init_opencl_brain_buffers(cl_mem &creature_input_buffer,
                               cl_mem &true_output_buffer,
                               cl_mem &weight_buffer,
                               cl_mem &cost_function_buffer,
                               cl_context &context,
                               cl_int &err,
                               MainConfig &config,
                               cl_command_queue &queue) {
  
//creature inputs: energy, vx,vy,rotation, const ,mem1,mem2,mem3,mem4, bodycolor, n*eyecolor
// this is 11 guarunteed inputs + 2*n extra with a max of n = 6,so max input size of 23
// outputs are move,turn,eat,addforce, mem1,mem2,mem3,mem4
  int MAX_CREATURES = config.worldConfig.MaxCreatures;
    
    int input_size  =  (MAX_CREATURES * (config.brainConfig.input_size));
    int output_size =  MAX_CREATURES * (config.brainConfig.output_size);

    int brain_size =   MAX_CREATURES*config.brainConfig.brain_number(); 
    int cost_function_size = MAX_CREATURES*config.brainConfig.cost_function_size(); 

  creature_input_buffer  = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*input_size,  NULL, &err);
  weight_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*brain_size,  NULL, &err);
  true_output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*output_size, NULL, &err);
  cost_function_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*cost_function_size, NULL, &err);


  float zero = 0.0f;
  size_t bytes =output_size * sizeof(cl_float);
  err = clEnqueueFillBuffer(queue,true_output_buffer,&zero,sizeof(float),0,bytes,0,NULL,NULL);

}

void create_brain_input_buffer(BrainConfig &brainConfig,
                              cl_mem &color_output_buffer, 
                              std::vector<Creature> &creatures,
                              std::vector<int> &indices, 
                              cl_mem &brain_input_buffer,
                              cl_mem &cost_function_input_buffer,
                              cl_mem &previous_output_buffer, 
                              cl_int &err, 
                              cl_command_queue &queue) {
    

    int input_size = brainConfig.input_size;
    int output_size= brainConfig.output_size;


    int num_creatures = creatures.size();
    int num_entries = indices.size();

    std::vector<float> color_result(2 * num_entries);
    err = clEnqueueReadBuffer(queue, color_output_buffer, CL_TRUE, 0, sizeof(float) * color_result.size(), color_result.data(), 0, NULL, NULL);

    std::vector<float> output_result(output_size * num_creatures);
    err = clEnqueueReadBuffer(queue, previous_output_buffer, CL_TRUE, 0, sizeof(float) * output_result.size(), output_result.data(), 0, NULL, NULL);

    std::vector<float> brain_input(input_size * num_creatures, 0.0f);

    for(int i = 0; i < num_creatures; i++) {
      int j = input_size*i;
        Creature &C = creatures[i];

      float xhat =  std::cos(C.rotation);
      float yhat =  std::sin(C.rotation);
        brain_input[j]     = C.energy/C.config->initial_energy;
        brain_input[j + 1] = std::tanh(C.velo.x*xhat + C.velo.y*yhat);
        brain_input[j + 2] = output_result[2+(output_size*i)];
        brain_input[j + 3] = 1.0f;
        
    }

    std::vector<int> eye_counter(num_creatures, -1);


    for(int i = 0; i < num_entries; i++) {
        int creature_idx = indices[i];
        int j = 2 * i;

        if(creature_idx < 0 || creature_idx >= num_creatures) continue;

        eye_counter[creature_idx]++;


        int slot = eye_counter[creature_idx];
        if(slot== 0) {
          brain_input[input_size * creature_idx + 4] = color_result[j];
          brain_input[input_size * creature_idx + 5] = color_result[j + 1];
          continue;
        }
        int eye_slot = slot - 1; 
        if(eye_slot >= 2) continue;
        int index = input_size * creature_idx + 6 + (eye_slot * 2);

        if(index + 1 >= (int)brain_input.size()) continue;

        brain_input[index]     = color_result[j];
        brain_input[index + 1] = color_result[j + 1];
    }

    std::vector<float> cost_function_input;
    cost_function_input.reserve(input_size*output_size*num_creatures);
    for (Creature &C : creatures) {
      std::vector<float> &cost = C.brain.cost_function;
      cost_function_input.insert(cost_function_input.end(),cost.begin(),cost.end());
    }

    
    err = clEnqueueWriteBuffer(queue, cost_function_input_buffer, CL_TRUE, 0, sizeof(float) * cost_function_input.size(), cost_function_input.data(), 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, brain_input_buffer, CL_TRUE, 0, sizeof(float) * brain_input.size(), brain_input.data(), 0, NULL, NULL);
}


void upload_weights(cl_command_queue &queue,cl_mem &weights,cl_int &err,std::vector<Creature> &creatures,MainConfig &config) {

    std::vector<float> buffer;
    buffer.reserve(creatures.size() * config.brainConfig.brain_number());

    for (Creature &C : creatures) {
        std::vector<float>& w = C.brain.neurons;
        buffer.insert(buffer.end(), w.begin(), w.end());
    }

    err= clEnqueueWriteBuffer(queue, weights,CL_TRUE,0,sizeof(float) * buffer.size(),buffer.data(),0,NULL,NULL);
}

void update_opencl_input_buffers(cl_mem& input_buffer,cl_mem &lattice_input_buffer,cl_int &err, cl_command_queue &queue,
                                 std::vector<float>& positions,std::vector<float>& lattice) {
    int input_size = ((positions.size()));

    err = clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(float)*input_size, positions.data(), 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, lattice_input_buffer, CL_TRUE, 0, sizeof(float)*lattice.size(), lattice.data(), 0,NULL,NULL);
}

void set_input_kernel(cl_int &err,cl_kernel &kernel,cl_mem &position_input,cl_mem &lattice_input,int creatures_size,int lattice_size,cl_mem &output){
  clSetKernelArg(kernel,0,sizeof(cl_mem),&position_input);
  clSetKernelArg(kernel,1,sizeof(cl_mem),&lattice_input);
  clSetKernelArg(kernel,2,sizeof(int),&creatures_size);
  clSetKernelArg(kernel,3,sizeof(int),&lattice_size);
  clSetKernelArg(kernel,4,sizeof(cl_mem),&output);
}

void set_brain_input_kernel(cl_int &err,cl_kernel &kernel,cl_mem &weights,cl_mem &inputs,cl_mem &output,MainConfig &config,int creatures_size,cl_mem &cost_functions){

  clSetKernelArg(kernel,0,sizeof(cl_mem),&inputs);
  clSetKernelArg(kernel,1,sizeof(cl_mem),&weights);
  clSetKernelArg(kernel,2,sizeof(int),&config.brainConfig.N_layers);
  clSetKernelArg(kernel,3,sizeof(int),&config.brainConfig.layer_size);
  clSetKernelArg(kernel,4,sizeof(int),&config.brainConfig.input_size);
  clSetKernelArg(kernel,5,sizeof(int),&config.brainConfig.output_size);
  clSetKernelArg(kernel,6,sizeof(int),&creatures_size);
  clSetKernelArg(kernel,7,sizeof(cl_mem),&cost_functions);
  clSetKernelArg(kernel,8,sizeof(cl_mem),&output);

}

void generate_colors(cl_mem& input_buffer,cl_mem &lattice_input_buffer,cl_mem &output_buffer,cl_int &err, cl_command_queue &queue,
                                   std::vector<float>& positions,std::vector<float>& lattice,cl_kernel &kernel,cl_device_id &device,MainConfig &config){

  update_opencl_input_buffers(input_buffer,lattice_input_buffer,err,queue,positions,lattice);

  int c_size = positions.size()/2;//this sets the number of sets of points the gpu sees 2 coordinates per point

  set_input_kernel(err,kernel,input_buffer,lattice_input_buffer,c_size,config.worldConfig.L,output_buffer);

  size_t global = c_size;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
  clFlush(queue);
}

void fullpass(cl_kernel &kernel,cl_kernel &NN_kernel,cl_device_id &device, cl_int &err, cl_command_queue &queue,
    cl_mem& input_buffer,cl_mem &lattice_input_buffer,cl_mem &output_buffer, cl_mem &weights, cl_mem &brain_input_buffer,cl_mem &cost_function_buffer,cl_mem &true_output_buffer,
    std::vector<float>& positions,std::vector<float>& lattice,std::vector<int> indices,std::vector<Creature> &creatures,MainConfig &config){

    generate_colors(input_buffer,lattice_input_buffer,output_buffer,err,queue,positions,lattice,kernel,device,config);

    upload_weights(queue,weights,err,creatures,config);
    create_brain_input_buffer(config.brainConfig,output_buffer,creatures,indices,brain_input_buffer,cost_function_buffer,true_output_buffer,err,queue);

    set_brain_input_kernel(err,NN_kernel,weights,brain_input_buffer,true_output_buffer,config,creatures.size(),cost_function_buffer);
  size_t global = creatures.size();
  err = clEnqueueNDRangeKernel(queue, NN_kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
  clFlush(queue);
}
