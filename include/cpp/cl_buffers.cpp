#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/opencl.hpp>
#include <vector>
#include <header/creature.h>
#include <iostream>
#include <header/config.h>

void init_opencl_color_buffers(cl_mem &input_buffer, cl_mem &output_buffer,cl_mem &lattice_buffer,cl_context &context,cl_int &err,MainConfig &config) {
  int MAX_CREATURES = config.worldConfig.MaxCreatures;

    int input_size  =  (MAX_CREATURES * (28));
    int lattice_input_size = config.worldConfig.L_sq();
    int output_size =  MAX_CREATURES * (28);

  input_buffer  = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*input_size,  NULL, &err);
  lattice_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*lattice_input_size,  NULL, &err);
  output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*output_size, NULL, &err);
}

void init_opencl_brain_buffers(cl_mem &creature_input_buffer, cl_mem &true_output_buffer,cl_mem &weight_buffer,cl_context &context,cl_int &err,MainConfig &config) {
    int MAX_CREATURES = config.worldConfig.MaxCreatures;
    
    int input_size  =  (MAX_CREATURES * (32));
    int brain_size =   MAX_CREATURES*config.brainConfig.brain_number(); 
    int output_size =  MAX_CREATURES * (4);

  creature_input_buffer  = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*input_size,  NULL, &err);
  weight_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*brain_size,  NULL, &err);
  true_output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*output_size, NULL, &err);
}

void create_brain_input_buffer(cl_mem &color_output_buffer, std::vector<Creature> &creatures,std::vector<int> &indices, cl_mem &brain_input_buffer, cl_int &err, cl_command_queue &queue) {
    
    int num_creatures = creatures.size();
    int num_entries = indices.size();

    std::vector<float> result(4 * num_entries);
    err = clEnqueueReadBuffer(queue, color_output_buffer, CL_TRUE, 0, sizeof(float) * result.size(), result.data(), 0, NULL, NULL);

    std::vector<float> brain_input(32 * num_creatures, 0.0f);

    for(int i = 0; i < num_creatures; i++) {
        Creature &C = creatures[i];
        brain_input[32 * i]     = C.energy;
        brain_input[32 * i + 1] = C.velocity[0];
        brain_input[32 * i + 2] = C.velocity[1];
        brain_input[32 * i + 3] = C.rotation;
    }

    std::vector<int> eye_counter(num_creatures, -1);

    for(int i = 0; i < num_entries; i++) {
        int creature_idx = indices[i];
        int j = 4 * i;

        if(creature_idx < 0 || creature_idx >= num_creatures) continue;

        eye_counter[creature_idx]++;

        if(eye_counter[creature_idx] == 0) continue;

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

void set_brain_input_kernel(cl_int &err,cl_kernel &kernel,cl_mem &weights,cl_mem &inputs,cl_mem &output,MainConfig &config){
  clSetKernelArg(kernel,0,sizeof(cl_mem),&inputs);
  clSetKernelArg(kernel,1,sizeof(cl_mem),&weights);
  clSetKernelArg(kernel,2,sizeof(int),&config.brainConfig.N_layers);
  clSetKernelArg(kernel,3,sizeof(int),&config.brainConfig.layer_size);
  clSetKernelArg(kernel,4,sizeof(int),&config.brainConfig.input_size);
  clSetKernelArg(kernel,5,sizeof(cl_mem),&output);
}

void generate_colors(cl_mem& input_buffer,cl_mem &lattice_input_buffer,cl_mem &output_buffer,cl_int &err, cl_command_queue &queue,
                                   std::vector<float>& positions,std::vector<float>& lattice,cl_kernel &kernel,cl_device_id &device,MainConfig &config){

  update_opencl_input_buffers(input_buffer,lattice_input_buffer,err,queue,positions,lattice);

  int c_size = positions.size()/4;//this sets the number of sets of points the gpu sees, since this buffer is shared by opengl, this number is 4 
                                  //2 point for initial position and 2 for final

  set_input_kernel(err,kernel,input_buffer,lattice_input_buffer,c_size,config.worldConfig.L,output_buffer);

  size_t global = c_size;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
  clFinish(queue);
}

void fullpass(cl_kernel &kernel,cl_kernel &NN_kernel,cl_device_id &device, cl_int &err, cl_command_queue &queue,
    cl_mem& input_buffer,cl_mem &lattice_input_buffer,cl_mem &output_buffer, cl_mem &weights, cl_mem &brain_input_buffer,cl_mem &true_output_buffer,
    std::vector<float>& positions,std::vector<float>& lattice,std::vector<int> indices,std::vector<Creature> &creatures,MainConfig &config){

    generate_colors(input_buffer,lattice_input_buffer,output_buffer,err,queue,positions,lattice,kernel,device,config);

    upload_weights(queue,weights,err,creatures,config);
    create_brain_input_buffer(output_buffer,creatures,indices,brain_input_buffer,err,queue);

    set_brain_input_kernel(err,NN_kernel,weights,brain_input_buffer,true_output_buffer,config);
  size_t global = creatures.size();
  err = clEnqueueNDRangeKernel(queue, NN_kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
  clFinish(queue);
}

