#ifndef CLBUFFERS 
#define CLBUFFERS

#include <CL/opencl.h>

#include <vector>
#include <header/creature.h>
#include <iostream>
#include <header/config.h>

void init_opencl_color_buffers(cl_mem &input_buffer, cl_mem &output_buffer,cl_mem &lattice_buffer,cl_context &context,cl_int &err,MainConfig &config);


void init_opencl_brain_buffers(cl_mem &creature_input_buffer,
                               cl_mem &true_output_buffer,
                               cl_mem &weight_buffer,
                               cl_mem &cost_function_buffer,
                               cl_context &context,
                               cl_int &err,
                               MainConfig &config,
                               cl_command_queue &queue);

void create_brain_input_buffer(BrainConfig &brainConfig,
                              cl_mem &color_output_buffer, 
                              std::vector<Creature> &creatures,
                              std::vector<int> &indices, 
                              cl_mem &brain_input_buffer,
                              cl_mem &cost_function_input_buffer,
                              cl_mem &previous_output_buffer, 
                              cl_int &err, 
                              cl_command_queue &queue);


void upload_weights(cl_command_queue &queue,cl_mem &weights,cl_int &err,std::vector<Creature> &creatures,MainConfig &config);

void update_opencl_input_buffers(cl_mem& input_buffer,cl_mem &lattice_input_buffer,cl_int &err, cl_command_queue &queue,
                                 std::vector<float>& positions,std::vector<float>& lattice);

void set_input_kernel(cl_int &err,cl_kernel &kernel,cl_mem &position_input,cl_mem &lattice_input,int creatures_size,int lattice_size,cl_mem &output);

void set_brain_input_kernel(cl_int &err,cl_kernel &kernel,cl_mem &weights,cl_mem &inputs,cl_mem &output,MainConfig &config,int creatures_size,cl_mem &cost_functions);

void generate_colors(cl_mem& input_buffer,cl_mem &lattice_input_buffer,cl_mem &output_buffer,cl_int &err, cl_command_queue &queue,
                                   std::vector<float>& positions,std::vector<float>& lattice,cl_kernel &kernel,cl_device_id &device,MainConfig &config);


void fullpass(cl_kernel &kernel,cl_kernel &NN_kernel,cl_device_id &device, cl_int &err, cl_command_queue &queue,
    cl_mem& input_buffer,cl_mem &lattice_input_buffer,cl_mem &output_buffer, cl_mem &weights, cl_mem &brain_input_buffer,cl_mem &cost_function_buffer,cl_mem &true_output_buffer,
    std::vector<float>& positions,std::vector<float>& lattice,std::vector<int> indices,std::vector<Creature> &creatures,MainConfig &config);
#endif

