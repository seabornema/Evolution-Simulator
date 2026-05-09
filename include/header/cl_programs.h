#ifndef CLPROGRAMS 
#define CLPROGRAMS

#include <string>
#include <CL/opencl.hpp>


auto load_kernel = [](const char* path) {
  std::ifstream f(path);
    return std::string(std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>{});
};

cl_program createclProgram(cl_context &context,cl_int &err,const char* path) {
  std::string input_src = load_kernel(path);
    const char* input_src_ptr = input_src.c_str();
    return clCreateProgramWithSource(context, 1, &input_src_ptr, NULL, &err);
}

#endif
