
#include <string>
#include <CL/opencl.hpp>


auto load_kernel = [](const char* path) {
    ifstream f(path);
    return string(istreambuf_iterator<char>(f),istreambuf_iterator<char>{});
};

cl_program createclProgram(cl_context &context,cl_int &err,const char* path) {
    string input_src = load_kernel(path);
    const char* input_src_ptr = input_src.c_str();
    return clCreateProgramWithSource(context, 1, &input_src_ptr, NULL, &err);
}
