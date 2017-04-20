#pragma once

#include <CL/cl.hpp>
#include <opencv2/opencv.hpp>

#define DEBUG_RESOLUTION 0

#if DEBUG_RESOLUTION
constexpr size_t WIDTH = 1024;
constexpr size_t HEIGHT = 1024;

#else
constexpr size_t WIDTH = 320;
constexpr size_t HEIGHT = 240;
#endif

namespace MPBAS
{

#define DEBUG_MODE 0
constexpr int N = 20;

class PBASImpl
{
  public:
    PBASImpl();
    ~PBASImpl() = default;
    void calculate_I_and_M(const cv::Mat I, std::vector<cv::Mat> &feature);

    float distance(float &I_i, float &I_m, float &B_i, float &B_m,
                   float alpha = 0, float avarage_m = 1);

    void process(cv::Mat src, cv::Mat &mask);

  private:
    int _index;
    cv::RNG randomGenerator;
    /*
    // ===========================
    //
    // NOTE: OpenCL 1.2
    //
    */

    cl::Context m_context;
    cl::Platform m_platform;
    cl::Device m_device;
    cl::CommandQueue m_queue;
    cl::Program m_program;
    int m_cl_index;
    cl_float m_cl_avrg_Im;

    cl::Kernel m_cl_fill_R_T_kernel;
    cl::Kernel m_cl_magnitude_kernel;
    cl::Kernel m_cl_average_Im_kernel;
    cl::Kernel m_cl_pbas_part1_kernel;
    cl::Kernel m_cl_pbas_part2_kernel;
    cl::Kernel m_cl_update_T_R_kernel;

    // ==========================
    // image

    // buffer
    cl::Buffer m_cl_mem_I;
    cl::Buffer m_cl_mem_feature;
    cl::Buffer m_cl_mem_avrg_Im;
    cl::Buffer m_cl_mem_index_r;
    cl::Buffer m_cl_mem_D[N];
    cl::Buffer m_cl_mem_M[N];
    cl::Buffer m_cl_mem_mask;
    cl::Buffer m_cl_mem_avrg_d;
    cl::Buffer m_cl_mem_R;
    cl::Buffer m_cl_mem_T;
    cl::Buffer m_cl_mem_random_numbers;

    // ==========================
    // ocl function

    void create_kernels();
    void create_buffers();

    void set_args();

    void set_arg_fill_R_T_kernel(cl_kernel &cl_fill_R_T_kernel, cl_mem &mem_T,
                                 const cl_uint &width, const cl_uint &height,
                                 cl_mem &mem_R, cl_int T, cl_int R);

    void set_arg_magnitude_kernel(cl_kernel &cl_magnitude_kernel, cl_mem &src,
                                  cl_uint width, cl_uint height, cl_mem &mag);

    void set_arg_average_Im(cl_kernel &cl_average_Im, cl_mem &mem_Im,
                            cl_uint width, cl_uint height, cl_mem &mem_avrg_Im);

    void set_arg_pbas_part1(cl_kernel &cl_pbas_part1, cl_mem &mem_feature,
                            const int &width, const int &height, cl_mem &mem_R,
                            cl_mem &mem_D, cl_mem &mem_M, cl_mem &mem_index_r,
                            cl_float average_mag);

    void set_arg_pbas_part2(cl_kernel &cl_pbas_part2, cl_mem &mem_feature,
                            const int width, const int height, cl_mem &mem_R,
                            cl_mem &mem_T, cl_mem &mem_index_r, cl_uint min_v,
                            int &cl_index, const int model_size,
                            cl_mem &mem_mask, cl_mem &mem_avrg_d,
                            cl_mem &mem_rand);

    void set_arg_update_R_T(cl_kernel &cl_update_R_T, cl_mem &mem_mask,
                            const int width, const int height, cl_mem &mem_R,
                            cl_mem &mem_T, cl_mem &mem_avrg_d);
};
};
