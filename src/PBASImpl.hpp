#pragma once

#include <CL/cl.hpp>
#include <opencv2/opencv.hpp>
struct ImageInfo
{
  cl_uint width;
  cl_uint height;

  constexpr ImageInfo(cl_uint width, cl_uint height)
      : width(width), height(height)
  {
  }
};
struct PBASParameter
{
  ImageInfo imageInfo;
  cl_uint modelSize = 20;
  cl_uint minModels = 2;
  cl_uint T_lower = 2;
  cl_uint R_lower = 10;
  cl_uint min_index = 2;
  cl_uint R_scale = 5;
  cl_uint T_inc = 1;
  cl_uint T_upper = 200;
  cl_float min_R = 18.f;
  cl_float R_inc_dec = 0.05f;
  cl_float T_dec = 0.05f;
  cl_float alpha = 10.f;
  cl_float beta = 1.f;
  constexpr PBASParameter(cl_uint width, cl_uint height)
      : imageInfo(width, height)
  {
  }
};

class PBASImpl
{
public:
  PBASImpl(const PBASParameter param);
  ~PBASImpl() = default;

  cv::Mat process(cv::Mat src);

private:
  PBASParameter m_parameters;

  cl_uint m_cl_index = 0;

  cl_float m_cl_avrg_Im = 0.f;

  cv::RNG m_random_generator;
  cv::Mat m_result_mask;

  cl::Context m_context;
  cl::Platform m_platform;
  cl::Device m_device;
  cl::CommandQueue m_queue;
  cl::Program m_program;

  cl::Kernel m_cl_fill_R_T_kernel;
  cl::Kernel m_cl_fill_model_kernel;
  cl::Kernel m_cl_magnitude_kernel;
  cl::Kernel m_cl_average_Im_kernel;
  cl::Kernel m_cl_pbas_kernel;
  cl::Kernel m_cl_update_T_R_kernel;

  // ==========================
  // image

  // buffer
  cl::Buffer m_cl_mem_I;
  cl::Buffer m_cl_mem_feature;
  cl::Buffer m_cl_mem_avrg_Im;
  cl::Buffer m_cl_mem_index_r;

  cl_uint m_model_index;
  cl::Buffer m_cl_mem_D;
  cl::Buffer m_cl_mem_M;

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
};
