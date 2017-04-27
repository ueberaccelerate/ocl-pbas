#include "PBASImpl.hpp"
#include "Utiles.hpp"

#include <iterator>
#include <vector>

void GetPlatformInfo(cl_platform_id pcl, cl_platform_info param_name)
{
  std::size_t param_size;
  clGetPlatformInfo(pcl, param_name, 0, nullptr, &param_size);
  std::string buf{};
  buf.resize(param_size);
  clGetPlatformInfo(pcl, param_name, param_size, (void *)buf.data(), nullptr);
  std::cout << buf << "\n";
}

template <typename... T>
void GetPlatformInfoList(cl_platform_id pcl, T... param_name)
{
  std::initializer_list<int>{(GetPlatformInfo(pcl, param_name), 0)...};
}

PBASImpl::PBASImpl(const PBASParameter param)
    : m_parameters(param), m_cl_index(0), m_cl_avrg_Im(0.f), m_model_index(0)
{
  std::cout << "PBASImpl()\n";
  utility::timeThis("Create Context time: ", [&]() {
    // m_context = cl::Context{CL_DEVICE_TYPE_CPU};
    // std::vector<cl::Platform> platforms;

    // cl::Platform::get(&platforms);
    // m_platform = platforms[1];
    // m_device = cl::Device::getDefault();
    // m_queue = cl::CommandQueue(m_context);

    m_context = cl::Context::getDefault();
    m_platform = cl::Platform::getDefault();
    m_device = cl::Device::getDefault();
    m_queue = cl::CommandQueue(m_context);

    std::fstream stream{"../src/opencl_kernels.cl", std::ios::in};
    std::string sourceSample =
        std::string((std::istreambuf_iterator<char>(stream)),
                    std::istreambuf_iterator<char>());
    if (!sourceSample.length())
    {
      std::cerr << "OpenCL Source Empty\n";
      std::terminate();
    }
    GetPlatformInfoList(m_platform(), CL_PLATFORM_PROFILE, CL_PLATFORM_VERSION,
                        CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
                        CL_PLATFORM_EXTENSIONS);

    m_program = cl::Program{m_context, sourceSample.data()};
    if (m_program.build() != 0)
    {
      std::string buildLog =
          m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
      std::cerr << buildLog << "\n";
      std::terminate();
    }
  });
  // kernels initialize
  utility::timeThis("Create Kernels time: ", [&]() { create_kernels(); });

  // buffer initialize

  utility::timeThis("Create Buffers time: ", [&]() { create_buffers(); });
}

void PBASImpl::process(cv::Mat src, cv::Mat &mask)
{
  // NOTE: OpenCL work position
  assert(src.type() == CV_8UC1);

  mask = cv::Mat::zeros(m_parameters.imageHeight, m_parameters.imageWidth,
                        CV_8UC1);
  m_queue.enqueueWriteBuffer(m_cl_mem_I, true, 0,
                             m_parameters.imageWidth * m_parameters.imageHeight,
                             src.data);

  set_arg_magnitude_kernel(m_cl_magnitude_kernel(), m_cl_mem_I(),
                           m_parameters.imageWidth, m_parameters.imageHeight,
                           m_cl_mem_feature());
  m_queue.enqueueNDRangeKernel(
      m_cl_magnitude_kernel, cl::NDRange{},
      cl::NDRange{m_parameters.imageWidth, m_parameters.imageHeight});

  if (m_cl_index < m_parameters.modelSize)
  {
    if (m_cl_index == 0)
    {
      set_arg_fill_R_T_kernel(m_cl_fill_R_T_kernel(), m_cl_mem_T(),
                              m_parameters.imageWidth, m_parameters.imageHeight,
                              m_cl_mem_R(), m_parameters.T_lower,
                              m_parameters.R_lower);

      m_queue.enqueueNDRangeKernel(
          m_cl_fill_R_T_kernel, cl::NDRange{},
          cl::NDRange{m_parameters.imageWidth, m_parameters.imageHeight});
    }

    set_arg_fill_model(m_cl_fill_model_kernel(), m_cl_mem_feature(),
                       m_parameters.imageWidth, m_parameters.imageHeight,
                       m_cl_index, m_cl_mem_M());
    m_queue.enqueueNDRangeKernel(
        m_cl_fill_model_kernel, cl::NDRange{},
        cl::NDRange{m_parameters.imageWidth, m_parameters.imageHeight});
    m_cl_index++;
  }

  cl_float average_mag = 0;

  m_queue.enqueueFillBuffer(m_cl_mem_avrg_Im, average_mag, 0, sizeof(cl_float));

  set_arg_average_Im(m_cl_average_Im_kernel(), m_cl_mem_feature(),
                     m_parameters.imageWidth, m_parameters.imageHeight,
                     m_cl_mem_avrg_Im());
  m_queue.enqueueNDRangeKernel(
      m_cl_average_Im_kernel, cl::NDRange{},
      cl::NDRange{m_parameters.imageWidth, m_parameters.imageHeight});

  m_queue.enqueueReadBuffer(m_cl_mem_avrg_Im, true, 0, sizeof(cl_float),
                            &m_cl_avrg_Im);
  m_cl_avrg_Im /= m_parameters.imageWidth * m_parameters.imageHeight;

  set_arg_pbas_part1(m_cl_pbas_part1_kernel(), m_cl_mem_feature(),
                     m_parameters.imageWidth, m_parameters.imageHeight,
                     m_cl_mem_R(), m_cl_index, m_cl_mem_D(), m_cl_mem_M(),
                     m_cl_mem_index_r(), m_cl_avrg_Im);
  m_queue.enqueueNDRangeKernel(
      m_cl_pbas_part1_kernel, cl::NDRange{},
      cl::NDRange{m_parameters.imageWidth, m_parameters.imageHeight});

  set_arg_pbas_part2(
      m_cl_pbas_part2_kernel(), m_cl_mem_feature(), m_parameters.imageWidth,
      m_parameters.imageHeight, m_cl_mem_R(), m_cl_mem_T(), m_cl_mem_index_r(),
      m_parameters.minModels, m_cl_index, m_parameters.modelSize,
      m_cl_mem_mask(), m_cl_mem_avrg_d(), m_cl_mem_random_numbers());
  m_queue.enqueueNDRangeKernel(
      m_cl_pbas_part2_kernel, cl::NDRange{},
      cl::NDRange{m_parameters.imageWidth, m_parameters.imageHeight});

  set_arg_update_R_T(m_cl_update_T_R_kernel(), m_cl_mem_mask(),
                     m_parameters.imageWidth, m_parameters.imageHeight,
                     m_cl_mem_R(), m_cl_mem_T(), m_cl_mem_avrg_d());

  m_queue.enqueueNDRangeKernel(
      m_cl_update_T_R_kernel, cl::NDRange{},
      cl::NDRange{m_parameters.imageWidth, m_parameters.imageHeight});
  m_queue.enqueueReadBuffer(m_cl_mem_mask, true, 0,
                            m_parameters.imageWidth * m_parameters.imageHeight,
                            mask.data);
}

void PBASImpl::create_kernels()
{
  m_cl_fill_R_T_kernel = cl::Kernel{m_program, "fill_T_R"};
  m_cl_fill_model_kernel = cl::Kernel{m_program, "fill_model"};
  m_cl_magnitude_kernel = cl::Kernel{m_program, "magnitude"};
  m_cl_average_Im_kernel = cl::Kernel{m_program, "average"};
  m_cl_pbas_part1_kernel = cl::Kernel{m_program, "pbas_part1"};
  m_cl_pbas_part2_kernel = cl::Kernel{m_program, "pbas_part2"};
  m_cl_update_T_R_kernel = cl::Kernel{m_program, "update_T_R"};
}

void PBASImpl::create_buffers()
{
  const size_t BufferSize =
      sizeof(cl_float) * m_parameters.imageWidth * m_parameters.imageHeight;
  m_cl_mem_T = cl::Buffer{m_context, CL_MEM_READ_WRITE, BufferSize, nullptr};
  m_cl_mem_R = cl::Buffer{m_context, CL_MEM_READ_WRITE, BufferSize, nullptr};

  m_cl_mem_I =
      cl::Buffer{m_context, CL_MEM_READ_WRITE,
                 m_parameters.imageWidth * m_parameters.imageHeight, nullptr};
  m_cl_mem_mask =
      cl::Buffer{m_context, CL_MEM_READ_WRITE,
                 m_parameters.imageWidth * m_parameters.imageHeight, nullptr};

  m_cl_mem_index_r = cl::Buffer{m_context, CL_MEM_READ_WRITE,
                                sizeof(cl_uint) * m_parameters.imageWidth *
                                    m_parameters.imageHeight,
                                nullptr};
  m_cl_mem_feature = cl::Buffer{m_context, CL_MEM_READ_WRITE,
                                sizeof(cl_float2) * m_parameters.imageWidth *
                                    m_parameters.imageHeight,
                                nullptr};
  m_cl_mem_avrg_Im =
      cl::Buffer{m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                 sizeof(cl_float), &m_cl_avrg_Im};

  m_cl_mem_avrg_d =
      cl::Buffer{m_context, CL_MEM_READ_WRITE, BufferSize, nullptr};
  m_cl_mem_random_numbers = cl::Buffer{
      m_context, CL_MEM_READ_WRITE,
      sizeof(cl_uint) * m_parameters.imageWidth * m_parameters.imageHeight,
      nullptr};

  // sizeof(cl_float2) * m_parameters.imageWidth * m_parameters.imageHeight * N
  // = 8 * 320 * 240 *
  // 20
  m_cl_mem_M = cl::Buffer{m_context, CL_MEM_READ_WRITE,
                          sizeof(cl_float2) * m_parameters.imageWidth *
                              m_parameters.imageHeight * m_parameters.modelSize,
                          nullptr};
  // sizeof(cl_float) * m_parameters.imageWidth * m_parameters.imageHeight * N =
  // 4 * 320 * 240 *
  // 20
  m_cl_mem_D = cl::Buffer{m_context, CL_MEM_READ_WRITE,
                          sizeof(cl_float) * m_parameters.imageWidth *
                              m_parameters.imageHeight * m_parameters.modelSize,
                          nullptr};

  cl_uint index_r = 0;
  m_queue.enqueueFillBuffer(m_cl_mem_index_r, index_r, 0,
                            sizeof(cl_uint) * m_parameters.imageWidth *
                                m_parameters.imageHeight);

  std::vector<cl_uint> r_numbers;
  std::generate_n(std::back_insert_iterator<std::vector<cl_uint>>(r_numbers),
                  (m_parameters.imageWidth * m_parameters.imageHeight),
                  randomGenerator);
  m_queue.enqueueWriteBuffer(m_cl_mem_random_numbers, true, 0,
                             sizeof(cl_uint) * m_parameters.imageWidth *
                                 m_parameters.imageHeight,
                             r_numbers.data());
}

void PBASImpl::set_arg_fill_R_T_kernel(cl_kernel &cl_fill_R_T_kernel,
                                       cl_mem &mem_T, const cl_uint &width,
                                       const cl_uint &height, cl_mem &mem_R,
                                       cl_int T, cl_int R)
{
  cl_int err;
  int index = 0;
  err =
      clSetKernelArg(cl_fill_R_T_kernel, index, sizeof(cl_mem), (void *)&mem_T);
  index++;
  err |=
      clSetKernelArg(cl_fill_R_T_kernel, index, sizeof(cl_mem), (void *)&mem_R);
  index++;
  err |= clSetKernelArg(cl_fill_R_T_kernel, index, sizeof(cl_uint),
                        (void *)&width);
  index++;
  err |= clSetKernelArg(cl_fill_R_T_kernel, index, sizeof(cl_uint),
                        (void *)&height);
  index++;
  err |= clSetKernelArg(cl_fill_R_T_kernel, index, sizeof(cl_int), (void *)&T);
  index++;
  err |= clSetKernelArg(cl_fill_R_T_kernel, index, sizeof(cl_int), (void *)&R);

  MCLASSERT(err);
}
void PBASImpl::set_arg_fill_model(cl_kernel &fill_model_kernel,
                                  cl_mem &cl_mem_feature, const cl_uint width,
                                  const cl_uint height, const cl_uint cl_index,
                                  cl_mem &cl_mem_M)
{

  cl_int err;
  int index = 0;
  err = clSetKernelArg(fill_model_kernel, index, sizeof(cl_mem),
                       (void *)&cl_mem_feature);
  index++;
  err |=
      clSetKernelArg(fill_model_kernel, index, sizeof(cl_uint), (void *)&width);
  index++;
  err |= clSetKernelArg(fill_model_kernel, index, sizeof(cl_uint),
                        (void *)&height);
  index++;
  err |= clSetKernelArg(fill_model_kernel, index, sizeof(cl_uint),
                        (void *)&cl_index);
  index++;
  err |= clSetKernelArg(fill_model_kernel, index, sizeof(cl_mem),
                        (void *)&cl_mem_M);
  MCLASSERT(err);
}

void PBASImpl::set_arg_magnitude_kernel(cl_kernel &cl_magnitude_kernel,
                                        cl_mem &src, cl_uint width,
                                        cl_uint height, cl_mem &mag)
{
  cl_int err;
  int index = 0;
  err =
      clSetKernelArg(cl_magnitude_kernel, index, sizeof(cl_mem), (void *)&src);
  index++;
  err |= clSetKernelArg(cl_magnitude_kernel, index, sizeof(cl_uint),
                        (void *)&width);
  index++;
  err |= clSetKernelArg(cl_magnitude_kernel, index, sizeof(cl_uint),
                        (void *)&height);
  index++;
  err |=
      clSetKernelArg(cl_magnitude_kernel, index, sizeof(cl_mem), (void *)&mag);

  MCLASSERT(err);
}

void PBASImpl::set_arg_average_Im(cl_kernel &kernel, cl_mem &mem_Im,
                                  cl_uint width, cl_uint height,
                                  cl_mem &mem_avrg_Im)
{
  cl_int err;
  int index = 0;
  err = clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_Im);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_uint), (void *)&width);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_uint), (void *)&height);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_avrg_Im);

  MCLASSERT(err);
}

void PBASImpl::set_arg_pbas_part1(cl_kernel &kernel, cl_mem &mem_feature,
                                  const int &width, const int &height,
                                  cl_mem &mem_R, cl_uint model_index,
                                  cl_mem &mem_D, cl_mem &mem_M,
                                  cl_mem &mem_index_r, cl_float average_mag)
{
  cl_int err;
  int index = 0;
  err = clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_feature);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_uint), (void *)&width);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_uint), (void *)&height);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_R);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_uint), (void *)&model_index);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_D);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_M);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_index_r);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_float), (void *)&average_mag);
  MCLASSERT(err);
}

void PBASImpl::set_arg_pbas_part2(cl_kernel &kernel, cl_mem &mem_feature,
                                  const int width, const int height,
                                  cl_mem &mem_R, cl_mem &mem_T,
                                  cl_mem &mem_index_r, cl_uint min_v,
                                  const cl_uint cl_index,
                                  const cl_uint model_size, cl_mem &mem_mask,
                                  cl_mem &mem_avrg_d, cl_mem &mem_rand)
{
  cl_int err;
  int index = 0;
  err = clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_feature);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_uint), (void *)&width);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_uint), (void *)&height);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_R);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_T);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_index_r);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_uint), (void *)&min_v);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_uint), (void *)&cl_index);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_uint), (void *)&model_size);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_mask);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_avrg_d);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_rand);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&m_cl_mem_M);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&m_cl_mem_D);
  index++;
  MCLASSERT(err);
}

void PBASImpl::set_arg_update_R_T(cl_kernel &kernel, cl_mem &mem_mask,
                                  const int width, const int height,
                                  cl_mem &mem_R, cl_mem &mem_T,
                                  cl_mem &mem_avrg_d)
{
  cl_int err;
  int index = 0;
  err = clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_mask);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_uint), (void *)&width);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_uint), (void *)&height);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_R);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_T);
  index++;
  err |= clSetKernelArg(kernel, index, sizeof(cl_mem), (void *)&mem_avrg_d);
  index++;
  MCLASSERT(err);
}
