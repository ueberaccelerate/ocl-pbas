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

template <typename T> void setArgument(cl::Kernel &kernel, const int i, T arg)
{
  kernel.setArg(i, arg);
}

template <typename... T> void SetArgs(cl::Kernel &kernel, T... args)
{
  int i = 0;
  std::initializer_list<int>{(setArgument(kernel, i++, args), 0)...};
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

  utility::timeThis("Setting args time: ", [&]() { set_args(); });

  utility::timeThis("Create mask time: ", [&]() {
    m_result_mask = cv::Mat::zeros(m_parameters.imageInfo.height,
                                   m_parameters.imageInfo.width, CV_8UC1);
  });
}

cv::Mat PBASImpl::process(cv::Mat src)
{
  assert(src.type() == CV_8UC1);

  m_queue.enqueueWriteBuffer(
      m_cl_mem_I, true, 0,
      m_parameters.imageInfo.width * m_parameters.imageInfo.height, src.data);
  SetArgs(m_cl_magnitude_kernel, m_cl_mem_I, m_cl_mem_feature, m_parameters);
  m_queue.enqueueNDRangeKernel(
      m_cl_magnitude_kernel, cl::NDRange{},
      cl::NDRange{m_parameters.imageInfo.width, m_parameters.imageInfo.height});

  if (m_cl_index < m_parameters.modelSize)
  {
    if (m_cl_index == 0)
    {
      SetArgs(m_cl_fill_R_T_kernel, m_cl_mem_R, m_cl_mem_T, m_parameters);
      m_queue.enqueueNDRangeKernel(m_cl_fill_R_T_kernel, cl::NDRange{},
                                   cl::NDRange{m_parameters.imageInfo.width,
                                               m_parameters.imageInfo.height});
    }
    SetArgs(m_cl_fill_model_kernel, m_cl_mem_feature, m_cl_mem_M, m_cl_index,
            m_parameters);
    m_queue.enqueueNDRangeKernel(m_cl_fill_model_kernel, cl::NDRange{},
                                 cl::NDRange{m_parameters.imageInfo.width,
                                             m_parameters.imageInfo.height});
    ++m_cl_index;
  }
  cl_float average_mag = 0;
  m_queue.enqueueFillBuffer(m_cl_mem_avrg_Im, average_mag, 0, sizeof(cl_float));
  SetArgs(m_cl_average_Im_kernel, m_cl_mem_feature, m_parameters,
          m_cl_mem_avrg_Im);
  m_queue.enqueueNDRangeKernel(
      m_cl_average_Im_kernel, cl::NDRange{},
      cl::NDRange{m_parameters.imageInfo.width, m_parameters.imageInfo.height});

  m_queue.enqueueReadBuffer(m_cl_mem_avrg_Im, true, 0, sizeof(cl_float),
                            &m_cl_avrg_Im);
  m_cl_avrg_Im /= m_parameters.imageInfo.width * m_parameters.imageInfo.height;

  SetArgs(m_cl_pbas_kernel, m_cl_mem_feature, m_cl_mem_R, m_cl_mem_T,
          m_cl_mem_D, m_cl_mem_M, m_cl_mem_mask, m_cl_mem_avrg_d,
          m_cl_mem_random_numbers, m_cl_index, m_cl_avrg_Im, m_parameters);

  m_queue.enqueueNDRangeKernel(
      m_cl_pbas_kernel, cl::NDRange{},
      cl::NDRange{m_parameters.imageInfo.width, m_parameters.imageInfo.height});
  SetArgs(m_cl_update_T_R_kernel, m_cl_mem_mask, m_cl_mem_R, m_cl_mem_T,
          m_cl_mem_avrg_d, m_parameters);
  m_queue.enqueueNDRangeKernel(
      m_cl_update_T_R_kernel, cl::NDRange{},
      cl::NDRange{m_parameters.imageInfo.width, m_parameters.imageInfo.height});
  m_queue.enqueueReadBuffer(m_cl_mem_mask, true, 0,
                            m_parameters.imageInfo.width *
                                m_parameters.imageInfo.height,
                            m_result_mask.data);
  return m_result_mask;
}

void PBASImpl::set_args() {}
void PBASImpl::create_kernels()
{
  m_cl_fill_R_T_kernel = cl::Kernel{m_program, "fill_T_R"};
  m_cl_fill_model_kernel = cl::Kernel{m_program, "fill_model"};
  m_cl_magnitude_kernel = cl::Kernel{m_program, "magnitude"};
  m_cl_average_Im_kernel = cl::Kernel{m_program, "average"};
  m_cl_pbas_kernel = cl::Kernel{m_program, "pbas"};
  m_cl_update_T_R_kernel = cl::Kernel{m_program, "update_T_R"};
}

void PBASImpl::create_buffers()
{
  const size_t BufferSize = sizeof(cl_float) * m_parameters.imageInfo.width *
                            m_parameters.imageInfo.height;
  m_cl_mem_T = cl::Buffer{m_context, CL_MEM_READ_WRITE, BufferSize, nullptr};
  m_cl_mem_R = cl::Buffer{m_context, CL_MEM_READ_WRITE, BufferSize, nullptr};

  m_cl_mem_I = cl::Buffer{
      m_context, CL_MEM_READ_WRITE,
      m_parameters.imageInfo.width * m_parameters.imageInfo.height, nullptr};
  m_cl_mem_mask = cl::Buffer{
      m_context, CL_MEM_READ_WRITE,
      m_parameters.imageInfo.width * m_parameters.imageInfo.height, nullptr};

  m_cl_mem_index_r = cl::Buffer{m_context, CL_MEM_READ_WRITE,
                                sizeof(cl_uint) * m_parameters.imageInfo.width *
                                    m_parameters.imageInfo.height,
                                nullptr};
  m_cl_mem_feature =
      cl::Buffer{m_context, CL_MEM_READ_WRITE,
                 sizeof(cl_float2) * m_parameters.imageInfo.width *
                     m_parameters.imageInfo.height,
                 nullptr};
  m_cl_mem_avrg_Im =
      cl::Buffer{m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                 sizeof(cl_float), &m_cl_avrg_Im};

  m_cl_mem_avrg_d =
      cl::Buffer{m_context, CL_MEM_READ_WRITE, BufferSize, nullptr};
  m_cl_mem_random_numbers =
      cl::Buffer{m_context, CL_MEM_READ_WRITE,
                 sizeof(cl_uint) * m_parameters.imageInfo.width *
                     m_parameters.imageInfo.height,
                 nullptr};

  // sizeof(cl_float2) * m_parameters.imageInfo.width *
  // m_parameters.imageInfo.height * N
  // = 8 * 320 * 240 *
  // 20
  m_cl_mem_M =
      cl::Buffer{m_context, CL_MEM_READ_WRITE,
                 sizeof(cl_float2) * m_parameters.imageInfo.width *
                     m_parameters.imageInfo.height * m_parameters.modelSize,
                 nullptr};
  // sizeof(cl_float) * m_parameters.imageInfo.width *
  // m_parameters.imageInfo.height * N =
  // 4 * 320 * 240 *
  // 20
  m_cl_mem_D =
      cl::Buffer{m_context, CL_MEM_READ_WRITE,
                 sizeof(cl_float) * m_parameters.imageInfo.width *
                     m_parameters.imageInfo.height * m_parameters.modelSize,
                 nullptr};

  cl_uint index_r = 0;
  m_queue.enqueueFillBuffer(m_cl_mem_index_r, index_r, 0,
                            sizeof(cl_uint) * m_parameters.imageInfo.width *
                                m_parameters.imageInfo.height);

  std::vector<cl_uint> r_numbers;
  std::generate_n(
      std::back_insert_iterator<std::vector<cl_uint>>(r_numbers),
      (m_parameters.imageInfo.width * m_parameters.imageInfo.height),
      m_random_generator);
  m_queue.enqueueWriteBuffer(m_cl_mem_random_numbers, true, 0,
                             sizeof(cl_uint) * m_parameters.imageInfo.width *
                                 m_parameters.imageInfo.height,
                             r_numbers.data());
}
