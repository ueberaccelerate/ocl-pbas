#include "PBASImpl.hpp"
#include "Utiles.hpp"

#include <iterator>
#include <vector>

void GetPlatformInfo(cl_platform_id pcl, cl_platform_info param_name) {
  std::size_t param_size;
  clGetPlatformInfo(pcl, param_name, 0, nullptr, &param_size);
  std::string buf{};
  buf.resize(param_size);
  clGetPlatformInfo(pcl, param_name, param_size, (void *)buf.data(), nullptr);
  std::cout << buf << "\n";
}

template <typename... T>
void GetPlatformInfoList(cl_platform_id pcl, T... param_name) {
  std::initializer_list<int>{(GetPlatformInfo(pcl, param_name), 0)...};
}

template <typename T>
void SetArgument(cl::Kernel &kernel, const int i, T &arg) {
  kernel.setArg(i, arg);
}
template <typename... T>
void SetArgs(cl::Kernel &kernel, T... args) {
  int i = 0;
  std::initializer_list<int>{(SetArgument(kernel, i++, args), 0)...};
}

PBASImpl::PBASImpl(const PBASParameter param, const std::string &cl_source)
    : m_parameters(param), m_cl_index(0), m_cl_avrg_Im(0.f), m_model_index(0) {
  std::cout << "PBASImpl()\n";

  utility::timeThis("Create Context time: ", [&]() {
    //m_context = cl::Context{CL_DEVICE_TYPE_GPU};
    //std::vector<cl::Platform> platforms;
    //cl::Platform::get(&platforms);
    //m_platform = platforms[1];
    //m_device = cl::Device::getDefault();
    //m_queue = cl::CommandQueue(m_context);

    cl_int error;
    m_context = cl::Context::getDefault(&error);

    m_platform = cl::Platform::getDefault(&error);

    m_device = cl::Device::getDefault();
    m_queue = cl::CommandQueue(m_context);

    if (error) {
      std::cerr << "OpenCL not supported\n";
      std::terminate();
    }

    std::vector<cl::Device> devices;
    m_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    std::string cl_source_path = "../src/opencl_kernels.cl";
    if (!cl_source.empty()) cl_source_path = cl_source;

    std::fstream stream{cl_source_path, std::ios::in};
    // std::fstream stream{ "opencl_kernels.bin", std::ios::in |
    // std::ios::binary };

    const auto sourceSample =
        std::string((std::istreambuf_iterator<char>(stream)),
                    std::istreambuf_iterator<char>());
    if (!sourceSample.length()) {
      std::cerr << "OpenCL Source Empty\n";
      std::terminate();
    }
    GetPlatformInfoList(m_platform(), CL_PLATFORM_PROFILE, CL_PLATFORM_VERSION,
                        CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
                        CL_PLATFORM_EXTENSIONS);

    std::vector<std::pair<const void *, size_t>> binaries{
        std::pair<const void *, size_t>{sourceSample.data(),
                                        sourceSample.size()}};

    m_program = cl::Program{m_context, sourceSample};
    // m_program = cl::Program{m_context, devices, binaries};
    if (m_program.build() != 0) {
      std::string buildLog =
          m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
      std::cerr << buildLog << "\n";
      std::terminate();
    }
  });
  // kernels initialize
  utility::timeThis("Create Kernels time: ", [&]() { create_kernels(); });

  // buffer initialize

  utility::timeThis("Create Buffers time: ", [&]() { create_buffers(); });

  utility::timeThis("Create mask time: ", [&]() {
    m_result_mask = cv::Mat::zeros(m_parameters.imageInfo.height,
                                   m_parameters.imageInfo.width, CV_8UC1);
  });

  m_clahe = cv::createCLAHE();
  m_clahe->setClipLimit(4);
}

cv::Mat PBASImpl::process(cv::Mat src) {
  assert(src.type() == CV_8UC1);

  m_queue.enqueueWriteBuffer(
      m_cl_mem_I, true, 0,
      m_parameters.imageInfo.width * m_parameters.imageInfo.height, src.data);

  SetArgs(m_cl_magnitude_kernel, m_cl_mem_I, m_cl_mem_feature,
          m_cl_mem_parameters);

  m_queue.enqueueNDRangeKernel(m_cl_magnitude_kernel, cl::NDRange{1, 1},
                               cl::NDRange{m_parameters.imageInfo.width - 1,
                                           m_parameters.imageInfo.height - 1});

  if (m_cl_index < m_parameters.modelSize) {
    if (m_cl_index == 0) {
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

  SetArgs(m_cl_pbas_kernel, m_cl_mem_feature, m_cl_mem_R, m_cl_mem_T,
          m_cl_mem_D, m_cl_mem_M, m_cl_mem_mask, m_cl_mem_avrg_d,
          m_cl_mem_random_numbers, m_cl_index, m_cl_mem_model_out,
          m_cl_mem_parameters);

  m_queue.enqueueNDRangeKernel(m_cl_pbas_kernel, cl::NDRange{2, 2},
                               cl::NDRange{m_parameters.imageInfo.width - 4,
                                           m_parameters.imageInfo.height - 4});
  SetArgs(m_cl_update_T_R_kernel, m_cl_mem_mask, m_cl_mem_R, m_cl_mem_T,
          m_cl_mem_avrg_d, m_parameters);

  m_queue.enqueueNDRangeKernel(
      m_cl_update_T_R_kernel, cl::NDRange{},
      cl::NDRange{m_parameters.imageInfo.width, m_parameters.imageInfo.height});
  m_queue.enqueueReadBuffer(
      m_cl_mem_mask, true, 0,
      m_parameters.imageInfo.width * m_parameters.imageInfo.height,
      m_result_mask.data);
  cv::Mat model_out = cv::Mat(m_parameters.imageInfo.height,
                              m_parameters.imageInfo.width, CV_32FC1);

  m_queue.enqueueReadBuffer(m_cl_mem_model_out, true, 0,
                            m_parameters.imageInfo.width *
                                m_parameters.imageInfo.height *
                                sizeof(cl_float),
                            model_out.data);

  cv::imshow("Model", (model_out / 255.f));
  return m_result_mask;
}

cv::Mat PBASImpl::run(cv::Mat src) {
  cv::resize(
      src, src,
      cv::Size(m_parameters.imageInfo.width, m_parameters.imageInfo.height));
  cv::Mat out;

  cv::Mat lab_image;
  cv::cvtColor(src, lab_image, CV_BGR2Lab);

  // Extract the L channel
  std::vector<cv::Mat> lab_planes(3);
  cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

  cv::Mat dst;
  m_clahe->apply(lab_planes[0], dst);

  // Merge the the color planes back into an Lab image
  dst.copyTo(lab_planes[0]);
  cv::merge(lab_planes, lab_image);

  // convert back to RGB
  cv::Mat image_clahe;
  cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);

  cv::cvtColor(image_clahe, image_clahe, CV_BGR2GRAY);
  cv::GaussianBlur(image_clahe, image_clahe, cv::Size(5, 5), 3.5);

  utility::timeThis("Process time: ", [&]() {
    out = process(image_clahe);

  });

  cv::dilate(out, out,
             cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3, 3)));

  cv::medianBlur(out, out, 5);

  return out;
}

void PBASImpl::create_kernels() {
  m_cl_fill_R_T_kernel = cl::Kernel{m_program, "fill_T_R"};
  m_cl_fill_model_kernel = cl::Kernel{m_program, "fill_model"};
  m_cl_magnitude_kernel = cl::Kernel{m_program, "magnitude"};
  m_cl_average_Im_kernel = cl::Kernel{m_program, "average"};
  m_cl_pbas_kernel = cl::Kernel{m_program, "pbas"};
  m_cl_update_T_R_kernel = cl::Kernel{m_program, "update_T_R"};
}

void PBASImpl::create_buffers() {
  const size_t BufferSize = sizeof(cl_float) * m_parameters.imageInfo.width *
                            m_parameters.imageInfo.height *
                            m_parameters.imageInfo.channels;

  m_cl_mem_model_out =
      cl::Buffer{m_context, CL_MEM_READ_WRITE, BufferSize, nullptr};

  m_cl_mem_T = cl::Buffer{m_context, CL_MEM_READ_WRITE, BufferSize, nullptr};
  m_cl_mem_R = cl::Buffer{m_context, CL_MEM_READ_WRITE, BufferSize, nullptr};

  m_cl_mem_I =
      cl::Buffer{m_context, CL_MEM_READ_WRITE,
                 m_parameters.imageInfo.width * m_parameters.imageInfo.height *
                     m_parameters.imageInfo.channels,
                 nullptr};

  m_cl_mem_mask =
      cl::Buffer{m_context, CL_MEM_READ_WRITE,
                 m_parameters.imageInfo.width * m_parameters.imageInfo.height *
                     m_parameters.imageInfo.channels,
                 nullptr};

  m_cl_mem_index_r = cl::Buffer{m_context, CL_MEM_READ_WRITE,
                                sizeof(cl_uint) * m_parameters.imageInfo.width *
                                    m_parameters.imageInfo.height *
                                    m_parameters.imageInfo.channels,
                                nullptr};
  m_cl_mem_feature = cl::Buffer{
      m_context, CL_MEM_READ_WRITE,
      sizeof(cl_float2) * m_parameters.imageInfo.width *
          m_parameters.imageInfo.height * m_parameters.imageInfo.channels,
      nullptr};
  m_cl_mem_parameters =
      cl::Buffer{m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                 sizeof(PBASParameter), &m_parameters};

  m_cl_mem_avrg_d =
      cl::Buffer{m_context, CL_MEM_READ_WRITE, BufferSize, nullptr};
  m_cl_mem_random_numbers = cl::Buffer{
      m_context, CL_MEM_READ_WRITE,
      sizeof(cl_uint) * m_parameters.imageInfo.width *
          m_parameters.imageInfo.height * m_parameters.imageInfo.channels,
      nullptr};

  m_cl_mem_M =
      cl::Buffer{m_context, CL_MEM_READ_WRITE,
                 sizeof(cl_float2) * m_parameters.imageInfo.width *
                     m_parameters.imageInfo.height * m_parameters.modelSize *
                     m_parameters.imageInfo.channels,
                 nullptr};

  m_cl_mem_D =
      cl::Buffer{m_context, CL_MEM_READ_WRITE,
                 sizeof(cl_float) * m_parameters.imageInfo.width *
                     m_parameters.imageInfo.height * m_parameters.modelSize *
                     m_parameters.imageInfo.channels,
                 nullptr};

  cl_uint index_r = 0;
  m_queue.enqueueFillBuffer(m_cl_mem_index_r, index_r, 0,
                            sizeof(cl_uint) * m_parameters.imageInfo.width *
                                m_parameters.imageInfo.height *
                                m_parameters.imageInfo.channels);

  std::vector<cl_uint> r_numbers;
  std::generate_n(
      std::back_insert_iterator<std::vector<cl_uint>>(r_numbers),
      (m_parameters.imageInfo.width * m_parameters.imageInfo.height *
       m_parameters.imageInfo.channels),
      m_random_generator);
  m_queue.enqueueWriteBuffer(m_cl_mem_random_numbers, true, 0,
                             sizeof(cl_uint) * m_parameters.imageInfo.width *
                                 m_parameters.imageInfo.height *
                                 m_parameters.imageInfo.channels,
                             r_numbers.data());
}
