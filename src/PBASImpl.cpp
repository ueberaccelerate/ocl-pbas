#include "PBASImpl.h"

#include <iterator>
#include <vector>

// NOTE: define
#define DEBUG_MEM_CL 0

namespace MPBAS {
cl_int R_lower = 20;
cl_int T_lower = 2;
}

MPBAS::PBASImpl::PBASImpl() : _index(0), m_cl_index(0), m_cl_avrg_Im(0.f) {
  printf("PBASImpl()\n");
  // length of random array
  countOfRandomNumb = 1000;

  for (int l = 0; l < countOfRandomNumb; l++) {
    randomN.push_back((int)randomGenerator.uniform((int)0, (int)N));
    randomX.push_back((int)randomGenerator.uniform(-1, +2));
    randomY.push_back((int)randomGenerator.uniform(-1, +2));
    randomMinDist.push_back((int)randomGenerator.uniform((int)0, (int)N));
    randomT.push_back((int)randomGenerator.uniform((int)0, (int)T_upper));
    randomTN.push_back((int)randomGenerator.uniform((int)0, (int)T_upper));
  }

  // kernels initialize
  create_kernels();

  // images initialize
  create_images();

  // buffer initialize
  create_buffers();
}

MPBAS::PBASImpl::~PBASImpl() {
  clReleaseKernel(m_cl_fill_R_T_kernel);
  clReleaseKernel(m_cl_magnitude_kernel);
  clReleaseKernel(m_cl_average_Im_kernel);
  clReleaseKernel(m_cl_pbas_part1_kernel);
  clReleaseKernel(m_cl_pbas_part2_kernel);
  clReleaseKernel(m_cl_update_T_R_kernel);

  // ==========================
  // image

  // buffer
  clReleaseMemObject(m_cl_mem_I);
  clReleaseMemObject(m_cl_mem_feature);
  clReleaseMemObject(m_cl_mem_avrg_Im);
  clReleaseMemObject(m_cl_mem_index_r);

  for (auto obj : m_cl_mem_D) {
    clReleaseMemObject(obj);
  }
  for (auto obj : m_cl_mem_M) {
    clReleaseMemObject(obj);
  }

  clReleaseMemObject(m_cl_mem_mask);
  clReleaseMemObject(m_cl_mem_avrg_d);
  clReleaseMemObject(m_cl_mem_R);
  clReleaseMemObject(m_cl_mem_T);
  clReleaseMemObject(m_cl_mem_random_numbers);
}
void MPBAS::PBASImpl::init_model(const cv::Mat I) {
  if (_index < N) {

    if (_index == 0) {
      _T = cv::Mat::ones(I.size(), CV_32FC1) * T_lower;
      _R = cv::Mat::ones(I.size(), CV_32FC1) * R_lower;
    }
    auto l_R = _R.clone();
    _D.push_back(l_R);
    calculate_I_and_M(I, _model);
    _index++;
  }
}

void MPBAS::PBASImpl::calculate_I_and_M(const cv::Mat I,
                                        std::vector<cv::Mat> &feature) {
  feature.push_back(I);

    cv::Mat Ix;
    cv::Mat Iy;
    cv::Mat Im;
    cv::Mat Io;

    cv::Sobel(I, Ix, CV_32FC1, 1, 0);
    cv::Sobel(I, Iy, CV_32FC1, 0, 1);
    cv::cartToPolar(Ix, Iy, Im, Io);

    feature.push_back(Im);

    Ix.convertTo(Ix, CV_32FC1, 1.f / 255.f);
    Iy.convertTo(Iy, CV_32FC1, 1.f / 255.f);

    Im.convertTo(Im, CV_32FC1, 1.f / 255.f);
}

float MPBAS::PBASImpl::distance(float &I_i, float &I_m, float &B_i, float &B_m,
                                float alpha /*= 0*/, float avarage_m /*= 1*/)
{
    float res =
        (alpha / avarage_m) * std::abs((I_m - B_m)) + std::abs(I_i - B_i);
    return res;
}
void MPBAS::PBASImpl::process(cv::Mat src, cv::Mat &mask, bool is_cpu)
{
    if (is_cpu)
    {
        assert(src.data);
        mask = cv::Mat::zeros(src.size(), CV_8UC1);
        init_model(src);

        std::vector<cv::Mat> feature;

        calculate_I_and_M(src, feature);

        float min_R = 200;
        float average_mag = 0;
        float t_while = 0;
        float t_update = 0;
        for (size_t y = 0; y < src.rows; y++)
        {
            for (size_t x = 0; x < src.cols; x++)
            {
                float I_m = feature[1].at<float>(y, x);
                average_mag += I_m;
            }
        }
        average_mag /= src.rows * src.cols;

        for (size_t y = 0; y < src.rows; y++)
        {
            for (size_t x = 0; x < src.cols; x++)
            {
                int entry = randomGenerator.uniform(3, countOfRandomNumb - 4);
                float min_d = 0;
                int index_r = 0;
                int index_l = 0;

                float I_i = feature[0].at<uchar>(y, x);
                float I_m = feature[1].at<float>(y, x);
                float R = _R.at<float>(y, x);
                float diff;
                Timer t;
                t.start();
                while (index_r < min && index_l < _index)
                {
                    float B_i = _model[index_l * 2 + 0].at<uchar>(y, x);
                    float B_m = _model[index_l * 2 + 1].at<float>(y, x);
                    diff = distance(I_i, I_m, B_i, B_m, _alpha, average_mag);

                    if (diff < R)
                    {
                        if (diff < min_R)
                        {
                            _D[index_l].at<float>(y, x) = diff;
                        }

                        index_r++;
                    }

                    index_l++;
                }
                t_while += t.get();

                if (index_r >= min)
                {
                    mask.at<uchar>(y, x) = 0;

                    t.start();
                    if (_index == N)
                    {
                        double ratio = std::ceil((double)T_upper /
                                                 (double)_T.at<float>(y, x));

                        int rand_ind = randomN.at(entry);

                        if (randomT.at(entry) < ratio)
                        {
                            _model[rand_ind * 2 + 0].at<uchar>(y, x) = I_i;
                            _model[rand_ind * 2 + 1].at<float>(y, x) = I_m;

                            for (size_t i = 0; i < N; i++)
                            {
                                min_d += _D[i].at<float>(y, x);
                            }
                            min_d /= N;
                        }

                        if (randomTN.at(entry) < ratio)
                        {
                            rand_ind = randomN.at(entry);
                            int n_y = -1 + rand() % 3;
                            int n_x = -1 + rand() % 3;
                            if (y + n_y > 0 && y + n_y < src.rows &&
                                x + n_x > 0 && x + n_x < src.cols)
                            {
                                _model[rand_ind * 2 + 0].at<uchar>(y + n_y,
                                                                   x + n_x) =
                                    feature[0].at<uchar>(y + n_y, x + n_x);
                                _model[rand_ind * 2 + 1].at<float>(y + n_y,
                                                                   x + n_x) =
                                    feature[1].at<float>(y + n_y, x + n_x);
                            }
                        }
                    }

                    t_update += t.get();
                }
                else
                {
                    mask.at<uchar>(y, x) = 255;
                }

                update_R(x, y, min_d);
                update_T(x, y, mask.at<uchar>(y, x), min_d);
            }
        }
    }
    else
    {
        // NOTE: OpenCL work position
        assert(src.type() == CV_8UC1);
        cl_event e{};
        mask = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC1);

        m_work.enqueue_buffer_write(m_cl_mem_I, WIDTH * HEIGHT, src.data, e);

        set_arg_magnitude_kernel(m_cl_magnitude_kernel, m_cl_mem_I, WIDTH,
                                 HEIGHT, m_cl_mem_feature);
        m_work.run_cl_kernel(m_cl_magnitude_kernel, cv::Vec2i(WIDTH, HEIGHT),
                             cv::Vec2i(16, 16), e);
#if DEBUG_MEM_CL
        cv::Mat test = cv::Mat(HEIGHT, WIDTH, CV_8UC1);
        m_work.enqueue_buffer_read(m_cl_mem_I, WIDTH * HEIGHT, test.data, e);

        cv::imshow("test", test);
#endif

        if (m_cl_index < N)
        {
            if (m_cl_index == 0)
            {
                set_arg_fill_R_T_kernel(m_cl_fill_R_T_kernel, m_cl_mem_T, WIDTH,
                                        HEIGHT, m_cl_mem_R, T_lower, R_lower);
                m_work.run_cl_kernel(m_cl_fill_R_T_kernel,
                                     cv::Vec2i(WIDTH, HEIGHT),
                                     cv::Vec2i(16, 16), e);
            }

            m_work.enqueue_buffer_copy(m_cl_mem_feature, m_cl_mem_M[m_cl_index],
                                       sizeof(cl_float2) * WIDTH * HEIGHT, e);
            // m_work.enqueue_buffer_copy(m_cl_mem_R, m_cl_mem_D[m_cl_index],
            // sizeof(cl_float) * WIDTH * HEIGHT, e);

            // cl_float fill = 0.f;
            // m_work.enqueue_buffer_fill(m_cl_mem_D[m_cl_index],
            // sizeof(cl_float) *
            // WIDTH * HEIGHT, &fill, sizeof(cl_float), e);

            m_cl_index++;
        }

        cl_float min_R = 200;
        cl_float average_mag = 0;
        cl_float t_while = 0;
        cl_float t_update = 0;

        float min_d = 0;
        int index_l = 0;

        //// WARNING: giving different avrg from cpu process
        m_work.enqueue_buffer_fill(m_cl_mem_avrg_Im, sizeof(cl_float),
                                   &average_mag, sizeof(cl_float), e);
        set_arg_average_Im(m_cl_mem_feature, WIDTH, HEIGHT, m_cl_mem_avrg_Im);
        m_work.run_cl_kernel(m_cl_average_Im_kernel, cv::Vec2i(WIDTH, HEIGHT),
                             cv::Vec2i(16, 16), e);
        cl_uint index_r = 0;

        m_work.enqueue_buffer_read(m_cl_mem_avrg_Im, sizeof(cl_float),
                                   &m_cl_avrg_Im, e);
        m_cl_avrg_Im /= WIDTH * HEIGHT;

        m_work.enqueue_buffer_fill(m_cl_mem_index_r,
                                   sizeof(cl_uint) * WIDTH * HEIGHT, &index_r,
                                   sizeof(cl_uint), e);
        while (index_l < m_cl_index)
        {
            set_arg_pbas_part1(m_cl_mem_feature, WIDTH, HEIGHT, m_cl_mem_R,
                               m_cl_mem_D[index_l], m_cl_mem_M[index_l],
                               m_cl_mem_index_r, m_cl_avrg_Im);
            m_work.run_cl_kernel(m_cl_pbas_part1_kernel,
                                 cv::Vec2i(WIDTH, HEIGHT), cv::Vec2i(16, 16),
                                 e);

            index_l++;
        }

        set_arg_pbas_part2(m_cl_mem_feature, WIDTH, HEIGHT, m_cl_mem_R,
                           m_cl_mem_T, m_cl_mem_index_r, min, m_cl_index, N,
                           m_cl_mem_mask, m_cl_mem_avrg_d,
                           m_cl_mem_random_numbers);
        m_work.run_cl_kernel(m_cl_pbas_part2_kernel, cv::Vec2i(WIDTH, HEIGHT),
                             cv::Vec2i(16, 16), e);

        set_arg_update_R_T(m_cl_mem_mask, WIDTH, HEIGHT, m_cl_mem_R, m_cl_mem_T,
                           m_cl_mem_avrg_d);
        m_work.run_cl_kernel(m_cl_update_T_R_kernel, cv::Vec2i(WIDTH, HEIGHT),
                             cv::Vec2i(16, 16), e);
        m_work.enqueue_buffer_read(m_cl_mem_mask, WIDTH * HEIGHT, mask.data, e);

        cv::Mat T = cv::Mat(HEIGHT, WIDTH, CV_32FC1);
        cv::Mat R = cv::Mat(HEIGHT, WIDTH, CV_32FC1);

        m_work.enqueue_buffer_read(
            m_cl_mem_T, sizeof(cl_float) * WIDTH * HEIGHT, T.data, e);
        m_work.enqueue_buffer_read(
            m_cl_mem_R, sizeof(cl_float) * WIDTH * HEIGHT, R.data, e);

#if DEBUG_MEM_CL

        cv::imshow("gT", T);
        cv::imshow("gR", R);
        cv::Mat mask_test = cv::Mat(HEIGHT, WIDTH, CV_8UC1);
        m_work.enqueue_buffer_read(m_cl_mem_mask, WIDTH * HEIGHT,
                                   mask_test.data, e);

        cv::imshow("mask_test", mask_test);
        cv::waitKey(1);
#endif
    }
}

void MPBAS::PBASImpl::run(cv::Mat src)
{
    Timer t;
    cv::Mat mask_cpu;
    cv::Mat mask_gpu;

    //   t.start();
    //   process(src, mask_cpu, true);
    //   printf("Process(2) time: %d ms\n", t.get());

    t.start();
    process(src, mask_gpu, false);
    printf("Process(3) time: %d ms\n", t.get());
    cv::medianBlur(mask_gpu, mask_gpu, 3);

    cv::imshow("pbasImplGpu", mask_gpu);
}

void MPBAS::PBASImpl::update_R(int x, int y, float mr)
{
    float R = _R.at<float>(y, x);

    if (R > mr * R_scale)
    {
        R = R * (1.f - R_inc_dec);
    }
    else
    {
        R = R * (1.f + R_inc_dec);
    }

    if (R < R_lower) R = (float)R_lower;

    _R.at<float>(y, x) = R;
}

void MPBAS::PBASImpl::update_T(size_t x, size_t y, uchar p_color, float min_d)
{
    float t = _T.at<float>(y, x);
    if (p_color == 0)
    {
        t -= (T_dec / min_d);
    }
    else
    {
        t += (T_inc / min_d);
    }
    if (t < T_lower)
        t = T_lower;
    else if (t > T_upper)
        t = T_upper;
    _T.at<float>(y, x) = t;
}

void MPBAS::PBASImpl::create_kernels()
{
    m_work.create_kernel(m_cl_fill_R_T_kernel, "fill_T_R");
    m_work.create_kernel(m_cl_magnitude_kernel, "magnitude");
    m_work.create_kernel(m_cl_average_Im_kernel, "average");
    m_work.create_kernel(m_cl_pbas_part1_kernel, "pbas_part1");
    m_work.create_kernel(m_cl_pbas_part2_kernel, "pbas_part2");
    m_work.create_kernel(m_cl_update_T_R_kernel, "update_T_R");
}

void MPBAS::PBASImpl::create_images()
{
    // m_work.create_image(m_cl_mem_T, CL_MEM_READ_WRITE, WIDTH, HEIGHT,
    // nullptr);
    // m_work.create_image(m_cl_mem_R, CL_MEM_READ_WRITE, WIDTH, HEIGHT,
    // nullptr);
    return;
    for (size_t i = 0; i < N; i++)
    {
        //         m_work.create_image(m_cl_MI[i], CL_MEM_READ_WRITE, WIDTH,
        //         HEIGHT,
        //         nullptr); m_work.create_image(m_cl_MS[i], CL_MEM_READ_WRITE,
        //         WIDTH, HEIGHT, nullptr);
    }

    // deprecate, but may be used
    for (size_t i = 0; i < N; i++)
    {
        m_work.create_image(m_cl_mem_D[i], CL_MEM_READ_WRITE, WIDTH, HEIGHT,
                            nullptr);
    }
}

void MPBAS::PBASImpl::create_buffers()
{
    m_work.create_buffer(m_cl_mem_T, CL_MEM_READ_WRITE,
                         sizeof(cl_float) * WIDTH * HEIGHT, nullptr);
    m_work.create_buffer(m_cl_mem_R, CL_MEM_READ_WRITE,
                         sizeof(cl_float) * WIDTH * HEIGHT, nullptr);

    m_work.create_buffer(m_cl_mem_I, CL_MEM_READ_WRITE, WIDTH * HEIGHT,
                         nullptr);
    m_work.create_buffer(m_cl_mem_mask, CL_MEM_READ_WRITE, WIDTH * HEIGHT,
                         nullptr);
    m_work.create_buffer(m_cl_mem_index_r, CL_MEM_READ_WRITE,
                         sizeof(cl_uint) * WIDTH * HEIGHT, nullptr);
    m_work.create_buffer(m_cl_mem_feature, CL_MEM_READ_WRITE,
                         sizeof(cl_float2) * WIDTH * HEIGHT, nullptr);
    m_work.create_buffer(m_cl_mem_avrg_Im,
                         CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                         sizeof(cl_float), &m_cl_avrg_Im);

    m_work.create_buffer(m_cl_mem_avrg_d, CL_MEM_READ_WRITE,
                         sizeof(cl_float) * WIDTH * HEIGHT, nullptr);

    m_work.create_buffer(m_cl_mem_random_numbers, CL_MEM_READ_WRITE,
                         sizeof(cl_uint) * WIDTH * HEIGHT, nullptr);

    for (size_t i = 0; i < N; i++)
    {
        m_work.create_buffer(m_cl_mem_M[i], CL_MEM_READ_WRITE,
                             sizeof(cl_float2) * WIDTH * HEIGHT, nullptr);
    }
    for (size_t i = 0; i < N; i++)
    {
        m_work.create_buffer(m_cl_mem_D[i], CL_MEM_READ_WRITE,
                             sizeof(cl_float) * WIDTH * HEIGHT, nullptr);
    }
    cl_event e{};
    cl_uint index_r = 0;

    m_work.enqueue_buffer_fill(m_cl_mem_index_r, WIDTH * HEIGHT, &index_r,
                               sizeof(cl_uint), e);
    cl_float average_d = 0.f;
    m_work.enqueue_buffer_fill(m_cl_mem_index_r,
                               sizeof(cl_float) * WIDTH * HEIGHT, &average_d,
                               sizeof(cl_uint), e);
    std::vector<cl_uint> r_numbers;

    std::generate_n(std::back_insert_iterator<std::vector<cl_uint>>(r_numbers),
                    (WIDTH * HEIGHT), std::rand);

    m_work.enqueue_buffer_write(m_cl_mem_random_numbers,
                                sizeof(cl_uint) * WIDTH * HEIGHT,
                                r_numbers.data(), e);
}

void MPBAS::PBASImpl::init_cl_model() {}

void MPBAS::PBASImpl::set_arg_fill_R_T_kernel(
    cl_kernel &m_cl_fill_R_T_kernel, cl_mem &mem_T, const cl_uint &width,
    const cl_uint &height, cl_mem &mem_R, cl_int T, cl_int R) {
  cl_int err;
  int index = 0;
  err = clSetKernelArg(m_cl_fill_R_T_kernel, index, sizeof(cl_mem),
                       (void *)&mem_T);
  index++;
  err |= clSetKernelArg(m_cl_fill_R_T_kernel, index, sizeof(cl_mem),
                        (void *)&mem_R);
  index++;
  err |= clSetKernelArg(m_cl_fill_R_T_kernel, index, sizeof(cl_uint),
                       (void *)&width);
  index++;
  err |= clSetKernelArg(m_cl_fill_R_T_kernel, index, sizeof(cl_uint),
                       (void *)&height);
  index++;
  err |= clSetKernelArg(m_cl_fill_R_T_kernel, index, sizeof(cl_int), (void *)&T);
  index++;
  err |=
      clSetKernelArg(m_cl_fill_R_T_kernel, index, sizeof(cl_int), (void *)&R);

  MCLASSERT(err);
}

void MPBAS::PBASImpl::set_arg_magnitude_kernel(cl_kernel &m_cl_magnitude_kernel,
                                               cl_mem &src, cl_uint width,
                                               cl_uint height, cl_mem &mag) {
  cl_int err;
  int index = 0;
  err = clSetKernelArg(m_cl_magnitude_kernel, index, sizeof(cl_mem),
                       (void *)&src);
  index++;
  err |= clSetKernelArg(m_cl_magnitude_kernel, index, sizeof(cl_uint),
                       (void *)&width);
  index++;
  err |= clSetKernelArg(m_cl_magnitude_kernel, index, sizeof(cl_uint),
                       (void *)&height);
  index++;
  err |= clSetKernelArg(m_cl_magnitude_kernel, index, sizeof(cl_mem),
                        (void *)&mag);

  MCLASSERT(err);
}

void MPBAS::PBASImpl::set_arg_average_Im(cl_mem &mem_Im, cl_uint width,
                                         cl_uint height, cl_mem &mem_avrg_Im) {
  cl_int err;
  int index = 0;
  err = clSetKernelArg(m_cl_average_Im_kernel, index, sizeof(cl_mem),
                       (void *)&mem_Im);
  index++;
  err |= clSetKernelArg(m_cl_average_Im_kernel, index, sizeof(cl_uint),
                       (void *)&width);
  index++;
  err |= clSetKernelArg(m_cl_average_Im_kernel, index, sizeof(cl_uint),
                       (void *)&height);
  index++;
  err |= clSetKernelArg(m_cl_average_Im_kernel, index, sizeof(cl_mem),
                        (void *)&mem_avrg_Im);

  MCLASSERT(err);
}

void MPBAS::PBASImpl::set_arg_pbas_part1(cl_mem &mem_feature, const int &width,
                                         const int &height, cl_mem &mem_R,
                                         cl_mem &mem_D, cl_mem &mem_M,
                                         cl_mem &mem_index_r,
                                         cl_float average_mag) {
  cl_int err;
  int index = 0;
  err = clSetKernelArg(m_cl_pbas_part1_kernel, index, sizeof(cl_mem),
                       (void *)&mem_feature);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part1_kernel, index, sizeof(cl_uint),
                       (void *)&width);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part1_kernel, index, sizeof(cl_uint),
                       (void *)&height);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part1_kernel, index, sizeof(cl_mem),
                        (void *)&mem_R);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part1_kernel, index, sizeof(cl_mem),
                        (void *)&mem_D);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part1_kernel, index, sizeof(cl_mem),
                        (void *)&mem_M);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part1_kernel, index, sizeof(cl_mem),
                        (void *)&mem_index_r);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part1_kernel, index, sizeof(cl_float),
      (void *)&average_mag);
  MCLASSERT(err);
}

void MPBAS::PBASImpl::set_arg_pbas_part2(cl_mem &mem_feature, const int width,
                                         const int height, cl_mem &mem_R,
                                         cl_mem &mem_T, cl_mem &mem_index_r,
                                         cl_uint min_v, int &cl_index,
                                         const int model_size, cl_mem &mem_mask,
                                         cl_mem &mem_avrg_d, cl_mem &mem_rand) {
  cl_int err;
  int index = 0;
  err = clSetKernelArg(m_cl_pbas_part2_kernel, index, sizeof(cl_mem),
                       (void *)&mem_feature);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part2_kernel, index, sizeof(cl_uint),
                       (void *)&width);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part2_kernel, index, sizeof(cl_uint),
                       (void *)&height);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part2_kernel, index, sizeof(cl_mem),
                        (void *)&mem_R);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part2_kernel, index, sizeof(cl_mem),
                        (void *)&mem_T);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part2_kernel, index, sizeof(cl_mem),
                        (void *)&mem_index_r);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part2_kernel, index, sizeof(cl_uint),
                        (void *)&min_v);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part2_kernel, index, sizeof(int),
                        (void *)&cl_index);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part2_kernel, index, sizeof(int),
                        (void *)&model_size);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part2_kernel, index, sizeof(cl_mem),
                        (void *)&mem_mask);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part2_kernel, index, sizeof(cl_mem),
                        (void *)&mem_avrg_d);
  index++;
  err |= clSetKernelArg(m_cl_pbas_part2_kernel, index, sizeof(cl_mem),
                        (void *)&mem_rand);
  index++;
  /************************************************************************/
  /*                            all model                                 */
  /************************************************************************/
  for (size_t i = 0; i < N; i++) {
    err |= clSetKernelArg(m_cl_pbas_part2_kernel, index, sizeof(cl_mem),
                          (void *)&m_cl_mem_M[i]);
    index++;
  }
  for (size_t i = 0; i < N; i++) {
    err |= clSetKernelArg(m_cl_pbas_part2_kernel, index, sizeof(cl_mem),
                          (void *)&mem_avrg_d);
    index++;
  }
  MCLASSERT(err);
}

void MPBAS::PBASImpl::set_arg_update_R_T(cl_mem &mem_mask, const int width,
                                         const int height, cl_mem &mem_R,
                                         cl_mem &mem_T, cl_mem &mem_avrg_d) {
  cl_int err;
  int index = 0;
  err = clSetKernelArg(m_cl_update_T_R_kernel, index, sizeof(cl_mem),
                       (void *)&mem_mask);
  index++;
  err |= clSetKernelArg(m_cl_update_T_R_kernel, index, sizeof(cl_uint),
                       (void *)&width);
  index++;
  err |= clSetKernelArg(m_cl_update_T_R_kernel, index, sizeof(cl_uint),
                       (void *)&height);
  index++;
  err |= clSetKernelArg(m_cl_update_T_R_kernel, index, sizeof(cl_mem),
                        (void *)&mem_R);
  index++;
  err |= clSetKernelArg(m_cl_update_T_R_kernel, index, sizeof(cl_mem),
                        (void *)&mem_T);
  index++;
  err |= clSetKernelArg(m_cl_update_T_R_kernel, index, sizeof(cl_mem),
                        (void *)&mem_avrg_d);
  index++;
  MCLASSERT(err);
}
