#pragma once

#include <opencv2/opencv.hpp>

#include "OCLWork.hpp"

#define DEBUG_RESOLUTION 0

#if DEBUG_RESOLUTION
const int WIDTH = 640;
const int HEIGHT = 480;

#else
const int WIDTH = 320;
const int HEIGHT = 240;
#endif

static const char *PATH_TO_SOURCE = "D:\\pbas_source\\";


namespace MPBAS
{
#define IS_NOT_NULL(ptr) (ptr == NULL)
#define SDEBUG(msg) \
    printf("Error at %d line in (%s)\n",__LINE__,__FILE__);\
    printf("%s\n", msg);\

#define DEBUG_MODE 0

    const int N = 20;                   
    const int min = 2;
    const cl_float R_inc_dec = 0.05f;
    const int R_scale = 5;
    const cl_float T_dec = 0.05f;
    const int T_inc = 1;
    const int T_upper = 200;

    const cl_float _alpha = 10.f;
    const cl_float _beta = 1.f;

    // opencl constant
    const cl_uint OFFSET_SIZE = 2;



    class PBASImpl
    {
    public:
        PBASImpl();
        ~PBASImpl();

        void init_model(const cv::Mat I);
        void calculate_I_and_M(const cv::Mat &I, std::vector<cv::Mat> &feature);

        float distance(float &I_i, float &I_m, float &B_i, float &B_m, float alpha = 0, float avarage_m = 1);

        void process(cv::Mat src, cv::Mat &mask, bool is_cpu);

        void run(cv::Mat src);


        
        void update_R(int x, int y, float mr);
        void update_T(size_t x, size_t y, uchar p_color, float min_d);
    private:

        cv::Mat _input_image;
        cv::Mat _T;
        cv::Mat _R;
        std::vector<cv::Mat> _model;
        std::vector<cv::Mat> _D;

        int _index;

        cv::RNG randomGenerator;

        //length of random array initialization
        long countOfRandomNumb;

        //pre - initialize the randomNumbers for better performance
        std::vector<int> randomN, randomMinDist, randomX, randomY, randomT, randomTN;
        
        /*
        // ===========================
        //
        // NOTE: OpenCL 2.0
        //
        */
        
        OCLWork m_work;

        int m_cl_index;
        cl_float m_cl_avrg_Im;

        cl_kernel m_cl_fill_R_T_kernel;
        cl_kernel m_cl_magnitude_kernel;
        cl_kernel m_cl_average_Im_kernel;
        cl_kernel m_cl_pbas_part1_kernel;
        cl_kernel m_cl_pbas_part2_kernel;
        cl_kernel m_cl_update_T_R_kernel;

        // ==========================
        // image



        // buffer 
        cl_mem m_cl_mem_I;
        cl_mem m_cl_mem_feature;
        cl_mem m_cl_mem_avrg_Im;
        cl_mem m_cl_mem_index_r;
        cl_mem m_cl_mem_D[N];
        cl_mem m_cl_mem_M[N];
        cl_mem m_cl_mem_mask; 
        cl_mem m_cl_mem_avrg_d;
        cl_mem m_cl_mem_R;
        cl_mem m_cl_mem_T;
        cl_mem m_cl_mem_random_numbers;

        // ==========================
        // ocl function

        void create_kernels();
        void create_images();
        void create_buffers();
        
        void init_cl_model();

        void set_arg_fill_R_T_kernel(cl_kernel &m_cl_fill_R_T_kernel, cl_mem &mem_T, const cl_uint &width, const cl_uint &height, cl_mem &mem_R, cl_int T, cl_int R);
        void set_arg_magnitude_kernel(cl_kernel &m_cl_magnitude_kernel, cl_mem &src, cl_uint width, cl_uint height, cl_mem &mag);
        void set_arg_average_Im(cl_mem &mem_Im, cl_uint width, cl_uint height, cl_mem &mem_avrg_Im);
        void set_arg_pbas_part1(cl_mem &mem_feature, const int &width, const int &height, cl_mem &mem_R, cl_mem &mem_D, cl_mem &mem_M, cl_mem &mem_index_r, cl_float average_mag);
        void set_arg_pbas_part2(cl_mem &mem_feature, const int width, const int height, cl_mem &mem_R, cl_mem &mem_T,  cl_mem &mem_index_r, cl_uint min_v, int &cl_index, const int model_size, cl_mem &mem_mask, cl_mem &mem_avrg_d, cl_mem &mem_rand);
        void set_arg_update_R_T(cl_mem &mem_mask, const int width, const int height, cl_mem &mem_R, cl_mem &mem_T, cl_mem &mem_avrg_d);
    };

};

