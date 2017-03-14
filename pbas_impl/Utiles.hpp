#pragma once
#include <cl/cl.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>



#define MCLASSERT(ERR)\
	if(ERR != CL_SUCCESS)                                                   \
                {                                                           \
        printf(                                                             \
            "OpenCL error %s happened in file %s at line %s.",              \
            Utiles::LOG(ERR).data(), __FILE__,std::to_string(__LINE__).data());    \
                                                             \
            }


#define MPASSERT(name,data) \
    if(!data)\
    {\
        Utiles::LOG(name, data);\
        printf("\n"); \
        getchar(); \
        exit(-1); \
    }\
        else\
        Utiles::LOG(name, data);\

#define MATINFO(data)\
    Utiles::mat_info(data);\


class Utiles
{
public:
    Utiles();
    ~Utiles();
    static std::string LOG(cl_uint error_id);
    static void LOG(std::string name, void *data);
    static void mat_info(cv::Mat mat);

    static std::string type2str(int type);

    static std::string load_program_cl_from_file(const std::string &filename);


};

class Timer
{
public:
    Timer(){}
    ~Timer(){}
    void start() {
        _start = std::chrono::system_clock::now();
    }
    long long get(){
        return static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::system_clock::now() - _start).count());
    }

private:
    std::chrono::time_point<std::chrono::system_clock> _start;
};
