#pragma once
#include <CL/cl.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#define MCLASSERT(ERR)                                                         \
  if (ERR != CL_SUCCESS)                                                       \
  {                                                                            \
    printf("OpenCL error %s happened in file %s at line %s.\n",                \
           utility::Utiles::LOG(ERR).data(), __FILE__,                         \
           std::to_string(__LINE__).data());                                   \
    int ibreak;                                                                \
    ibreak = 0;                                                                \
  }
namespace utility
{

#define MPASSERT(name, data)                                                   \
  if (!data)                                                                   \
  {                                                                            \
    Utiles::LOG(name, data);                                                   \
    printf("\n");                                                              \
    getchar();                                                                 \
    exit(-1);                                                                  \
  }                                                                            \
  else                                                                         \
    Utiles::LOG(name, data);

#define MATINFO(data) Utiles::mat_info(data)

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

void timeThis(const std::string &label, std::function<void()> &&func);

}
