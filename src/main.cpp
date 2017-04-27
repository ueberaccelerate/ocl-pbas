#include "PBASImpl.hpp"
#include "Utiles.hpp"

int main()
{
  /* cv::VideoCapture cap("../dataset/highwayI.avi"); */
  cv::VideoCapture cap("../dataset/Crowded1.avi");
  /* cv::VideoCapture cap(0); */

  constexpr cl_uint width = 320;
  constexpr cl_uint height = 240;

  PBASImpl pbas_m{PBASParameter{width, height}};

  cv::Mat input;
  utility::timeThis("Total time: ", [&]() {
    while (cap.read(input))
    {
      cv::Mat out;

      cv::resize(input, input, cv::Size(width, height));
      cv::cvtColor(input, input, CV_BGR2GRAY);
      cv::GaussianBlur(input, input, cv::Size(5, 5), 1.5);

      utility::timeThis("Process time: ",
                        [&]() { pbas_m.process(input, out); });

      cv::medianBlur(out, out, 3);
      cv::imshow("Src", input);
      cv::imshow("Msk", out);
      if (cv::waitKey(1) == 27)
        break;
    }
  });
  return 0;
}
