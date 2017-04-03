#include "PBASImpl.h"
#include "PBAS.h"
#include "Utiles.hpp"


int main()
{
    cv::VideoCapture cap(0);
    int index = 0;

    MPBAS::PBASImpl pbas_m;
    PBAS pbas;
        
    while (true)
    {
        cv::Mat input;
        cap >> input;

        cv::resize(input, input, cv::Size(WIDTH, HEIGHT));
        cv::cvtColor(input, input, CV_BGR2GRAY);
        cv::GaussianBlur(input, input, cv::Size(5, 5), 1.5);
        pbas_m.run(input);

        if(cv::waitKey(1) == 27) break;
    }
    return 0;
}

