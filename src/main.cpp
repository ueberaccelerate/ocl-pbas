#include "PBASImpl.h"
#include "PBAS.h"
#include "Utiles.hpp"


int main()
{
    //cv::VideoCapture cap(0);
    cv::VideoCapture cap("C:\\data\\sofa\\input\\in%06d.jpg");

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

//         cv::Mat out;
//         Timer t;
//         t.start();
//         pbas.process(&input, &out);
//             
//         printf("Process(1) time: %d ms\n", t.get());

//         cv::medianBlur(out, out, 3);
// 
         pbas_m.run(input);
// 
// 
//         cv::imshow("input src", input);
//         cv::imshow("pbas", out);
            
            
        if(cv::waitKey(1) == 27) break;

    }

    return 0;
}

