#include "PBASImpl.hpp"
#include "Utiles.hpp"

#include <argagg/argagg.hpp>
#include <string>

int main(int argc, char **argv) {
  argagg::parser argparser{ {
    { "help",{ "-h", "--help" },
    "shows this help message", 0 },
    { "cl",{ "-c", "--cl" },
    "Default value: opencl_kernels.cl", 1 },
    { "input",{ "-i", "--input" },
    "input (default: ../dataset/baseline/pedestrians/input/in%06d.jpg)", 1 }
    } };

  argagg::parser_results args;

  try {
    args = argparser.parse(argc, argv);
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  if (args["help"]) {
    std::cerr << argparser;

    return EXIT_SUCCESS;
  }
  std::string input_source;
  if (args["input"]) {
    input_source = args["input"].as<std::string>() + "in%06d.jpg";
    std::cerr << "Source: " << input_source << "\n";
  }
  else
  {
    input_source = "../dataset/baseline/PETS2006/input/in%06d.jpg";
    std::cerr << "Source: " << input_source << "\n";
  }

  std::string cl_source;
  if (args["cl"])
  {
    cl_source = args["cl"].as<std::string>();
    std::cout << "OpenCL path: " << cl_source << "\n";
  }
  else
  {
    cl_source = "opencl_kernels.cl";
    std::cout << "OpenCL path: " << cl_source << "\n";
  }

  //const std::string category_names[] = {// 0
  //                                      "baseline",
  //                                      // 1
  //                                      "cameraJitter",
  //                                      // 2
  //                                      "dynamicBackground",
  //                                      // 3
  //                                      "intermittentObjectMotion",
  //                                      // 4
  //                                      "shadow",
  //                                      // 5
  //                                      "thermal"};
  //const std::string dataset_names[] = {
  //    "PETS2006", "pedestrians", "office", "highway",
  //    "badminton", "boulevard", "sidewalk", "traffic",
  //    "boats", "canoe", "fall", "fountain01",
  //    "abandonedBox", "parking", "sofa", "streetLight", 
  //    "backdoor", "bungalows", "busStation", "copyMachine", 
  //    "corridor", "diningRoom", "lakeSide", "library"};

  const int category_index = 0;
  cv::VideoCapture cap;
  // bug
  if (input_source.size() == 1) 
  {
    cap.open(std::atoi(input_source.data()));
  }
  else
  {
    cap.open(input_source );
  }
  (input_source);
  if (!cap.isOpened())
  {
    std::cout << "Source is invalid!\n";
    std::terminate();
  }
  //cv::VideoCapture gt_cap("../dataset/" + category_name + "/" +
  //                        dataset_name +
  //                        "/groundtruth/gt%06d.png");

  const cl_uint width = 320 * 1;//cap.get(CV_CAP_PROP_FRAME_WIDTH);    // 320 * 1; //
  const cl_uint height = 240 * 1;// cap.get(CV_CAP_PROP_FRAME_HEIGHT);  // 240 * 1; //

  PBASImpl pbas_m{PBASParameter{width, height, 1},cl_source };

  cv::Mat input;
  
  int frame_count = 0;
  std::cout << "\nWidth: " << width << "\nHeight: " << height << "\n";
  cv::Mat model;
  utility::timeThis("Total time: ", [&]() {
    while (cap.read(input) /*&& gt_cap.read(input_gt)*/) {
      cv::Mat out;
    
      out = pbas_m.run(input);

      cv::imshow("Src", input);
      cv::imshow("Msk", out);
      //cv::imshow("TrueMsk", input_gt);
      if (cv::waitKey(1) == 27) break;
    }
  });
  return 0;
}
