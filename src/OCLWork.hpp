#pragma once
#include "Utiles.hpp"

#include <CL/cl.h>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

class Timer;



//#define mydbg(dev, format, arg…) \
//do { \
//  if (mycooldriver->debug > 0) \
//    dev_printk(KERN_DEBUG, dev,  format, ##arg); \
//} while (0)
#define PYRAMID_LEVEL 4


class OCLWork
{
public:
	OCLWork();
	~OCLWork();  

	bool create_kernel(cl_kernel &l_kernel, const std::string &kernel_name);

    void create_image(cl_mem &mem, const cl_mem_flags &flags, const cl_uint &width, const cl_uint &height, void *data);

    void create_buffer(cl_mem &mem, const cl_mem_flags &flags, cl_uint size, void *data);

    void enqueue_image_read(cl_mem &mem,cl_uint width, cl_uint height, void *data);
    void enqueue_image_write(cl_mem &mem, cl_uint width, cl_uint height, void *data);

    void enqueue_buffer_read(cl_mem &mem, cl_uint size, void *data, cl_event &event);
    void enqueue_buffer_write(cl_mem &mem, cl_uint size, void *data, cl_event &event);
    void enqueue_buffer_copy(cl_mem &mem_src, cl_mem &mem_dest, cl_uint size, cl_event &event);
    void enqueue_buffer_fill(cl_mem &mem_buf, cl_uint size_buf, void *fill_val, cl_uint size_fill, cl_event &event);

    void enqueue_image_buffer_copy(cl_mem &mem_src, cl_mem &mem_dest, cl_uint width, cl_uint height, cl_event &event);
    
    void run_cl_kernel(cl_kernel &l_kernel, const cv::Vec2i &globalNDSize, const cv::Vec2i &localNDSize, cl_event &e);

	void get_platform_all_info();
	std::string get_platform_info(const cl_platform_id &platform, const cl_platform_info &platform_info);
	void get_device_info();
	

private:
	std::vector<cl_platform_id> _platform_ids;
	cl_context _context;
	cl_device_id _device_ids;

	cl_uint _count_platform;
	cl_uint _count_device;	


	cl_command_queue _command_queue;
	cl_program _program;

};


