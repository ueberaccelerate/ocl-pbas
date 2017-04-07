#include "OCLWork.hpp"
#define CL_VERSION_2_0 0

OCLWork::OCLWork() 
{

    cl_uint err = clGetPlatformIDs(0, NULL, &_count_platform);
    MCLASSERT(err);
    _platform_ids.resize(_count_platform);
    err = clGetPlatformIDs(_count_platform, &_platform_ids[0], nullptr);
    MCLASSERT(err);
    get_platform_all_info();
    err = clGetDeviceIDs(_platform_ids[0], CL_DEVICE_TYPE_CPU, 1, &_device_ids,
                         nullptr);
    cl_int uerr;
    _context = clCreateContext(NULL, 1, &_device_ids, NULL, NULL, &uerr);
    //_context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_CPU, NULL, NULL,
    //&uerr);
    MCLASSERT(uerr);
    get_device_info();

#ifdef CL_VERSION_2_0
    //    _command_queue = clCreateCommandQueueWithProperties(_context,
    //    _device_ids, NULL, &uerr);
    // MCLASSERT(uerr);
    _command_queue = clCreateCommandQueue(_context, _device_ids, NULL, &uerr);
    MCLASSERT(uerr);
#else
    _command_queue = clCreateCommandQueue(_context, _device_ids, NULL, &uerr);
    MCLASSERT(uerr);
#endif

    std::string source_program = Utiles::load_program_cl_from_file(
        "src/opencl_kernels.cl");
    const char *source = source_program.c_str();
    const size_t size = source_program.size();
    _program = clCreateProgramWithSource(_context, 1, &source, &size, &uerr);
    MCLASSERT(uerr);

    std::string buildOptions = "";  // -DName=Value

    uerr = clBuildProgram(_program, 1, &_device_ids, buildOptions.data(), NULL,
                          NULL);

    if (uerr != CL_SUCCESS)
    {
        std::vector<char> buff_erro;
        cl_int errcode;
        size_t build_log_len;
        errcode =
            clGetProgramBuildInfo(_program, _device_ids, CL_PROGRAM_BUILD_LOG,
                                  0, NULL, &build_log_len);
        if (errcode)
        {
            printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
            getchar();
            exit(-1);
        }
        buff_erro.resize(build_log_len);

        if (!buff_erro.size())
        {
            printf("malloc failed at line %d\n", __LINE__);
            getchar();
            exit(-2);
        }

        errcode =
            clGetProgramBuildInfo(_program, _device_ids, CL_PROGRAM_BUILD_LOG,
                                  build_log_len, buff_erro.data(), NULL);
        if (errcode)
        {
            printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
            getchar();
            exit(-3);
        }

        fprintf(stderr, "Build log: \n%s\n",
                buff_erro.data());  // Be careful with  the fprint
        fprintf(stderr, "clBuildProgram failed\n");
        getchar();
        exit(EXIT_FAILURE);
    }
}

OCLWork::~OCLWork()
{
    clReleaseProgram(_program);
    clReleaseCommandQueue(_command_queue);
    clReleaseContext(_context);
}

bool OCLWork::create_kernel(cl_kernel &l_kernel, const std::string &kernel_name)
{
    assert(!kernel_name.empty());

    if (kernel_name.empty()) return false;

    cl_int err;
    l_kernel = clCreateKernel(_program, kernel_name.c_str(), &err);
    MCLASSERT(err);

    return true;
}

void OCLWork::create_image(cl_mem &mem, const cl_mem_flags &flags,
                           const cl_uint &width, const cl_uint &height,
                           void *data)
{
    cl_int err;
    cl_image_format format;

    format.image_channel_order = CL_INTENSITY;
    format.image_channel_data_type = CL_FLOAT;

#ifdef CL_VERSION_2_0
    cl_image_desc desc;
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = width;
    desc.image_height = height;
    desc.image_depth = 1;
    desc.image_array_size = 1;
    desc.image_row_pitch = 0;
    desc.image_slice_pitch = 0;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
    // desc.mem_object = NULL;

    mem = clCreateImage(_context, flags, &format, &desc, data, &err);
    MCLASSERT(err);
#else
    mem =
        clCreateImage2D(_context, flags, &format, width, height, 0, data, &err);
    MCLASSERT(err);
#endif
}

void OCLWork::create_buffer(cl_mem &mem, const cl_mem_flags &flags,
                            cl_uint size, void *data)
{
    cl_int err;
    mem = clCreateBuffer(_context, flags, size, data, &err);
    MCLASSERT(err);
}

void OCLWork::enqueue_image_read(cl_mem &mem, cl_uint width, cl_uint height,
                                 void *data)
{
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};

    cl_int err;
    err = clEnqueueReadImage(_command_queue, mem, CL_TRUE, origin, region, 0, 0,
                             data, 0, NULL, NULL);
    MCLASSERT(err);
}

void OCLWork::enqueue_image_write(cl_mem &mem, cl_uint width, cl_uint height,
                                  void *data)
{
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};

    cl_int err;
    err = clEnqueueWriteImage(_command_queue, mem, CL_TRUE, origin, region, 0,
                              0, data, 0, NULL, NULL);
    MCLASSERT(err);
}

void OCLWork::enqueue_buffer_read(cl_mem &mem, cl_uint size, void *data,
                                  cl_event &event)
{
    cl_int err;
    err = clEnqueueReadBuffer(_command_queue, mem, CL_TRUE, 0, size, data, 0,
                              NULL, NULL);
    clFinish(_command_queue);
    MCLASSERT(err);
}

void OCLWork::enqueue_buffer_write(cl_mem &mem, cl_uint size, void *data,
                                   cl_event &event)
{
    cl_int err;
    err = clEnqueueWriteBuffer(_command_queue, mem, CL_TRUE, 0, size, data, 0,
                               NULL, NULL);
    clFinish(_command_queue);
    MCLASSERT(err);
}

void OCLWork::enqueue_buffer_copy(cl_mem &mem_src, cl_mem &mem_dest,
                                  cl_uint size, cl_event &event)
{
    cl_int err;
    err = clEnqueueCopyBuffer(_command_queue, mem_src, mem_dest, 0, 0, size, 0,
                              NULL, NULL);
    MCLASSERT(err);
}

void OCLWork::enqueue_buffer_fill(cl_mem &mem_buf, cl_uint size_buf,
                                  void *fill_val, cl_uint size_fill,
                                  cl_event &event)
{
    cl_int err;

    err = clEnqueueFillBuffer(_command_queue, mem_buf, fill_val, size_fill, 0,
                              size_buf, 0, NULL, NULL);
    clFinish(_command_queue);
    MCLASSERT(err);
}

void OCLWork::enqueue_image_buffer_copy(cl_mem &mem_src, cl_mem &mem_dest,
                                        cl_uint width, cl_uint height,
                                        cl_event &event)
{
    cl_int err;
    const size_t src_origin[3] = {0, 0, 0};
    const size_t region[3] = {width, height, 1};

    err = clEnqueueCopyImageToBuffer(_command_queue, mem_src, mem_dest,
                                     src_origin, region, 0, 0, NULL, &event);
    MCLASSERT(err);
}

void OCLWork::get_platform_all_info()
{
    const int num_info = 4;
    cl_platform_info list_info[num_info] = {
        CL_PLATFORM_VERSION, CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
        CL_PLATFORM_EXTENSIONS};
    for (size_t i = 0; i < _count_platform; i++)
    {
        cl_int err;
        size_t size;
        err = clGetPlatformInfo(_platform_ids[i], CL_PLATFORM_NAME, 0, NULL,
                                &size);
        char *name = new char[size];
        err = clGetPlatformInfo(_platform_ids[i], CL_PLATFORM_NAME, size, name,
                                NULL);
        std::cout << "-----------" << name << "-----------" << std::endl;
        delete name;
        for (size_t j = 0; j < num_info; j++)
        {
            err = clGetPlatformInfo(_platform_ids[i], list_info[j], 0, NULL,
                                    &size);
            char *name = new char[size];
            err = clGetPlatformInfo(_platform_ids[i], list_info[j], size, name,
                                    NULL);
            std::cout << name << std::endl;
            delete name;
        }
        std::cout << "----------------------" << std::endl;
    }
}

std::string OCLWork::get_platform_info(const cl_platform_id &platform,
                                       const cl_platform_info &platform_info)
{
    cl_int err;
    size_t size;
    err = clGetPlatformInfo(platform, platform_info, 0, NULL, &size);
    char *info = new char[size];
    err = clGetPlatformInfo(platform, platform_info, size, info, NULL);
    return info;
}

void OCLWork::get_device_info()
{
    const int num_info = 4;
    cl_int err;
    size_t size;
    err = clGetDeviceInfo(_device_ids, CL_DEVICE_NAME, 0, NULL, &size);
    char name[256];
    err = clGetDeviceInfo(_device_ids, CL_DEVICE_NAME, size, name, NULL);

    std::cout << "Current device name: " << name << std::endl;
    err = clGetDeviceInfo(_device_ids, CL_DEVICE_OPENCL_C_VERSION, 0, NULL,
                          &size);
    err = clGetDeviceInfo(_device_ids, CL_DEVICE_OPENCL_C_VERSION, size, name,
                          NULL);
    std::cout << "OpenCL C version: " << name << std::endl;
}

void OCLWork::run_cl_kernel(cl_kernel &l_kernel, const cv::Vec2i &globalNDSize,
                            const cv::Vec2i &localNDSize, cl_event &e)
{
    size_t localWorkSize[2] = {localNDSize[0], localNDSize[1]};
    size_t globalWorkSize[2] = {globalNDSize[0], globalNDSize[1]};

    cl_int err;
    err = clEnqueueNDRangeKernel(_command_queue, l_kernel, 2, NULL,
                                 globalWorkSize, localWorkSize, 0, NULL, NULL);
    clFinish(_command_queue);
    MCLASSERT(err);
}
