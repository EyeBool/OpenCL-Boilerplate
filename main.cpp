#include <iostream>
#include <vector>

#include <CL/cl.h>

#include "fileLoader.h"
#include "CheckOpenCLError.h"

// program source
const char* programSource =
"__kernel void ADD(__global float* a, __global float* b, __global float* c) {\n"
"	const int i = get_global_id(0);\n"
"	c[i] = a[i] + b[i];\n"
"}";

// data
const cl_uint numberComponents = 5;
const size_t dataSize = numberComponents * sizeof(cl_float);
cl_float vec_a[numberComponents] = { 0.1, 0.25, 0.1, 3.1, 1.5 };
cl_float vec_b[numberComponents] = {2.0, 0.4, -0.1, 0.2, 0.4};
cl_float vec_c[numberComponents];

// to check for OpenCL errors
cl_int error;

int main() {

	// platforms
	// query for the number of platforms
	cl_uint platformIdCount = 0;
	error = clGetPlatformIDs(0, nullptr, &platformIdCount);
	CheckOpenCLError(error);

	// query for the set of platforms
	std::vector<cl_platform_id> platformIDs(platformIdCount);
	error = clGetPlatformIDs(platformIdCount, platformIDs.data(), nullptr);
	CheckOpenCLError(error);



	// devices
	// select a platform to work on
	cl_uint selectedPlatform = 0;

	// get selected platform name length
	size_t platformNameLength = 0;
	error = clGetPlatformInfo(
		platformIDs[selectedPlatform],
		CL_PLATFORM_NAME,
		0,
		0,
		&platformNameLength
	);
	CheckOpenCLError(error);

	// print selected platform name
	char* platformName = new char[platformNameLength];
	error = clGetPlatformInfo(
		platformIDs[selectedPlatform],
		CL_PLATFORM_NAME,
		platformNameLength,
		platformName,
		0
	);
	CheckOpenCLError(error);

	std::cout << "Platform: " << platformName << std::endl;

	// query for the number of devices on the selected platform
	cl_uint deviceIdCount = 0;
	error = clGetDeviceIDs(platformIDs[selectedPlatform], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);
	CheckOpenCLError(error);

	// query for the set of devices on the selected platform
	std::vector<cl_device_id> deviceIDs(deviceIdCount);
	clGetDeviceIDs(platformIDs[selectedPlatform], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIDs.data(), nullptr);



	// create a context
	/* const cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		reinterpret_cast<cl_context_properties> (platformIDs[0]),
		0,0
	};
	*/

	cl_context context;
	context = clCreateContext(
		nullptr,
		deviceIdCount,
		deviceIDs.data(),
		nullptr,
		nullptr,
		&error
	);
	CheckOpenCLError(error);



	// create a command queue
	/* command queue properties
	const cl_command_queue_properties commandQueueProperties[] = {
		CL_QUEUE_PROPERTIES,
		CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT,
		0
	};
	*/

	cl_command_queue queue;
	queue = clCreateCommandQueueWithProperties(
		context,
		deviceIDs[0],
		0,
		&error
	);
	CheckOpenCLError(error);



	// create data buffer objects
	cl_mem buffer_a;
	buffer_a = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		dataSize,
		nullptr,
		&error
	);
	CheckOpenCLError(error);

	cl_mem buffer_b;
	buffer_b = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		dataSize,
		nullptr,
		&error
	);
	CheckOpenCLError(error);

	cl_mem buffer_c;
	buffer_c = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		dataSize,
		nullptr,
		&error
	);
	CheckOpenCLError(error);



	// write host data to device memory
	error = clEnqueueWriteBuffer(
		queue,
		buffer_a,
		CL_FALSE,
		0,
		dataSize,
		vec_a,
		0,
		nullptr,
		nullptr
	);
	CheckOpenCLError(error);

	error = clEnqueueWriteBuffer(
		queue,
		buffer_b,
		CL_FALSE,
		0,
		dataSize,
		vec_b,
		0,
		nullptr,
		nullptr
	);
	CheckOpenCLError(error);


	// program

	// create program
	cl_program program;
	program = clCreateProgramWithSource(
		context,
		1,
		(const char**)&programSource,
		nullptr,
		&error
	);
	CheckOpenCLError(error);

	// build program
	error = clBuildProgram(
		program,
		deviceIdCount,
		deviceIDs.data(),
		nullptr,
		nullptr,
		nullptr
	);
	CheckOpenCLError(error);



	// create kernel
	cl_kernel kernel;
	kernel = clCreateKernel(
		program,
		"ADD",
		&error
	);
	CheckOpenCLError(error);

	// set kernel arguments
	error = clSetKernelArg(
		kernel,
		0,
		sizeof(cl_mem),
		&buffer_a
	);
	CheckOpenCLError(error);

	error = clSetKernelArg(
		kernel,
		1,
		sizeof(cl_mem),
		&buffer_b
	);
	CheckOpenCLError(error);

	error = clSetKernelArg(
		kernel,
		2,
		sizeof(cl_mem),
		&buffer_c
	);
	CheckOpenCLError(error);



	// configure the work-item structure
	// localWorkSize is not required
	// 	size_t localWorkSize = numberComponents;
	size_t globalWorkSize[1];
	globalWorkSize[0] = numberComponents;

	error = clEnqueueNDRangeKernel(
		queue,
		kernel,
		1,
		nullptr,
		globalWorkSize,
		nullptr,
		0,
		nullptr,
		nullptr
	);
	CheckOpenCLError(error);

	// finish queue
	error = clFinish(queue);
	CheckOpenCLError(error);

	// read buffer
	error = clEnqueueReadBuffer(
		queue,
		buffer_c,
		CL_TRUE,
		0,
		numberComponents * sizeof(cl_float),
		vec_c,
		0,
		nullptr,
		nullptr
	);
	CheckOpenCLError(error);

	// printout result
	for (int i = 0; i < numberComponents; i++) {
		std::cout << "c[" << i << "] = " << vec_c[i] << std::endl;
	}

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(buffer_a);
	clReleaseMemObject(buffer_b);
	clReleaseMemObject(buffer_c);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	system("pause");
	return 0;
}