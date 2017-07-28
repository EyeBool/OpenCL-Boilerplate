#include <iostream>
#include <CL/cl.h>

void CheckOpenCLError(cl_int error)
{
	if (error != CL_SUCCESS)
		switch (error)
		{
		case CL_DEVICE_NOT_FOUND: std::cerr << "OpenCL error: " << "CL_DEVICE_NOT_FOUND" << std::endl; break;
		case CL_DEVICE_NOT_AVAILABLE: std::cerr << "OpenCL error: " << "CL_DEVICE_NOT_AVAILABLE" << std::endl; break;
		case CL_COMPILER_NOT_AVAILABLE: std::cerr << "OpenCL error: " << "CL_COMPILER_NOT_AVAILABLE" << std::endl; break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE: std::cerr << "OpenCL error: " << "CL_MEM_OBJECT_ALLOCATION_FAILURE" << std::endl; break;
		case CL_OUT_OF_RESOURCES: std::cerr << "OpenCL error: " << "CL_OUT_OF_RESOURCES" << std::endl; break;
		case CL_OUT_OF_HOST_MEMORY: std::cerr << "OpenCL error: " << "CL_OUT_OF_HOST_MEMORY" << std::endl; break;
		case CL_PROFILING_INFO_NOT_AVAILABLE: std::cerr << "OpenCL error: " << "CL_PROFILING_INFO_NOT_AVAILABLE" << std::endl; break;
		case CL_MEM_COPY_OVERLAP: std::cerr << "OpenCL error: " << "CL_MEM_COPY_OVERLAP" << std::endl; break;
		case CL_IMAGE_FORMAT_MISMATCH: std::cerr << "OpenCL error: " << "CL_IMAGE_FORMAT_MISMATCH" << std::endl; break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED: std::cerr << "OpenCL error: " << "CL_IMAGE_FORMAT_NOT_SUPPORTED" << std::endl; break;
		case CL_BUILD_PROGRAM_FAILURE: std::cerr << "OpenCL error: " << "CL_BUILD_PROGRAM_FAILURE" << std::endl; break;
		case CL_MAP_FAILURE: std::cerr << "OpenCL error: " << "CL_MAP_FAILURE" << std::endl; break;
		case CL_MISALIGNED_SUB_BUFFER_OFFSET: std::cerr << "OpenCL error: " << "CL_MISALIGNED_SUB_BUFFER_OFFSET" << std::endl; break;
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: std::cerr << "OpenCL error: " << "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST" << std::endl; break;
		case CL_COMPILE_PROGRAM_FAILURE: std::cerr << "OpenCL error: " << "CL_COMPILE_PROGRAM_FAILURE" << std::endl; break;
		case CL_LINKER_NOT_AVAILABLE: std::cerr << "OpenCL error: " << "CL_LINKER_NOT_AVAILABLE" << std::endl; break;
		case CL_LINK_PROGRAM_FAILURE: std::cerr << "OpenCL error: " << "CL_LINK_PROGRAM_FAILURE" << std::endl; break;
		case CL_DEVICE_PARTITION_FAILED: std::cerr << "OpenCL error: " << "CL_DEVICE_PARTITION_FAILED" << std::endl; break;
		case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: std::cerr << "OpenCL error: " << "CL_KERNEL_ARG_INFO_NOT_AVAILABLE" << std::endl; break;
		case CL_INVALID_VALUE: std::cerr << "OpenCL error: " << "CL_INVALID_VALUE" << std::endl; break;
		case CL_INVALID_DEVICE_TYPE: std::cerr << "OpenCL error: " << "CL_INVALID_DEVICE_TYPE" << std::endl; break;
		case CL_INVALID_PLATFORM: std::cerr << "OpenCL error: " << "CL_INVALID_PLATFORM" << std::endl; break;
		case CL_INVALID_DEVICE: std::cerr << "OpenCL error: " << "CL_INVALID_DEVICE" << std::endl; break;
		case CL_INVALID_CONTEXT: std::cerr << "OpenCL error: " << "CL_INVALID_CONTEXT" << std::endl; break;
		case CL_INVALID_QUEUE_PROPERTIES: std::cerr << "OpenCL error: " << "CL_INVALID_QUEUE_PROPERTIES" << std::endl; break;
		case CL_INVALID_COMMAND_QUEUE: std::cerr << "OpenCL error: " << "CL_INVALID_COMMAND_QUEUE" << std::endl; break;
		case CL_INVALID_HOST_PTR: std::cerr << "OpenCL error: " << "CL_INVALID_HOST_PTR" << std::endl; break;
		case CL_INVALID_MEM_OBJECT: std::cerr << "OpenCL error: " << "CL_INVALID_MEM_OBJECT" << std::endl; break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: std::cerr << "OpenCL error: " << "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR" << std::endl; break;
		case CL_INVALID_IMAGE_SIZE: std::cerr << "OpenCL error: " << "CL_INVALID_IMAGE_SIZE" << std::endl; break;
		case CL_INVALID_SAMPLER: std::cerr << "OpenCL error: " << "CL_INVALID_SAMPLER" << std::endl; break;
		case CL_INVALID_BINARY: std::cerr << "OpenCL error: " << "CL_INVALID_BINARY" << std::endl; break;
		case CL_INVALID_BUILD_OPTIONS: std::cerr << "OpenCL error: " << "CL_INVALID_BUILD_OPTIONS" << std::endl; break;
		case CL_INVALID_PROGRAM: std::cerr << "OpenCL error: " << "CL_INVALID_PROGRAM" << std::endl; break;
		case CL_INVALID_PROGRAM_EXECUTABLE: std::cerr << "OpenCL error: " << "CL_INVALID_PROGRAM_EXECUTABLE" << std::endl; break;
		case CL_INVALID_KERNEL_NAME: std::cerr << "OpenCL error: " << "CL_INVALID_KERNEL_NAME" << std::endl; break;
		case CL_INVALID_KERNEL_DEFINITION: std::cerr << "OpenCL error: " << "CL_INVALID_KERNEL_DEFINITION" << std::endl; break;
		case CL_INVALID_KERNEL: std::cerr << "OpenCL error: " << "CL_INVALID_KERNEL" << std::endl; break;
		case CL_INVALID_ARG_INDEX: std::cerr << "OpenCL error: " << "CL_INVALID_ARG_INDEX" << std::endl; break;
		case CL_INVALID_ARG_VALUE: std::cerr << "OpenCL error: " << "CL_INVALID_ARG_VALUE" << std::endl; break;
		case CL_INVALID_ARG_SIZE: std::cerr << "OpenCL error: " << "CL_INVALID_ARG_SIZE" << std::endl; break;
		case CL_INVALID_KERNEL_ARGS: std::cerr << "OpenCL error: " << "CL_INVALID_KERNEL_ARGS" << std::endl; break;
		case CL_INVALID_WORK_DIMENSION: std::cerr << "OpenCL error: " << "CL_INVALID_WORK_DIMENSION" << std::endl; break;
		case CL_INVALID_WORK_GROUP_SIZE: std::cerr << "OpenCL error: " << "CL_INVALID_WORK_GROUP_SIZE" << std::endl; break;
		case CL_INVALID_WORK_ITEM_SIZE: std::cerr << "OpenCL error: " << "CL_INVALID_WORK_ITEM_SIZE" << std::endl; break;
		case CL_INVALID_GLOBAL_OFFSET: std::cerr << "OpenCL error: " << "CL_INVALID_GLOBAL_OFFSET" << std::endl; break;
		case CL_INVALID_EVENT_WAIT_LIST: std::cerr << "OpenCL error: " << "CL_INVALID_EVENT_WAIT_LIST" << std::endl; break;
		case CL_INVALID_EVENT: std::cerr << "OpenCL error: " << "CL_INVALID_EVENT" << std::endl; break;
		case CL_INVALID_OPERATION: std::cerr << "OpenCL error: " << "CL_INVALID_OPERATION" << std::endl; break;
		case CL_INVALID_GL_OBJECT: std::cerr << "OpenCL error: " << "CL_INVALID_GL_OBJECT" << std::endl; break;
		case CL_INVALID_BUFFER_SIZE: std::cerr << "OpenCL error: " << "CL_INVALID_BUFFER_SIZE" << std::endl; break;
		case CL_INVALID_MIP_LEVEL: std::cerr << "OpenCL error: " << "CL_INVALID_MIP_LEVEL" << std::endl; break;
		case CL_INVALID_GLOBAL_WORK_SIZE: std::cerr << "OpenCL error: " << "CL_INVALID_GLOBAL_WORK_SIZE" << std::endl; break;
		case CL_INVALID_PROPERTY: std::cerr << "OpenCL error: " << "CL_INVALID_PROPERTY" << std::endl; break;
		case CL_INVALID_IMAGE_DESCRIPTOR: std::cerr << "OpenCL error: " << "CL_INVALID_IMAGE_DESCRIPTOR" << std::endl; break;
		case CL_INVALID_COMPILER_OPTIONS: std::cerr << "OpenCL error: " << "CL_INVALID_COMPILER_OPTIONS" << std::endl; break;
		case CL_INVALID_LINKER_OPTIONS: std::cerr << "OpenCL error: " << "CL_INVALID_LINKER_OPTIONS" << std::endl; break;
		case CL_INVALID_DEVICE_PARTITION_COUNT: std::cerr << "OpenCL error: " << "CL_INVALID_DEVICE_PARTITION_COUNT" << std::endl; break;
		case CL_INVALID_PIPE_SIZE: std::cerr << "OpenCL error: " << "CL_INVALID_PIPE_SIZE" << std::endl; break;
		case CL_INVALID_DEVICE_QUEUE: std::cerr << "OpenCL error: " << "CL_INVALID_DEVICE_QUEUE" << std::endl; break;
		default: std::cerr << "OpenCL error flag: " << error << std::endl;
		}
}