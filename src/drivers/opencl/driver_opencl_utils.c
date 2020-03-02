/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

#include <common/config.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <starpu_opencl.h>
#include <starpu_profiling.h>
#include <core/workers.h>
#include <common/utils.h>
#include "driver_opencl_utils.h"
#include "driver_opencl.h"

#ifdef HAVE_CL_CL_EXT_H
#include <CL/cl_ext.h>
#endif

char *_starpu_opencl_program_dir;

#define _STARPU_STRINGIFY_(x) #x
#define _STARPU_STRINGIFY(x) _STARPU_STRINGIFY_(x)

static
int _starpu_opencl_locate_file(const char *source_file_name, char **located_file_name, char **located_dir_name)
{
	int ret = EXIT_FAILURE;

	*located_file_name = NULL;
	*located_dir_name = NULL;

	_STARPU_DEBUG("Trying to locate <%s>\n", source_file_name);
	if (access(source_file_name, R_OK) == 0)
	{
		_STARPU_CALLOC(*located_file_name, 1, strlen(source_file_name)+1);
		snprintf(*located_file_name, strlen(source_file_name)+1, "%s", source_file_name);
		ret = EXIT_SUCCESS;
	}

	if (ret == EXIT_FAILURE && _starpu_opencl_program_dir)
	{
		_STARPU_CALLOC(*located_file_name, 1, strlen(_starpu_opencl_program_dir)+1+strlen(source_file_name)+1);
		snprintf(*located_file_name, strlen(_starpu_opencl_program_dir)+1+strlen(source_file_name)+1, "%s/%s", _starpu_opencl_program_dir, source_file_name);
		_STARPU_DEBUG("Trying to locate <%s>\n", *located_file_name);
		if (access(*located_file_name, R_OK) == 0)
			ret = EXIT_SUCCESS;
	}

#ifdef STARPU_DEVEL
	if (ret == EXIT_FAILURE)
	{
		_STARPU_CALLOC(*located_file_name, 1, strlen(STARPU_SRC_DIR)+1+strlen(source_file_name)+1);
		snprintf(*located_file_name, strlen(STARPU_SRC_DIR)+1+strlen(source_file_name)+1, "%s/%s", STARPU_SRC_DIR, source_file_name);
		_STARPU_DEBUG("Trying to locate <%s>\n", *located_file_name);
		if (access(*located_file_name, R_OK) == 0)
			ret = EXIT_SUCCESS;
	}
#endif

	if (ret == EXIT_FAILURE)
	{
		_STARPU_CALLOC(*located_file_name, 1, strlen(_STARPU_STRINGIFY(STARPU_OPENCL_DATADIR))+1+strlen(source_file_name)+1);
		snprintf(*located_file_name, strlen(_STARPU_STRINGIFY(STARPU_OPENCL_DATADIR))+1+strlen(source_file_name)+1, "%s/%s", _STARPU_STRINGIFY(STARPU_OPENCL_DATADIR), source_file_name);
		_STARPU_DEBUG("Trying to locate <%s>\n", *located_file_name);
		if (access(*located_file_name, R_OK) == 0)
			ret = EXIT_SUCCESS;
	}

	if (ret == EXIT_FAILURE)
	{
		_STARPU_ERROR("Cannot locate file <%s>\n", source_file_name);
	}
	else
	{
		char *last = strrchr(*located_file_name, '/');

		if (!last)
		{
			_STARPU_CALLOC(*located_dir_name, 2, sizeof(char));
			snprintf(*located_dir_name, 2, "%s", "");
		}
		else
		{
			_STARPU_CALLOC(*located_dir_name, 1, 1+strlen(*located_file_name));
			snprintf(*located_dir_name, 1+strlen(*located_file_name), "%s", *located_file_name);
			(*located_dir_name)[strlen(*located_file_name)-strlen(last)+1] = '\0';
		}
	}

	return ret;
}

cl_int starpu_opencl_load_kernel(cl_kernel *kernel, cl_command_queue *queue, struct starpu_opencl_program *opencl_programs,
                                 const char *kernel_name, int devid)
{
	cl_int err;
	cl_device_id device;
	cl_program program;

	starpu_opencl_get_device(devid, &device);
	starpu_opencl_get_queue(devid, queue);

	program = opencl_programs->programs[devid];
	if (!program)
	{
		_STARPU_DISP("Program not available for device <%d>\n", devid);
		return CL_INVALID_PROGRAM;
	}

	// Create the compute kernel in the program we wish to run
	*kernel = clCreateKernel(program, kernel_name, &err);
	if (STARPU_UNLIKELY(err != CL_SUCCESS))
		STARPU_OPENCL_REPORT_ERROR(err);

	return CL_SUCCESS;
}

cl_int starpu_opencl_release_kernel(cl_kernel kernel)
{
	cl_int err;

	err = clReleaseKernel(kernel);
	if (STARPU_UNLIKELY(err != CL_SUCCESS))
		STARPU_OPENCL_REPORT_ERROR(err);

	return CL_SUCCESS;
}

static
char *_starpu_opencl_load_program_source(const char *filename)
{
	struct stat statbuf;
	FILE        *fh;
	char        *source;
	int         x;
	int         c;
	int         err;

	fh = fopen(filename, "r");
	if (!fh)
		return NULL;

	err = stat(filename, &statbuf);
	STARPU_ASSERT_MSG(err == 0, "could not open file %s\n", filename);
	_STARPU_MALLOC(source, statbuf.st_size + 1);

	for(c=fgetc(fh), x=0 ; c != EOF ; c =fgetc(fh), x++)
	{
		source[x] = (char)c;
	}
	source[x] = '\0';

	_STARPU_EXTRA_DEBUG("OpenCL kernel <%s>\n", source);

	fclose(fh);

	return source;
}

static
char *_starpu_opencl_load_program_binary(const char *filename, size_t *len)
{
	struct stat statbuf;
	FILE        *fh;
	char        *binary;
        int         err;

	fh = fopen(filename, "r");
	if (fh == 0)
		return NULL;

	err = stat(filename, &statbuf);
	STARPU_ASSERT_MSG(err == 0, "could not open file %s\n", filename);

	binary = (char *) malloc(statbuf.st_size);
	if (!binary)
	{
		fclose(fh);
		return binary;
	}

	err = fread(binary, statbuf.st_size, 1, fh);
	STARPU_ASSERT_MSG(err == 1, "could not read from file %s\n", filename);
	fclose(fh);

	*len = statbuf.st_size;
	return binary;
}

static
void _starpu_opencl_create_binary_directory(char *path, size_t maxlen)
{
	static int _directory_created = 0;

	snprintf(path, maxlen, "%s/.starpu/opencl/", _starpu_get_home_path());

	if (_directory_created == 0)
	{
		_STARPU_DEBUG("Creating directory %s\n", path);
		_starpu_mkpath_and_check(path, S_IRWXU);
		_directory_created = 1;
	}
}

char *_starpu_opencl_get_device_type_as_string(int id)
{
	cl_device_type type;

	type = _starpu_opencl_get_device_type(id);
	switch (type)
	{
		case CL_DEVICE_TYPE_GPU: return "gpu";
		case CL_DEVICE_TYPE_ACCELERATOR: return "acc";
		case CL_DEVICE_TYPE_CPU: return "cpu";
		default: return "unk";
	}
}

static
int _starpu_opencl_get_binary_name(char *binary_file_name, size_t maxlen, const char *source_file_name, int dev, cl_device_id device)
{
	char binary_directory[1024];
	char *p;
	cl_int err;
	cl_uint vendor_id;

	_starpu_opencl_create_binary_directory(binary_directory, sizeof(binary_directory));

	p = strrchr(source_file_name, '/');
	snprintf(binary_file_name, maxlen, "%s/%s", binary_directory, p?p:source_file_name);

	p = strstr(binary_file_name, ".cl");
	if (p == NULL) p=binary_file_name + strlen(binary_file_name);

	err = clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(vendor_id), &vendor_id, NULL);
	if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);

	sprintf(p, ".%s.vendor_id_%d_device_id_%d", _starpu_opencl_get_device_type_as_string(dev), (int)vendor_id, dev);

	return CL_SUCCESS;
}

static
int _starpu_opencl_compile_or_load_opencl_from_string(const char *opencl_program_source, const char* build_options,
						      struct starpu_opencl_program *opencl_programs, const char* source_file_name)
{
	unsigned int dev;
	unsigned int nb_devices;

	nb_devices = _starpu_opencl_get_device_count();
	// Iterate over each device
	for(dev = 0; dev < nb_devices; dev ++)
	{
		cl_device_id device;
		cl_context   context;
		cl_program   program;
		cl_int       err;

		if (opencl_programs)
		{
			opencl_programs->programs[dev] = NULL;
		}

		starpu_opencl_get_device(dev, &device);
		starpu_opencl_get_context(dev, &context);
		if (context == NULL)
		{
			_STARPU_DEBUG("[%u] is not a valid OpenCL context\n", dev);
			continue;
		}

		// Create the compute program from the source buffer
		program = clCreateProgramWithSource(context, 1, (const char **) &opencl_program_source, NULL, &err);
		if (!program || err != CL_SUCCESS)
		{
			_STARPU_DISP("Error: Failed to load program source with options %s!\n", build_options);
			return EXIT_FAILURE;
		}

		// Build the program executable
		err = clBuildProgram(program, 1, &device, build_options, NULL, NULL);

		// Get the status
		{
			cl_build_status status;
			size_t len;

			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
			if (len > 2)
			{
				char *buffer;
				_STARPU_MALLOC(buffer, len);

				clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, buffer, &len);
				_STARPU_DISP("Compilation output\n%s\n", buffer);

				free(buffer);
			}

			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, NULL);
			if (err != CL_SUCCESS || status != CL_BUILD_SUCCESS)
			{
				_STARPU_DISP("Error: Failed to build program executable!\n");
				_STARPU_DISP("clBuildProgram: %d - clGetProgramBuildInfo: %d\n", err, status);
				return EXIT_FAILURE;
			}
		}

		// Store program
		if (opencl_programs)
		{
			opencl_programs->programs[dev] = program;
		}
		else
		{
			char binary_file_name[2048];
			char *binary;
			size_t binary_len;
			FILE *fh;

			err = _starpu_opencl_get_binary_name(binary_file_name, sizeof(binary_file_name), source_file_name, dev, device);
			if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);

			err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_len, NULL);
			if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);
			_STARPU_MALLOC(binary, binary_len);

			err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(binary), &binary, NULL);
			if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);

			fh = fopen(binary_file_name, "w");
			if (fh == NULL)
			{
				_STARPU_DISP("Error: Failed to open file <%s>\n", binary_file_name);
				perror("fopen");
				return EXIT_FAILURE;
			}
			fwrite(binary, binary_len, 1, fh);
			fclose(fh);
			free(binary);
			_STARPU_DEBUG("File <%s> created\n", binary_file_name);

			err = clReleaseProgram(program);
			if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);
		}
	}
	return EXIT_SUCCESS;
}

void starpu_opencl_load_program_source_malloc(const char *source_file_name, char **located_file_name, char **located_dir_name, char **opencl_program_source)
{
	// Locate source file
	_starpu_opencl_locate_file(source_file_name, located_file_name, located_dir_name);
	_STARPU_DEBUG("Source file name : <%s>\n", *located_file_name);
	_STARPU_DEBUG("Source directory name : <%s>\n", *located_dir_name);

	// Load the compute program from disk into a char *
	char *source = _starpu_opencl_load_program_source(*located_file_name);
	if(!source)
		_STARPU_ERROR("Failed to load compute program from file <%s>!\n", *located_file_name);

	_STARPU_MALLOC(*opencl_program_source, strlen(source)+1);
	snprintf(*opencl_program_source, strlen(source)+1, "%s", source);
	free(source);
}

void starpu_opencl_load_program_source(const char *source_file_name, char *located_file_name, char *located_dir_name, char *opencl_program_source)
{
	char *_located_file_name;
	char *_located_dir_name;

	// Locate source file
	_starpu_opencl_locate_file(source_file_name, &_located_file_name, &_located_dir_name);
	_STARPU_DEBUG("Source file name : <%s>\n", _located_file_name);
	_STARPU_DEBUG("Source directory name : <%s>\n", _located_dir_name);

	// Load the compute program from disk into a char *
	char *source = _starpu_opencl_load_program_source(_located_file_name);
	if(!source)
		_STARPU_ERROR("Failed to load compute program from file <%s>!\n", _located_file_name);

	sprintf(located_file_name, "%s", _located_file_name);
	free(_located_file_name);
	sprintf(located_dir_name, "%s", _located_dir_name);
	free(_located_dir_name);
	sprintf(opencl_program_source, "%s", source);
	free(source);
}

static
int _starpu_opencl_compile_or_load_opencl_from_file(const char *source_file_name, struct starpu_opencl_program *opencl_programs, const char* build_options)
{
	int nb_devices;
	int ret;
	char *located_file_name;
	char *located_dir_name;
	char new_build_options[1024];
	char *opencl_program_source;

	// Do not try to load and compile the file if there is no devices
	nb_devices = starpu_opencl_worker_get_count();
	if (nb_devices == 0) return EXIT_SUCCESS;

	starpu_opencl_load_program_source_malloc(source_file_name, &located_file_name, &located_dir_name, &opencl_program_source);

	if (!build_options)
		build_options = "";

	if (!strcmp(located_dir_name, ""))
	{
		snprintf(new_build_options, sizeof(new_build_options), "%s", build_options);
	}
	else
	{
		snprintf(new_build_options, sizeof(new_build_options), "-I %s %s", located_dir_name, build_options);
	}
	_STARPU_DEBUG("Build options: <%s>\n", new_build_options);

	ret = _starpu_opencl_compile_or_load_opencl_from_string(opencl_program_source, new_build_options, opencl_programs, source_file_name);

	_STARPU_DEBUG("located_file_name : <%s>\n", located_file_name);
	_STARPU_DEBUG("located_dir_name : <%s>\n", located_dir_name);
	free(located_file_name);
	free(located_dir_name);
	free(opencl_program_source);

	return ret;
}

int starpu_opencl_compile_opencl_from_file(const char *source_file_name, const char* build_options)
{
	return _starpu_opencl_compile_or_load_opencl_from_file(source_file_name, NULL, build_options);
}

int starpu_opencl_compile_opencl_from_string(const char *opencl_program_source, const char *file_name, const char* build_options)
{
	return _starpu_opencl_compile_or_load_opencl_from_string(opencl_program_source, build_options, NULL, file_name);
}

int starpu_opencl_load_opencl_from_string(const char *opencl_program_source, struct starpu_opencl_program *opencl_programs,
					  const char* build_options)
{
	return _starpu_opencl_compile_or_load_opencl_from_string(opencl_program_source, build_options, opencl_programs, NULL);
}

int starpu_opencl_load_opencl_from_file(const char *source_file_name, struct starpu_opencl_program *opencl_programs,
					const char* build_options)
{
	return _starpu_opencl_compile_or_load_opencl_from_file(source_file_name, opencl_programs, build_options);
}

int starpu_opencl_load_binary_opencl(const char *kernel_id, struct starpu_opencl_program *opencl_programs)
{
	unsigned int dev;
	unsigned int nb_devices;

	nb_devices = _starpu_opencl_get_device_count();
	// Iterate over each device
	for(dev = 0; dev < nb_devices; dev ++)
	{
		cl_device_id device;
		cl_context   context;
		cl_program   program;
		cl_int       err;
		char        *binary;
		char         binary_file_name[1024];
		size_t       length;
		cl_int       binary_status;

		opencl_programs->programs[dev] = NULL;

		starpu_opencl_get_device(dev, &device);
		starpu_opencl_get_context(dev, &context);
		if (context == NULL)
		{
			_STARPU_DEBUG("[%u] is not a valid OpenCL context\n", dev);
			continue;
		}

		// Load the binary buffer
		err = _starpu_opencl_get_binary_name(binary_file_name, sizeof(binary_file_name), kernel_id, dev, device);
		if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);
		binary = _starpu_opencl_load_program_binary(binary_file_name, &length);

		// Create the compute program from the binary buffer
		program = clCreateProgramWithBinary(context, 1, &device, &length, (const unsigned char **) &binary, &binary_status, &err);
		if (!program || err != CL_SUCCESS)
		{
			_STARPU_DISP("Error: Failed to load program binary!\n");
			return EXIT_FAILURE;
		}

		// Build the program executable
		err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

		// Get the status
		{
			cl_build_status status;
			size_t len;

			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
			if (len > 2)
			{
				char *buffer;
				_STARPU_MALLOC(buffer, len);

				clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, buffer, &len);
				_STARPU_DISP("Compilation output\n%s\n", buffer);

				free(buffer);
			}

			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, NULL);
			if (err != CL_SUCCESS || status != CL_BUILD_SUCCESS)
			{
				_STARPU_DISP("Error: Failed to build program executable!\n");
				_STARPU_DISP("clBuildProgram: %d - clGetProgramBuildInfo: %d\n", err, status);
				return EXIT_FAILURE;
			}
		}

		// Store program
		opencl_programs->programs[dev] = program;
		free(binary);
	}
	return 0;
}

int starpu_opencl_unload_opencl(struct starpu_opencl_program *opencl_programs)
{
	unsigned int dev;
	unsigned int nb_devices;

	if (!starpu_opencl_worker_get_count())
		return 0;

	nb_devices = _starpu_opencl_get_device_count();
	// Iterate over each device
	for(dev = 0; dev < nb_devices; dev ++)
	{
		if (opencl_programs->programs[dev])
		{
			cl_int err;
			err = clReleaseProgram(opencl_programs->programs[dev]);
			if (STARPU_UNLIKELY(err != CL_SUCCESS))
				STARPU_OPENCL_REPORT_ERROR(err);
		}
	}
	return 0;
}

int starpu_opencl_collect_stats(cl_event event STARPU_ATTRIBUTE_UNUSED)
{
#if defined(CL_PROFILING_CLOCK_CYCLE_COUNT)||defined(CL_PROFILING_STALL_CYCLE_COUNT)||defined(CL_PROFILING_POWER_CONSUMED)
	struct starpu_task *task = starpu_task_get_current();
	struct starpu_profiling_task_info *info = task->profiling_info;
#endif

#ifdef CL_PROFILING_CLOCK_CYCLE_COUNT
	if (starpu_profiling_status_get() && info)
	{
		cl_int err;
		unsigned int clock_cycle_count;
		size_t size;
		err = clGetEventProfilingInfo(event, CL_PROFILING_CLOCK_CYCLE_COUNT, sizeof(clock_cycle_count), &clock_cycle_count, &size);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
		STARPU_ASSERT(size == sizeof(clock_cycle_count));
		info->used_cycles += clock_cycle_count;
	}
#endif
#ifdef CL_PROFILING_STALL_CYCLE_COUNT
	if (starpu_profiling_status_get() && info)
	{
		cl_int err;
		unsigned int stall_cycle_count;
		size_t size;
		err = clGetEventProfilingInfo(event, CL_PROFILING_STALL_CYCLE_COUNT, sizeof(stall_cycle_count), &stall_cycle_count, &size);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
		STARPU_ASSERT(size == sizeof(stall_cycle_count));

		info->stall_cycles += stall_cycle_count;
	}
#endif
#ifdef CL_PROFILING_POWER_CONSUMED
	if (info && (starpu_profiling_status_get() || (task->cl && task->cl->energy_model && task->cl->energy_model->benchmarking)))
	{
		cl_int err;
		double energy_consumed;
		size_t size;
		err = clGetEventProfilingInfo(event, CL_PROFILING_POWER_CONSUMED, sizeof(energy_consumed), &energy_consumed, &size);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
		STARPU_ASSERT(size == sizeof(energy_consumed));

		info->energy_consumed += energy_consumed;
	}
#endif

	return 0;
}

const char *starpu_opencl_error_string(cl_int status)
{
	const char *errormsg;
	switch (status)
	{
	case CL_SUCCESS:
		errormsg = "Success";
		break;
	case CL_DEVICE_NOT_FOUND:
		errormsg = "Device not found";
		break;
	case CL_DEVICE_NOT_AVAILABLE:
		errormsg = "Device not available";
		break;
	case CL_COMPILER_NOT_AVAILABLE:
		errormsg = "Compiler not available";
		break;
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		errormsg = "Memory object allocation failure";
		break;
	case CL_OUT_OF_RESOURCES:
		errormsg = "Out of resources";
		break;
	case CL_OUT_OF_HOST_MEMORY:
		errormsg = "Out of host memory";
		break;
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		errormsg = "Profiling info not available";
		break;
	case CL_MEM_COPY_OVERLAP:
		errormsg = "Memory copy overlap";
		break;
	case CL_IMAGE_FORMAT_MISMATCH:
		errormsg = "Image format mismatch";
		break;
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		errormsg = "Image format not supported";
		break;
	case CL_BUILD_PROGRAM_FAILURE:
		errormsg = "Build program failure";
		break;
	case CL_MAP_FAILURE:
		errormsg = "Map failure";
		break;
	case CL_INVALID_VALUE:
		errormsg = "Invalid value";
		break;
	case CL_INVALID_DEVICE_TYPE:
		errormsg = "Invalid device type";
		break;
	case CL_INVALID_PLATFORM:
		errormsg = "Invalid platform";
		break;
	case CL_INVALID_DEVICE:
		errormsg = "Invalid device";
		break;
	case CL_INVALID_CONTEXT:
		errormsg = "Invalid context";
		break;
	case CL_INVALID_QUEUE_PROPERTIES:
		errormsg = "Invalid queue properties";
		break;
	case CL_INVALID_COMMAND_QUEUE:
		errormsg = "Invalid command queue";
		break;
	case CL_INVALID_HOST_PTR:
		errormsg = "Invalid host pointer";
		break;
	case CL_INVALID_MEM_OBJECT:
		errormsg = "Invalid memory object";
		break;
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		errormsg = "Invalid image format descriptor";
		break;
	case CL_INVALID_IMAGE_SIZE:
		errormsg = "Invalid image size";
		break;
	case CL_INVALID_SAMPLER:
		errormsg = "Invalid sampler";
		break;
	case CL_INVALID_BINARY:
		errormsg = "Invalid binary";
		break;
	case CL_INVALID_BUILD_OPTIONS:
		errormsg = "Invalid build options";
		break;
	case CL_INVALID_PROGRAM:
		errormsg = "Invalid program";
		break;
	case CL_INVALID_PROGRAM_EXECUTABLE:
		errormsg = "Invalid program executable";
		break;
	case CL_INVALID_KERNEL_NAME:
		errormsg = "Invalid kernel name";
		break;
	case CL_INVALID_KERNEL_DEFINITION:
		errormsg = "Invalid kernel definition";
		break;
	case CL_INVALID_KERNEL:
		errormsg = "Invalid kernel";
		break;
	case CL_INVALID_ARG_INDEX:
		errormsg = "Invalid argument index";
		break;
	case CL_INVALID_ARG_VALUE:
		errormsg = "Invalid argument value";
		break;
	case CL_INVALID_ARG_SIZE:
		errormsg = "Invalid argument size";
		break;
	case CL_INVALID_KERNEL_ARGS:
		errormsg = "Invalid kernel arguments";
		break;
	case CL_INVALID_WORK_DIMENSION:
		errormsg = "Invalid work dimension";
		break;
	case CL_INVALID_WORK_GROUP_SIZE:
		errormsg = "Invalid work group size";
		break;
	case CL_INVALID_WORK_ITEM_SIZE:
		errormsg = "Invalid work item size";
		break;
	case CL_INVALID_GLOBAL_OFFSET:
		errormsg = "Invalid global offset";
		break;
	case CL_INVALID_EVENT_WAIT_LIST:
		errormsg = "Invalid event wait list";
		break;
	case CL_INVALID_EVENT:
		errormsg = "Invalid event";
		break;
	case CL_INVALID_OPERATION:
		errormsg = "Invalid operation";
		break;
	case CL_INVALID_GL_OBJECT:
		errormsg = "Invalid GL object";
		break;
	case CL_INVALID_BUFFER_SIZE:
		errormsg = "Invalid buffer size";
		break;
	case CL_INVALID_MIP_LEVEL:
		errormsg = "Invalid MIP level";
		break;
#ifdef CL_PLATFORM_NOT_FOUND_KHR
	case CL_PLATFORM_NOT_FOUND_KHR:
		errormsg = "Platform not found";
		break;
#endif
	default:
		errormsg = "unknown OpenCL error";
		break;
	}
	return errormsg;
}

void starpu_opencl_display_error(const char *func, const char *file, int line, const char* msg, cl_int status)
{
	_STARPU_MSG("oops in %s (%s:%d) (%s) ... <%s> (%d) \n", func, file, line, msg, starpu_opencl_error_string (status), status);
}

int starpu_opencl_set_kernel_args(cl_int *error, cl_kernel *kernel, ...)
{
	int i;
	va_list ap;

	va_start(ap, kernel);

	for (i = 0; ; i++)
	{
		int size = va_arg(ap, int);
		if (size == 0)
			break;

		cl_mem *ptr = va_arg(ap, cl_mem *);
		int err = clSetKernelArg(*kernel, i, size, ptr);
		if (STARPU_UNLIKELY(err != CL_SUCCESS))
		{
			*error = err;
			break;
		}
	}

	va_end(ap);
	return i;
}
