/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
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
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <starpu_opencl.h>
#include <common/list.h>
#include <common/htable32.h>
#include <core/workers.h>
#include "driver_opencl_utils.h"
#include "driver_opencl.h"

#define CRC32C_POLY_BE 0x1EDC6F41

static
inline uint32_t __attribute__ ((pure)) crc32_be_8(uint8_t inputbyte, uint32_t inputcrc)
{
	unsigned i;
	uint32_t crc;

	crc = inputcrc ^ (inputbyte << 24);
	for (i = 0; i < 8; i++)
		crc = (crc << 1) ^ ((crc & 0x80000000) ? CRC32C_POLY_BE : 0);

	return crc;
}

static
uint32_t crc32_string(char *str)
{
	uint32_t hash = 0;

	size_t len = strlen(str);

	unsigned i;
	for (i = 0; i < len; i++)
	{
		hash = crc32_be_8((uint8_t)str[i], hash);
	}

	return hash;
}

static
cl_uint _starpu_opencl_device_uniqueid(cl_device_id id)
{
	char name[1024];
	cl_int  err;

	err = clGetDeviceInfo(id, CL_DEVICE_NAME, 1024, name, NULL);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
	//  fprintf(stderr, "name %s\n", name);

	return crc32_string(name);
}

char *_starpu_opencl_codelet_dir;

#define _STARPU_STRINGIFY_(x) #x
#define _STARPU_STRINGIFY(x) _STARPU_STRINGIFY_(x)

static
int _starpu_opencl_locate_file(char *source_file_name, char *located_file_name) {
        _STARPU_OPENCL_DEBUG("Trying to locate <%s>\n", source_file_name);
        if (access(source_file_name, R_OK) == 0) {
                strcpy(located_file_name, source_file_name);
                return EXIT_SUCCESS;
        }
        if (_starpu_opencl_codelet_dir) {
                sprintf(located_file_name, "%s/%s", _starpu_opencl_codelet_dir, source_file_name);
                _STARPU_OPENCL_DEBUG("Trying to locate <%s>\n", located_file_name);
                if (access(located_file_name, R_OK) == 0) return EXIT_SUCCESS;
        }
        sprintf(located_file_name, "%s/%s", _STARPU_STRINGIFY(STARPU_OPENCL_DATADIR), source_file_name);
        _STARPU_OPENCL_DEBUG("Trying to locate <%s>\n", located_file_name);
        if (access(located_file_name, R_OK) == 0) return EXIT_SUCCESS;
        sprintf(located_file_name, "%s/%s", STARPU_SRC_DIR, source_file_name);
        _STARPU_OPENCL_DEBUG("Trying to locate <%s>\n", located_file_name);
        if (access(located_file_name, R_OK) == 0) return EXIT_SUCCESS;

        strcpy(located_file_name, "");
        OPENCL_ERROR("Cannot locate file <%s>\n", source_file_name);
        return EXIT_FAILURE;
}

static
unsigned char *_starpu_opencl_load_program_binary(char *filename, size_t *len)
{
	struct stat statbuf;
	FILE        *fh;
	unsigned char        *binary;

	fh = fopen(filename, "r");
	if (fh == 0)
		return NULL;

	stat(filename, &statbuf);

	binary = (unsigned char *) malloc(statbuf.st_size);
	if(!binary)
		return binary;

	fread(binary, statbuf.st_size, 1, fh);

	*len = statbuf.st_size;
	return binary;
}

static
char *_starpu_basename(const char *name)
{
        const char *base = name;
        while (*name) {
                if (*name++ == '/') {
                        base = name;
                }
        }
        return (char *) base;
}

static
int _starpu_opencl_get_binary_filename(char *program_name, cl_device_id device, char *binary_filename)
{
        char *p;
        char *basename;
	cl_uint uniqueid;
        cl_device_type type;
        uid_t uid;
        int err;

        // Get type of device
        err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
	if (err != CL_SUCCESS) {
                STARPU_OPENCL_REPORT_ERROR(err);
                return err;
        }

	uniqueid = _starpu_opencl_device_uniqueid(device);
        uid = getuid();

        sprintf(binary_filename, "/tmp/%llu_", (long long unsigned int)uid);
        basename = _starpu_basename(program_name);
        strcat(binary_filename, basename);
        p = strstr(binary_filename, ".cl");
	if (p == NULL) {
                OPENCL_ERROR("Program file name doesn't have the '.cl' extension!\n");
                return EXIT_FAILURE;
        }

        strcpy(p, (type == CL_DEVICE_TYPE_GPU) ? ".gpu." : ".cpu.");
	sprintf(p + strlen(p), "%u", uniqueid);

        return EXIT_SUCCESS;
}

static
cl_int _starpu_opencl_load_program(cl_context context, char *program_name, cl_device_id device, cl_program *program)
{
        //	cl_program     program;
        const unsigned char *binary;
	size_t         len;
	cl_int         err;
	cl_int         status;

	char     binary_filename[1024];

        // Get the name of the binary file
        _starpu_opencl_get_binary_filename(program_name, device, binary_filename);

        // Load the binary file
	binary = _starpu_opencl_load_program_binary(binary_filename, &len);
	if(binary == NULL)
		OPENCL_ERROR("Cannot load binary file %s\n", binary_filename);

	//_STARPU_OPENCL_DEBUG("[%s] binary file loaded.\n", binary_filename);
	*program = clCreateProgramWithBinary(context, 1, &device, &len, &binary, &status, &err);
	if (err != CL_SUCCESS) {
                STARPU_OPENCL_REPORT_ERROR(err);
                return err;
        }

	// Build the program executable
	err = clBuildProgram(*program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];

		fprintf(stderr, "Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		fprintf(stderr, "%s\n", buffer);
                return err;
	}

	return CL_SUCCESS;
}

static struct starpu_htbl32_node_s *history_program_hash[STARPU_MAXOPENCLDEVS] = {NULL};
LIST_TYPE(program,
          char *program_name;
          cl_program program;
          );
program_list_t history_program_list[STARPU_MAXOPENCLDEVS];

int _starpu_opencl_init_programs(int dev)
{
        history_program_list[dev] = program_list_new();
        return CL_SUCCESS;
}

int _starpu_opencl_release_programs(int dev)
{
        while (!program_list_empty(history_program_list[dev])) {
                program_t pp = program_list_pop_front(history_program_list[dev]);
                _STARPU_OPENCL_DEBUG("Releasing program=<%s> on dev=<%d>\n", pp->program_name, dev);
                clReleaseProgram(pp->program);
        }
        program_list_delete(history_program_list[dev]);
        return CL_SUCCESS;
}

int starpu_opencl_load_kernel(cl_kernel *kernel, cl_command_queue *queue, char *program_name, char *kernel_name, int devid)
{
        int err;
	cl_device_id device;
        cl_context context;
        uint32_t key;
        cl_program program;

        starpu_opencl_get_device(devid, &device);
        starpu_opencl_get_context(devid, &context);
        starpu_opencl_get_queue(devid, queue);

        key = crc32_string(program_name);
        program = _starpu_htbl_search_32(history_program_hash[devid], key);
        if (!program) {
                err = _starpu_opencl_load_program(context, program_name, device, &program);
                if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
                _starpu_htbl_insert_32(&(history_program_hash[devid]), key, program);
                program_t pp = program_new();
                pp->program_name = program_name;
                pp->program = program;
                program_list_push_front(history_program_list[devid], pp);
        }

        // Create the compute kernel in the program we wish to run
        *kernel = clCreateKernel(program, kernel_name, &err);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	return CL_SUCCESS;
}

int starpu_opencl_release(cl_kernel kernel) {
	cl_int err;

	err = clReleaseKernel(kernel);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

        return CL_SUCCESS;
}

static
char *_starpu_opencl_load_program_source(const char *filename)
{
        struct stat statbuf;
        FILE        *fh;
        char        *source;

        fh = fopen(filename, "r");
        if (fh == 0)
                return NULL;

        stat(filename, &statbuf);
        source = (char *) malloc(statbuf.st_size + 1);
        fread(source, statbuf.st_size, 1, fh);
        source[statbuf.st_size] = '\0';

        fclose(fh);

        return source;
}

static
int _starpu_opencl_store_program_binary(const char *filename, const char *binary, size_t len)
{
        FILE *fh;

        fh = fopen(filename, "w");
        if(fh == NULL) {
                perror("fopen"); return EXIT_FAILURE;
        }

        fwrite(binary, len, 1, fh);
        fclose(fh);

        return EXIT_SUCCESS;
}

int _starpu_opencl_compile_source_to_opencl(char *source_file_name)
{
        int              err;
        int              device_type = CL_DEVICE_TYPE_ALL;
        cl_device_id     devices[STARPU_MAXOPENCLDEVS];
        unsigned         max = STARPU_MAXOPENCLDEVS;
        unsigned         nb_devices = 0;
        cl_uint          history[STARPU_MAXOPENCLDEVS]; // To track similar devices
        char             preproc_file_name[1024];
        char             located_file_name[1024];
        cl_platform_id   platform_ids[STARPU_OPENCL_PLATFORM_MAX];
        cl_uint          platform, nb_platforms;
        char             *basename;

        // Locate source file
        _starpu_opencl_locate_file(source_file_name, located_file_name);
        _STARPU_OPENCL_DEBUG("Source file name : <%s>\n", located_file_name);
        basename = _starpu_basename(located_file_name);

        // Prepare preprocessor temporary filename
        {
                char *p;
                strcpy(preproc_file_name, basename);
                p = strstr(preproc_file_name, ".cl");
                if(p == NULL)
                        OPENCL_ERROR("Kernel file name doesn't have the '.cl' extension!\n");
                strcpy(p, ".pre");
        }

        // Get Platforms
        err = clGetPlatformIDs(STARPU_OPENCL_PLATFORM_MAX, platform_ids, &nb_platforms);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

        // Iterate over each platform
        for(platform=0; platform<nb_platforms; platform++) {
                // Get devices
                err = clGetDeviceIDs(platform_ids[platform], device_type, max, devices, &nb_devices);
                if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
                if(nb_devices > max)
                        nb_devices = max;

                // Iterate over each device
                unsigned int dev;
                for(dev = 0; dev < nb_devices; dev ++) {
                        cl_context       context;
                        cl_program       program;
                        cl_device_type   type;
                        cl_int           err;
                        cl_uint          uniqueid;

                        uniqueid =_starpu_opencl_device_uniqueid(devices[dev]);
                        // Look up and update history (to avoid unuseful compilations in the case of identical devices)
                        {
                                unsigned int d;
                                for(d = 0; d < dev; d++)
                                        if(history[d] == uniqueid)
                                                break; // Just skip compiling for this device
                                if(d != dev)
                                        continue;
                                history[dev] = uniqueid;
                        }

                        err = clGetDeviceInfo(devices[dev], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
                        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

                        // Create a compute context
                        context = clCreateContext(0, 1, devices + dev, NULL, NULL, &err);
                        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

                        // Run C preprocessor
                        {
                                pid_t pid;
                                pid = fork();
                                if(pid == 0) {
                                        execlp("cpp", "cpp", located_file_name, "-o", preproc_file_name, NULL);
                                        perror("execlp");
                                        exit(EXIT_FAILURE);
                                }
                                else {
                                        int status;
                                        waitpid(pid, &status, 0);
                                        if (WEXITSTATUS(status) != EXIT_SUCCESS)
                                                OPENCL_ERROR("Cannot preprocess file [%s]\n", located_file_name);
                                }
                        }

                        // Load the compute program from disk into a cstring buffer
                        char *source = _starpu_opencl_load_program_source(preproc_file_name);
                        if(!source)
                                OPENCL_ERROR("Failed to load compute program from file <%s>!\n", preproc_file_name);

                        // Delete preprocessed file
                        unlink(preproc_file_name);

                        // Create the compute program from the source buffer
                        program = clCreateProgramWithSource(context, 1, (const char **) & source, NULL, &err);
                        if (!program || err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

                        // Build the program executable
                        err = clBuildProgram(program, 1, devices + dev, "-Werror -cl-mad-enable", NULL, NULL);
                        if (err != CL_SUCCESS) {
                                size_t len;
                                static char buffer[4096];

                                fprintf(stderr, "Error: Failed to build program executable!\n");
                                clGetProgramBuildInfo(program, devices[dev], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

                                fprintf(stderr, "%s\n", buffer);
                                return EXIT_FAILURE;
                        }

                        // Store program binary
                        {
                                char     binary_filename[1024];
                                char    *binary;
                                size_t   binary_len;

                                // Get the name of the binary file
                                _starpu_opencl_get_binary_filename(located_file_name, devices[dev], binary_filename);

                                err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_len, NULL);
                                if(err != CL_SUCCESS)
                                        OPENCL_ERROR("Cannot get program binary size (err = %d)!\n", err);

                                binary = malloc(binary_len);

                                err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(binary), &binary, NULL);
                                if(err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

                                err = _starpu_opencl_store_program_binary(binary_filename, binary, binary_len);
                                if (err != EXIT_SUCCESS)
                                        OPENCL_ERROR("Cannot store program binary (err = %d)!\n", err);

                                free(binary);

                                _STARPU_OPENCL_DEBUG("Binary file [%s] successfully built (%ld bytes).\n", binary_filename, binary_len);
                        }

                        clReleaseProgram(program);
                        clReleaseContext(context);
                }
        }

        return EXIT_SUCCESS;
}
