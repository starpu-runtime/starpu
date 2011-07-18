/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010,2011 University of Bordeaux
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

// OpenCL 1.0 : Mandatory format: major_number.minor_number
static const char SOCL_DRIVER_VERSION[] = "0.1";

static const cl_uint SOCL_DEVICE_VENDOR_ID = 666;

struct _cl_device_id {
   cl_device_type    type;
   cl_uint           max_compute_units;        //OpenCL 1.0: minimum value is 1
   cl_uint           max_work_item_dimensions; //OpenCL 1.0: minimum value is 3
   size_t            max_work_item_sizes[3];   //array size should be set accordingly to the maximum max_work_item_dimensions
   size_t            max_work_group_size;      //OpenCL 1.0: minimum value is 1
   cl_uint           preferred_vector_widths[6]; //Char, Short, Int, Long, Float, Double
   cl_uint           max_clock_frequency;
   cl_uint           address_bits;             //OpenCL 1.0: 32 or 64
   cl_ulong          max_mem_alloc_size;       //OpenCL 1.0: minimum value is max(CL_DEVICE_GLOBAL_MEM_SIZE/4, 128*1024*1024)
   cl_bool           image_support;
   //image fields not present
   size_t            max_parameter_size;       //OpenCL 1.0: minimum is 256
   cl_uint           mem_base_addr_align;
   cl_uint           min_data_type_align_size;
   cl_device_fp_config single_fp_config;       //OpenCL 1.0: CL_FP_ROUND_TO_NEAREST and CL_FP_INF_NAN are mandatory
   cl_device_mem_cache_type global_mem_cache_type;
   cl_uint           global_mem_cacheline_size;
   cl_ulong          global_mem_cache_size;
   cl_ulong          global_mem_size;
   cl_ulong          max_constant_buffer_size; //OpenCL 1.0: minimum value is 64KB
   cl_uint           max_constant_args;        //OpenCL 1.0: minimum value is 8
   cl_device_local_mem_type local_mem_type;
   cl_ulong          local_mem_size;           //OpenCL 1.0: minimum value is 16KB
   cl_bool           error_correction_support;
   size_t            profiling_timer_resolution;
   cl_bool           endian_little;
   cl_bool           available;
   cl_bool           compiler_available;       //OpenCL 1.0: mandatory for FULL_PROFILE platforms
   cl_device_exec_capabilities execution_capabilities; //OpenCL 1.0: CL_EXEC_KERNEL is mandatory
   cl_command_queue_properties queue_properties; //OpenCL 1.0: CL_QUEUE_PROFILING_ENABLE is mandatory
   char              name[40];                 //Array size has been arbitrarily defined
   //versions, profile and vendor are statically defined for all devices
   char              extensions[100];          //Array size has been arbitrarily defined
};

static const struct _cl_device_id socl_devices[] = {
   { 
      .type = CL_DEVICE_TYPE_CPU,
      .max_compute_units = 1,
      .max_work_item_dimensions = 3,
      .max_work_item_sizes = {1,1,1},
      .max_work_group_size = 1,
      .preferred_vector_widths = {16,8,4,2,4,2},
      .max_clock_frequency = 3000,
      .address_bits = 64,
      .max_mem_alloc_size = 1024*1024*1024,
      .image_support = CL_FALSE,
      .max_parameter_size = 256,
      .mem_base_addr_align = 0,
      .min_data_type_align_size = 0,
      .single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN,
      .global_mem_cache_type = CL_READ_WRITE_CACHE,
      .global_mem_cacheline_size = 128,
      .global_mem_cache_size = 16*1024,
      .global_mem_size = (cl_ulong)4*1024*1024*1024,
      .max_constant_args = 8,
      .local_mem_type = CL_GLOBAL,
      .local_mem_size = 16*1024,
      .error_correction_support = CL_FALSE,
      .profiling_timer_resolution = 100,
      .endian_little = CL_TRUE,
      .available = CL_TRUE,
      .compiler_available = CL_TRUE,
      .execution_capabilities = CL_EXEC_KERNEL,
      .queue_properties = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
      .name = "StarPU virtual CPU 1",
      .extensions = ""
   },
   {
      .type = CL_DEVICE_TYPE_GPU,
      .max_compute_units = 12,
      .max_work_item_dimensions = 3,
      .max_work_item_sizes = {512,512,64},
      .max_work_group_size = 512,
      .preferred_vector_widths = {16,8,4,2,4,2},
      .max_clock_frequency = 1600,
      .address_bits = 32,
      .max_mem_alloc_size = 100*1024*1024,
      .image_support = CL_FALSE,
      .max_parameter_size = 256,
      .mem_base_addr_align = 0,
      .min_data_type_align_size = 0,
      .single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN,
      .global_mem_cache_type = CL_NONE,
      .global_mem_cacheline_size = 0,
      .global_mem_cache_size = 0,
      .global_mem_size = (cl_ulong)500*1024*1024,
      .max_constant_args = 8,
      .local_mem_type = CL_LOCAL,
      .local_mem_size = 16*1024,
      .error_correction_support = CL_FALSE,
      .profiling_timer_resolution = 10,
      .endian_little = CL_TRUE,
      .available = CL_TRUE,
      .compiler_available = CL_TRUE,
      .execution_capabilities = CL_EXEC_KERNEL,
      .queue_properties = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
      .name = "StarPU virtual GPU 1",
      .extensions = ""
   },
   {
      .type = CL_DEVICE_TYPE_GPU,
      .max_compute_units = 12,
      .max_work_item_dimensions = 3,
      .max_work_item_sizes = {512,512,64},
      .max_work_group_size = 512,
      .preferred_vector_widths = {16,8,4,2,4,2},
      .max_clock_frequency = 1600,
      .address_bits = 32,
      .max_mem_alloc_size = 100*1024*1024,
      .image_support = CL_FALSE,
      .max_parameter_size = 256,
      .mem_base_addr_align = 0,
      .min_data_type_align_size = 0,
      .single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN,
      .global_mem_cache_type = CL_NONE,
      .global_mem_cacheline_size = 0,
      .global_mem_cache_size = 0,
      .global_mem_size = (cl_ulong)500*1024*1024,
      .max_constant_args = 8,
      .local_mem_type = CL_LOCAL,
      .local_mem_size = 16*1024,
      .error_correction_support = CL_FALSE,
      .profiling_timer_resolution = 10,
      .endian_little = CL_TRUE,
      .available = CL_TRUE,
      .compiler_available = CL_TRUE,
      .execution_capabilities = CL_EXEC_KERNEL,
      .queue_properties = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
      .name = "StarPU virtual GPU 2",
      .extensions = ""
   },
   {
      .type = CL_DEVICE_TYPE_GPU,
      .max_compute_units = 12,
      .max_work_item_dimensions = 3,
      .max_work_item_sizes = {512,512,64},
      .max_work_group_size = 512,
      .preferred_vector_widths = {16,8,4,2,4,2},
      .max_clock_frequency = 1600,
      .address_bits = 32,
      .max_mem_alloc_size = 100*1024*1024,
      .image_support = CL_FALSE,
      .max_parameter_size = 256,
      .mem_base_addr_align = 0,
      .min_data_type_align_size = 0,
      .single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN,
      .global_mem_cache_type = CL_NONE,
      .global_mem_cacheline_size = 0,
      .global_mem_cache_size = 0,
      .global_mem_size = (cl_ulong)500*1024*1024,
      .max_constant_args = 8,
      .local_mem_type = CL_LOCAL,
      .local_mem_size = 16*1024,
      .error_correction_support = CL_FALSE,
      .profiling_timer_resolution = 10,
      .endian_little = CL_TRUE,
      .available = CL_TRUE,
      .compiler_available = CL_TRUE,
      .execution_capabilities = CL_EXEC_KERNEL,
      .queue_properties = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
      .name = "StarPU virtual GPU 3",
      .extensions = ""
   },
   {
      .type = CL_DEVICE_TYPE_GPU,
      .max_compute_units = 12,
      .max_work_item_dimensions = 3,
      .max_work_item_sizes = {512,512,64},
      .max_work_group_size = 512,
      .preferred_vector_widths = {16,8,4,2,4,2},
      .max_clock_frequency = 1600,
      .address_bits = 32,
      .max_mem_alloc_size = 100*1024*1024,
      .image_support = CL_FALSE,
      .max_parameter_size = 256,
      .mem_base_addr_align = 0,
      .min_data_type_align_size = 0,
      .single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN,
      .global_mem_cache_type = CL_NONE,
      .global_mem_cacheline_size = 0,
      .global_mem_cache_size = 0,
      .global_mem_size = (cl_ulong)500*1024*1024,
      .max_constant_args = 8,
      .local_mem_type = CL_LOCAL,
      .local_mem_size = 16*1024,
      .error_correction_support = CL_FALSE,
      .profiling_timer_resolution = 10,
      .endian_little = CL_TRUE,
      .available = CL_TRUE,
      .compiler_available = CL_TRUE,
      .execution_capabilities = CL_EXEC_KERNEL,
      .queue_properties = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
      .name = "StarPU virtual GPU 4",
      .extensions = ""
   },
   {
      .type = CL_DEVICE_TYPE_GPU,
      .max_compute_units = 12,
      .max_work_item_dimensions = 3,
      .max_work_item_sizes = {512,512,64},
      .max_work_group_size = 512,
      .preferred_vector_widths = {16,8,4,2,4,2},
      .max_clock_frequency = 1600,
      .address_bits = 32,
      .max_mem_alloc_size = 100*1024*1024,
      .image_support = CL_FALSE,
      .max_parameter_size = 256,
      .mem_base_addr_align = 0,
      .min_data_type_align_size = 0,
      .single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN,
      .global_mem_cache_type = CL_NONE,
      .global_mem_cacheline_size = 0,
      .global_mem_cache_size = 0,
      .global_mem_size = (cl_ulong)500*1024*1024,
      .max_constant_args = 8,
      .local_mem_type = CL_LOCAL,
      .local_mem_size = 16*1024,
      .error_correction_support = CL_FALSE,
      .profiling_timer_resolution = 10,
      .endian_little = CL_TRUE,
      .available = CL_TRUE,
      .compiler_available = CL_TRUE,
      .execution_capabilities = CL_EXEC_KERNEL,
      .queue_properties = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
      .name = "StarPU virtual GPU 5",
      .extensions = ""
   },
   {
      .type = CL_DEVICE_TYPE_GPU,
      .max_compute_units = 12,
      .max_work_item_dimensions = 3,
      .max_work_item_sizes = {512,512,64},
      .max_work_group_size = 512,
      .preferred_vector_widths = {16,8,4,2,4,2},
      .max_clock_frequency = 1600,
      .address_bits = 32,
      .max_mem_alloc_size = 100*1024*1024,
      .image_support = CL_FALSE,
      .max_parameter_size = 256,
      .mem_base_addr_align = 0,
      .min_data_type_align_size = 0,
      .single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN,
      .global_mem_cache_type = CL_NONE,
      .global_mem_cacheline_size = 0,
      .global_mem_cache_size = 0,
      .global_mem_size = (cl_ulong)500*1024*1024,
      .max_constant_args = 8,
      .local_mem_type = CL_LOCAL,
      .local_mem_size = 16*1024,
      .error_correction_support = CL_FALSE,
      .profiling_timer_resolution = 10,
      .endian_little = CL_TRUE,
      .available = CL_TRUE,
      .compiler_available = CL_TRUE,
      .execution_capabilities = CL_EXEC_KERNEL,
      .queue_properties = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
      .name = "StarPU virtual GPU 6",
      .extensions = ""
   },
   {
      .type = CL_DEVICE_TYPE_GPU,
      .max_compute_units = 12,
      .max_work_item_dimensions = 3,
      .max_work_item_sizes = {512,512,64},
      .max_work_group_size = 512,
      .preferred_vector_widths = {16,8,4,2,4,2},
      .max_clock_frequency = 1600,
      .address_bits = 32,
      .max_mem_alloc_size = 100*1024*1024,
      .image_support = CL_FALSE,
      .max_parameter_size = 256,
      .mem_base_addr_align = 0,
      .min_data_type_align_size = 0,
      .single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN,
      .global_mem_cache_type = CL_NONE,
      .global_mem_cacheline_size = 0,
      .global_mem_cache_size = 0,
      .global_mem_size = (cl_ulong)500*1024*1024,
      .max_constant_args = 8,
      .local_mem_type = CL_LOCAL,
      .local_mem_size = 16*1024,
      .error_correction_support = CL_FALSE,
      .profiling_timer_resolution = 10,
      .endian_little = CL_TRUE,
      .available = CL_TRUE,
      .compiler_available = CL_TRUE,
      .execution_capabilities = CL_EXEC_KERNEL,
      .queue_properties = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
      .name = "StarPU virtual GPU 7",
      .extensions = ""
   }

};

static const int socl_device_count = sizeof(socl_devices) / sizeof(struct _cl_device_id);



