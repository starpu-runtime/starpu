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


#ifndef CL_HEADERS
#include "CL/cl.h"
#else
#include CL_HEADERS "CL/cl.h"
#endif


#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>

#include <starpu.h>
#include <starpu_opencl.h>
#include <starpu_data_interfaces.h>
#include <starpu_profiling.h>
#include <starpu_task.h>

typedef struct starpu_task starpu_task;

#ifdef UNUSED
#elif defined(__GNUC__)
   #define UNUSED(x) UNUSED_ ## x __attribute__((unused))
#else
   #define UNUSED(x) x
#endif

#define RETURN_EVENT(ev, event) \
   if (event != NULL) \
      *event = ev; \
   else\
      gc_entity_release(ev);

#include "helper_debug.c.inc"
#include "helper_getinfo.c.inc"

/**
 * Entity that can be managed by the garbage collector
 */
typedef struct entity * entity;

struct entity {
  /* Reference count */
  size_t refs;

  /* Callback called on release */
  void (*release_callback)(void*entity);

  /* Next entity in garbage collector queue */
  entity prev;
  entity next;
};

/* OpenCL entities (context, command queues, buffers...) must use
 * this macro as their first field */
#define CL_ENTITY struct entity _entity;


struct _cl_platform_id {};

static struct _cl_platform_id socl_platform = {};

static const char SOCL_PROFILE[] = "FULL_PROFILE";
static const char SOCL_VERSION[] = "OpenCL 1.0 StarPU Edition (0.0.1)";
static const char SOCL_PLATFORM_NAME[]    = "StarPU Platform";
static const char SOCL_VENDOR[]  = "INRIA";
static const char SOCL_PLATFORM_EXTENSIONS[] = "";

struct _cl_context {
  CL_ENTITY;

  void (*pfn_notify)(const char *, const void *, size_t, void *);
  void *user_data;

  /* Associated devices */
  cl_device_id * devices;
  cl_uint num_devices;

  /* Properties */
  cl_context_properties * properties;
  cl_uint num_properties;

  /* ID  */
#ifdef DEBUG
  int id;
#endif
};


struct _cl_command_queue {
  CL_ENTITY;

  cl_command_queue_properties properties;
  cl_device_id device;
  cl_context context;

  /* Stored command events */
  cl_event events;

  /* Last enqueued barrier-like event */
  cl_event barrier;

  /* Mutex */
  pthread_spinlock_t spin;

  /* ID  */
#ifdef DEBUG
  int id;
#endif
};

struct _cl_event {
  CL_ENTITY;

  /* Command queue */
  cl_command_queue cq;

  /* Command type */
  cl_command_type type;

  /* Command queue list */
  cl_event prev;
  cl_event next;

  /* Event status */
  cl_int status;

  /* ID  
   * This ID is used as a tag for StarPU dependencies
   */
  int id;

  /* Profiling info are copied here */
  struct starpu_task_profiling_info *profiling_info;
};

struct _cl_mem {
  CL_ENTITY;

  /* StarPU handle */
  starpu_data_handle handle;

  /* Pointer to data in host memory */
  void *ptr;    

  /* Buffer size */
  size_t size;

  /* Indicates how many references (mapping, MEM_USE_HOST_PTR...) require
   * coherence in host memory. If set to zero, no coherency is maintained
   * (this is the most efficient) */
  int map_count; 

  /* Creation flags */
  cl_mem_flags flags;

  /* Creation context */
  cl_context context;

  /* Access mode */
  int mode;

  /* Host ptr */
  void * host_ptr;

  /* Fields used to store cl_mems in mem_objects list */
  cl_mem prev;
  cl_mem next;

  /* Indicates if a buffer may contain meaningful data. Otherwise
     we don't have to transfer it */
  int scratch;

  /* ID  */
#ifdef DEBUG
  int id;
#endif
};

struct _cl_program {
  CL_ENTITY;

  /* Real OpenCL Programs
   * There is one entry for each device (even non OpenCL ones)
   * in order to index this array with dev_id
   */
  cl_program *cl_programs;

  /* Context used to create this program */
  cl_context context;

  /* Options  */
  char * options;
  unsigned int options_size;

  /* ID  */
#ifdef DEBUG
  int id;
#endif
};

enum kernel_arg_type { Null, Buffer, Immediate };

struct _cl_kernel {
  CL_ENTITY;

  /* Associated program */
  cl_program program;

  /* Kernel name */
  char * kernel_name;

  /* Real OpenCL kernels */
  cl_kernel *cl_kernels;

  /* clCreateKernel return codes */
  cl_int *errcodes;

  /* Arguments */
  unsigned int arg_count;
  size_t *arg_size;
  enum kernel_arg_type  *arg_type;
  void  **arg_value;

  /* ID  */
#ifdef DEBUG
  int id;
#endif
};

/* Command queues with profiling enabled
 * This allows us to disable StarPU profiling it
 * is equal to 0
 */
static int profiling_queue_count = 0;

#include "helper_workerid.c.inc"

#include "gc.c.inc"

#include "cl_getplatformids.c.inc"
#include "cl_getplatforminfo.c.inc"

#include "device_descriptions.c.inc"
#include "cl_getdeviceids.c.inc"
#include "cl_getdeviceinfo.c.inc"

#include "cl_releasecontext.c.inc"
#include "cl_createcontext.c.inc"
#include "cl_createcontextfromtype.c.inc"
#include "cl_retaincontext.c.inc"
#include "cl_getcontextinfo.c.inc"

#include "cl_releasecommandqueue.c.inc"
#include "cl_createcommandqueue.c.inc"
#include "cl_retaincommandqueue.c.inc"
#include "cl_getcommandqueueinfo.c.inc"
#include "cl_setcommandqueueproperty.c.inc"

#include "cl_releaseevent.c.inc"
#include "helper_event.c.inc"
#include "helper_task.c.inc"
#include "cl_waitforevents.c.inc"
#include "cl_geteventinfo.c.inc"
#include "cl_retainevent.c.inc"

#include "helper_command_queue.c.inc"

#include "cl_enqueuemarker.c.inc"
#include "cl_enqueuewaitforevents.c.inc"
#include "cl_enqueuebarrier.c.inc"
#include "cl_flush.c.inc"
#include "cl_finish.c.inc"

#include "helper_mem_objects.c.inc"
#include "cl_releasememobject.c.inc"
#include "cl_createbuffer.c.inc"
#include "cl_createimage2d.c.inc"
#include "cl_createimage3d.c.inc"
#include "cl_retainmemobject.c.inc"
#include "cl_getsupportedimageformats.c.inc"
#include "cl_getmemobjectinfo.c.inc"
#include "cl_getimageinfo.c.inc"

#include "cl_createsampler.c.inc"
#include "cl_retainsampler.c.inc"
#include "cl_releasesampler.c.inc"
#include "cl_getsamplerinfo.c.inc"

#include "cl_releaseprogram.c.inc"
#include "cl_createprogramwithsource.c.inc"
#include "cl_createprogramwithbinary.c.inc"
#include "cl_retainprogram.c.inc"
#include "cl_buildprogram.c.inc"
#include "cl_unloadcompiler.c.inc"
#include "cl_getprograminfo.c.inc"
#include "cl_getprogrambuildinfo.c.inc"

#include "cl_releasekernel.c.inc"
#include "cl_createkernel.c.inc"
#include "cl_createkernelsinprogram.c.inc"
#include "cl_retainkernel.c.inc"
#include "cl_setkernelarg.c.inc"
#include "cl_getkernelinfo.c.inc"
#include "cl_getkernelworkgroupinfo.c.inc"

#include "cl_enqueuereadbuffer.c.inc"
#include "cl_enqueuewritebuffer.c.inc"
#include "cl_enqueuecopybuffer.c.inc"
#include "cl_enqueuereadimage.c.inc"
#include "cl_enqueuewriteimage.c.inc"
#include "cl_enqueuecopyimage.c.inc"
#include "cl_enqueuecopyimagetobuffer.c.inc"
#include "cl_enqueuecopybuffertoimage.c.inc"
#include "cl_enqueuemapbuffer.c.inc"
#include "cl_enqueuemapimage.c.inc"
#include "cl_enqueueunmapmemobject.c.inc"
#include "cl_enqueuetask.c.inc"
#include "cl_enqueuendrangekernel.c.inc"
#include "cl_enqueuenativekernel.c.inc"

#include "cl_geteventprofilinginfo.c.inc"
#include "cl_getextensionfunctionaddress.c.inc"

#include "init.c.inc"
