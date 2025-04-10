# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#

include $(top_srcdir)/make/starpu-notests.mk

CLEANFILES = *.gcno *.gcda

AM_CPPFLAGS = -DBUILDING_SOCL -I$(top_srcdir)/include/ -I$(top_builddir)/include -I$(top_builddir)/src -I$(top_srcdir)/src -I$(top_srcdir)/socl/src $(STARPU_H_CPPFLAGS)
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@ $(STARPU_EXPORTED_LIBS)
LIBS += $(STARPU_OPENCL_LDFLAGS)

SUBDIRS =

lib_LTLIBRARIES = libsocl-@STARPU_EFFECTIVE_VERSION@.la

noinst_HEADERS =				\
  command.h					\
  command_list.h				\
  command_queue.h				\
  debug.h					\
  event.h					\
  gc.h						\
  getinfo.h					\
  mem_objects.h					\
  ocl_icd.h					\
  socl.h					\
  task.h					\
  util.h					\
  init.h					\
  CL/cl_d3d10.h					\
  CL/cl_ext.h					\
  CL/cl.h					\
  CL/cl_d3d11.h					\
  CL/cl_gl_ext.h				\
  CL/cl_platform.h				\
  CL/cl_dx9_media_sharing.h			\
  CL/cl_gl.h					\
  CL/opencl.h

libsocl_@STARPU_EFFECTIVE_VERSION@_la_LDFLAGS = $(ldflags) -no-undefined			\
  -version-info $(LIBSOCL_INTERFACE_CURRENT):$(LIBSOCL_INTERFACE_REVISION):$(LIBSOCL_INTERFACE_AGE)

libsocl_@STARPU_EFFECTIVE_VERSION@_la_SOURCES = 						\
  command.c \
  command_list.c \
  command_queue.c \
  debug.c \
  event.c \
  gc.c \
  init.c \
  mem_objects.c \
  socl.c \
  task.c \
  util.c \
  cl_getplatformids.c \
  cl_getplatforminfo.c \
  cl_getdeviceids.c \
  cl_getdeviceinfo.c \
  cl_releasecontext.c \
  cl_createcontext.c \
  cl_createcontextfromtype.c \
  cl_retaincontext.c \
  cl_getcontextinfo.c \
  cl_releasecommandqueue.c \
  cl_createcommandqueue.c \
  cl_retaincommandqueue.c \
  cl_getcommandqueueinfo.c \
  cl_setcommandqueueproperty.c \
  cl_releaseevent.c \
  cl_waitforevents.c \
  cl_geteventinfo.c \
  cl_retainevent.c \
  cl_enqueuemarker.c \
  cl_enqueuewaitforevents.c \
  cl_enqueuebarrier.c \
  cl_flush.c \
  cl_finish.c \
  cl_releasememobject.c \
  cl_createbuffer.c \
  cl_createimage2d.c \
  cl_createimage3d.c \
  cl_retainmemobject.c \
  cl_getsupportedimageformats.c \
  cl_getmemobjectinfo.c \
  cl_getimageinfo.c \
  cl_createsampler.c \
  cl_retainsampler.c \
  cl_releasesampler.c \
  cl_getsamplerinfo.c \
  cl_releaseprogram.c \
  cl_createprogramwithsource.c \
  cl_createprogramwithbinary.c \
  cl_retainprogram.c \
  cl_buildprogram.c \
  cl_unloadcompiler.c \
  cl_getprograminfo.c \
  cl_getprogrambuildinfo.c \
  cl_releasekernel.c \
  cl_createkernel.c \
  cl_createkernelsinprogram.c \
  cl_retainkernel.c \
  cl_setkernelarg.c \
  cl_getkernelinfo.c \
  cl_getkernelworkgroupinfo.c \
  cl_enqueuereadbuffer.c \
  cl_enqueuewritebuffer.c \
  cl_enqueuecopybuffer.c \
  cl_enqueuereadimage.c \
  cl_enqueuewriteimage.c \
  cl_enqueuecopyimage.c \
  cl_enqueuecopyimagetobuffer.c \
  cl_enqueuecopybuffertoimage.c \
  cl_enqueuemapbuffer.c \
  cl_enqueuemapimage.c \
  cl_enqueueunmapmemobject.c \
  cl_enqueuetask.c \
  cl_enqueuendrangekernel.c \
  cl_enqueuenativekernel.c \
  cl_enqueuemarkerwithwaitlist.c \
  cl_enqueuebarrierwithwaitlist.c \
  cl_geteventprofilinginfo.c \
  cl_getextensionfunctionaddress.c \
  cl_icdgetplatformidskhr.c
