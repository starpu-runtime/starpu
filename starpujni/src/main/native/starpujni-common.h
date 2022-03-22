/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPUJNI_COMMON_H__
#define __STARPUJNI_COMMON_H__

#include <jni.h>
#include <starpu.h>

#ifdef __cplusplus
#  define EXTERN extern "C"
#else
#  define EXTERN extern
#endif /* __cplusplus */

#define STATIC_MEMBER 1
#define INSTANCE_MEMBER 0

struct jmethod_spec
{
	jmethodID *p_jid;
	int is_static;
	const char *name;
	const char *signature;
};

struct jfield_spec
{
	jfieldID *p_fid;
	int is_static;
	const char *name;
	const char *signature;
};

struct jclass_spec
{
	jclass *p_class;
	const char *name;
	const struct jmethod_spec *methods;
	const struct jfield_spec *fields;
};

#define STARPU_JNIVERSION JNI_VERSION_1_8

#define STARPUJNI_CLASSNAME(clname) "fr/labri/hpccloud/starpu/" #clname

#define STARPUJNI_FUNCNAME(_cls_, _method_) Java_fr_labri_hpccloud_starpu_ ## _cls_ ## _ ## _method_

# define PTR_GET_ADDR(_hdl_) ((void *)((_hdl_)&(~((jlong)1))))
# define PTR_HAS_MARK(_hdl_) ((((jlong)(_hdl_))&(jlong)0x1)!=0)
# define PTR_SET_MARK(_hdl_) (((jlong)(_hdl_))|((jlong)0x1))

EXTERN JavaVM *THE_VM;
EXTERN int starpujni_trace_enabled;

EXTERN int starpujni_common_init(JNIEnv *env);

EXTERN int starpujni_common_terminate(JNIEnv *env);

EXTERN jclass starpujni_find_class(JNIEnv *env, const char *classname);

/**
 * Cache class components;
 *
 * @param env the JNI context
 * @param cls the class from which the method IDs are looked for.
 * @param spec a NULL terminated array of methods specification
 * @return 1 on success
 */
EXTERN int starpujni_cache_class(JNIEnv *env, const struct jclass_spec *spec);

EXTERN void starpujni_clear_global_ref(JNIEnv *env, void **pptr);

EXTERN void starpujni_raise_exception(JNIEnv *env, const char *message);

EXTERN int starpujni_parse_access_mode(const char *strmode, enum starpu_data_access_mode *mode);

#define ERROR_MSG_GOTO(errorlabel_, fmt_, ...) \
do { \
  ERROR_MSG(fmt_, ## __VA_ARGS__); \
  goto errorlabel_; \
} while(0)

#define ERROR_MSG(fmt_, ...) \
do { \
  fprintf(stderr, "[starpujni][%s:%d] " fmt_ "\n",\
      __FUNCTION__,  __LINE__, ## __VA_ARGS__); \
} while(0)

#define STARPUJNI_TRACE(fmt_,...) \
  do { \
    if(starpujni_trace_enabled) \
      ERROR_MSG(fmt_, ## __VA_ARGS__); \
  } while(0)

#endif /* __STARPUJNI_COMMON_H__ */
