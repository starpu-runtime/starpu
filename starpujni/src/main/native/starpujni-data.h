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

#ifndef __STARPUJNI_DATA_H__
#define __STARPUJNI_DATA_H__

#include "starpujni-common.h"

#define STARPU_DATA_FUNCNAME(_cls_, _method_) STARPUJNI_FUNCNAME(data_ ## _cls_, _method_)

#define STARPUJNI_DATA_HANDLE_CLASSNAME STARPUJNI_CLASSNAME(data/DataHandle)
EXTERN jclass starpujni_data_handle_class;
EXTERN jfieldID starpujni_data_handle_id;

#define STARPUJNI_VECTOR_HANDLE_CLASSNAME STARPUJNI_CLASSNAME(data/VectorHandle)
EXTERN jclass starpujni_vector_handle_class;
EXTERN jmethodID starpujni_vector_handle_pack_method;
EXTERN jmethodID starpujni_vector_handle_unpack_method;
EXTERN jmethodID starpujni_vector_handle_packer_method;
EXTERN jmethodID starpujni_vector_handle_unpacker_method;

EXTERN int STARPUJNI_JOBJECT_VECTOR_INTERFACE_ID;

#define STARPUJNI_ACCESS_MODE_CLASSNAME STARPUJNI_DATA_HANDLE_CLASSNAME "$AccessMode"

EXTERN jboolean starpujni_data_init(JNIEnv *env);

EXTERN void starpujni_data_terminate(JNIEnv *env);

EXTERN int starpujni_get_access_mode(JNIEnv *env, jobject modeObj, enum starpu_data_access_mode *mode);

#endif /* __STARPUJNI_DATA_H__ */
