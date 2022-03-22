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

#ifndef __STARPUJNI_CODELET__
#define __STARPUJNI_CODELET__

#include "starpujni-common.h"

EXTERN int starpujni_codelet_init(JNIEnv *env);

EXTERN int starpujni_codelet_terminate(JNIEnv *env);

EXTERN struct starpujni_codelet *starpujni_codelet_create(JNIEnv *env, jobject codelet);

EXTERN struct starpujni_codelet *starpujni_codelet_create_for_1_buffer(JNIEnv *env, jobject codelet, jclass hdl);

EXTERN struct starpujni_codelet *starpujni_codelet_create_for_2_buffers(JNIEnv *env, jobject codelet,  jclass hdl1, jclass hdl2);

EXTERN size_t starpujni_codelet_get_size(struct starpujni_codelet *jnicl);

EXTERN int starpujni_codelet_get_nb_buffers(struct starpujni_codelet *jnicl);

EXTERN struct starpu_codelet *starpujni_codelet_get_codelet(struct starpujni_codelet *jnicl);

EXTERN void starpujni_codelet_set_ith_buffer_interface(JNIEnv *env, struct starpujni_codelet *cl, int index, jclass handle_class);

EXTERN void starpujni_codelet_destroy(JNIEnv *env, struct starpujni_codelet *cl);

#endif /* __STARPUJNI_CODELET__ */
