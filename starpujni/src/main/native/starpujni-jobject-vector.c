/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "starpujni-jobject-vector.h"

static const struct starpu_data_copy_methods JOBJECT_VECTOR_COPY_METHODS =
{
	.can_copy = NULL,
	.ram_to_ram = NULL,
	.ram_to_cuda = NULL,
	.ram_to_opencl = NULL,
	.cuda_to_ram = NULL,
	.cuda_to_cuda = NULL,
	.opencl_to_ram = NULL,
	.opencl_to_opencl = NULL,
	.ram_to_cuda_async = NULL,
	.cuda_to_ram_async = NULL,
	.cuda_to_cuda_async = NULL,
	.ram_to_opencl_async = NULL,
	.opencl_to_ram_async = NULL,
	.opencl_to_opencl_async = NULL,
	.any_to_any = NULL
};

static void s_jobject_vector_register_data_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface);

static starpu_ssize_t s_jobject_vector_allocate_data_on_node(void *data_interface, unsigned node);

static void s_jobject_vector_free_data_on_node(void *data_interface, unsigned node);

#define s_jobject_vector_handle_to_pointer NULL

static void *s_jobject_vector_to_pointer(void *data_interface, unsigned node);

static size_t s_jobject_vector_get_size(starpu_data_handle_t handle);

static uint32_t s_jobject_vector_footprint(starpu_data_handle_t handle);

#define s_jobject_vector_alloc_footprint NULL

static int s_jobject_vector_compare(void *data_interface_a, void *data_interface_b);

#define s_jobject_vector_alloc_compare NULL

static void s_jobject_vector_display(starpu_data_handle_t handle, FILE *f);

static starpu_ssize_t s_jobject_vector_describe(void *data_interface, char *buf, size_t size);

#define s_jobject_vector_get_mf_ops NULL

static int s_jobject_vector_pack_data(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count);

static int s_jobject_vector_unpack_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);

static struct starpu_data_interface_ops JOBJECT_VECTOR_INTERFACE_OPS =
{
	.register_data_handle = s_jobject_vector_register_data_handle,
	.allocate_data_on_node = s_jobject_vector_allocate_data_on_node,
	.free_data_on_node = s_jobject_vector_free_data_on_node,
	.copy_methods = &JOBJECT_VECTOR_COPY_METHODS,
	.handle_to_pointer = s_jobject_vector_handle_to_pointer,
	.to_pointer = s_jobject_vector_to_pointer,
	.get_size = s_jobject_vector_get_size,
	.footprint = s_jobject_vector_footprint,
	.alloc_footprint = s_jobject_vector_alloc_footprint,
	.compare = s_jobject_vector_compare,
	.alloc_compare = s_jobject_vector_alloc_compare,
	.display = s_jobject_vector_display,
	.describe = s_jobject_vector_describe,
	.interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
	.interface_size = sizeof(struct jobject_vector_interface),
	.is_multiformat = 0,
	.dontcache = 0,
	.get_mf_ops = s_jobject_vector_get_mf_ops,
	.pack_data = s_jobject_vector_pack_data,
	.unpack_data = s_jobject_vector_unpack_data,
	.name = (char *) "JOBJECT_VECTOR_INTERFACE"
};

static unsigned s_memory_node(void)
{
	int workerid = starpu_worker_get_id();
	unsigned result;
	if (workerid == -1)
		result = STARPU_MAIN_RAM;
	else
		result = starpu_worker_get_memory_node(workerid);

	return result;
}

void starpujni_jobject_vector_data_register(starpu_data_handle_t *handleptr, int home_node, uint32_t size)
{
	struct jobject_vector_interface vector =
	{
		.id = STARPUJNI_JOBJECT_VECTOR_INTERFACE_ID,
		.nx = size,
		.ptr = 0
	};

	if (JOBJECT_VECTOR_INTERFACE_OPS.interfaceid == STARPU_UNKNOWN_INTERFACE_ID)
		JOBJECT_VECTOR_INTERFACE_OPS.interfaceid = STARPUJNI_JOBJECT_VECTOR_INTERFACE_ID;
	starpu_data_register(handleptr, home_node, &vector, &JOBJECT_VECTOR_INTERFACE_OPS);
}

uint32_t starpujni_jobject_vector_get_nx(starpu_data_handle_t handle)
{
	struct jobject_vector_interface *vector = starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
	return vector->nx;
}

uintptr_t starpujni_jobject_vector_get_local_ptr(starpu_data_handle_t handle)
{
	unsigned node = s_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));
	struct jobject_vector_interface *vector = starpu_data_get_interface_on_node(handle, node);

	return vector->ptr;
}

static void s_jobject_vector_register_data_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface)
{
	struct jobject_vector_interface *vector_interface = data_interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct jobject_vector_interface *local_interface = starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
		{
			local_interface->ptr = vector_interface->ptr;
		}
		else
		{
			local_interface->ptr = 0;
		}
		local_interface->id = vector_interface->id;
		local_interface->nx = vector_interface->nx;
	}
}

static starpu_ssize_t s_jobject_vector_allocate_data_on_node(void *data_interface, unsigned node)
{
	struct jobject_vector_interface *vector = data_interface;
	size_t allocsize = sizeof(jobject) * vector->nx;
	uintptr_t addr = starpu_malloc_on_node(node, allocsize);

	if (!addr)
		return -ENOMEM;

	vector->ptr = addr;

	return allocsize;
}

static void s_jobject_vector_free_data_on_node(void *data_interface, unsigned node)
{
	struct jobject_vector_interface *vector = data_interface;

	starpu_free_on_node(node, vector->ptr, vector->nx * sizeof(jobject));
	vector->ptr = 0;
}

static void *s_jobject_vector_to_pointer(void *data_interface, unsigned node)
{
	(void) node;
	struct jobject_vector_interface *vector = data_interface;

	return (void *) vector->ptr;
}

static size_t s_jobject_vector_get_size(starpu_data_handle_t handle)
{
	struct jobject_vector_interface *vector = starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
	size_t size = vector->nx * sizeof(jobject);

	return size;
}

static uint32_t s_jobject_vector_footprint(starpu_data_handle_t handle)
{
	return starpu_hash_crc32c_be(starpujni_jobject_vector_get_nx(handle), 0);
}

static int s_jobject_vector_compare(void *data_interface_a, void *data_interface_b)
{
	struct jobject_vector_interface *v1 = data_interface_a;
	struct jobject_vector_interface *v2 = data_interface_b;

	return v1->nx == v2->nx;
}

static void s_jobject_vector_display(starpu_data_handle_t handle, FILE *f)
{
	struct jobject_vector_interface *vector = starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
	fprintf(f, "%u\t", vector->nx);
}

static starpu_ssize_t s_jobject_vector_describe(void *data_interface, char *buf, size_t size)
{
	struct jobject_vector_interface *vector = data_interface;
	return snprintf(buf, size, "JV%u",(unsigned) vector->nx);
}

static int s_jobject_vector_pack_data(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	jbyte *buf;
	JNIEnv *env = NULL;
	jbyteArray output;

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));
	if ((*THE_VM)->AttachCurrentThread(THE_VM, (void **) &env, NULL) != JNI_OK)
	{
		fprintf(stderr, "Cannot attach current thread.\n");
		return -1;
	}
	STARPU_ASSERT(env != NULL);
	struct jobject_vector_interface *vector = starpu_data_get_interface_on_node(handle, node);
	output = (*env)->CallStaticObjectMethod(env, starpujni_vector_handle_class,
						 starpujni_vector_handle_pack_method,
						 (jlong) PTR_SET_MARK(vector));
	*count = (*env)->GetArrayLength(env, output) * sizeof(jbyte);
	if (ptr != NULL)
	{
		starpu_malloc_flags(ptr, *count, 0);
		if (*ptr != NULL)
		{
			buf = (*env)->GetByteArrayElements(env, output, NULL);
			memcpy(*ptr, buf, *count);
			(*env)->ReleaseByteArrayElements(env, output, buf, JNI_ABORT);
		}
		else
			{
				*count = 0;
			}
	}

	return 0;
}

static int s_jobject_vector_unpack_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	jbyteArray input;
	jbyte *buf;
	JNIEnv *env = NULL;
	struct jobject_vector_interface *vector;

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));
	if ((*THE_VM)->AttachCurrentThread(THE_VM, (void **) &env, NULL) != JNI_OK)
	{
		fprintf(stderr, "Cannot attach current thread.\n");
		return -1;
	}
	STARPU_ASSERT(env != NULL);
	input = (*env)->NewByteArray(env, count);
	if (input == NULL)
		return -1;

	buf = (*env)->GetByteArrayElements(env, input, NULL);
	memcpy(buf, ptr, count);
	(*env)->ReleaseByteArrayElements(env, input, buf, 0);

	vector = starpu_data_get_interface_on_node(handle, node);

	(*env)->CallStaticObjectMethod(env, starpujni_vector_handle_class,
					starpujni_vector_handle_unpack_method,
					(jlong) PTR_SET_MARK(vector), input);
	(*env)->DeleteLocalRef(env, input);

	return 0;
}

JNIEXPORT jlong JNICALL STARPU_DATA_FUNCNAME(VectorHandle, vectorObjectRegister)(JNIEnv *env, jclass cls, jint size)
{
	int i;
	starpu_data_handle_t hdl = NULL;
	starpujni_jobject_vector_data_register(&hdl, -1, size);
	jobject *array;

	if(hdl == NULL)
		starpujni_raise_exception(env, "not enough memory");
	starpu_data_acquire(hdl, STARPU_W);
	array = (jobject *) starpu_data_get_local_ptr(hdl);
	for(i = 0; i < size; i++)
		array[i] = NULL;
	starpu_data_release(hdl);

	return (jlong) hdl;
}

JNIEXPORT jint JNICALL STARPU_DATA_FUNCNAME(VectorHandle, vectorObjectGetSize)(JNIEnv *env, jclass cls, jlong handle)
{
	starpu_data_handle_t hdl =(starpu_data_handle_t) PTR_GET_ADDR(handle);

	if (PTR_HAS_MARK(handle))
		return (jint) STARPUJNI_JOBJECT_VECTOR_GET_NX(handle);
	return (jint) starpujni_jobject_vector_get_nx(hdl);
}

JNIEXPORT jobject JNICALL STARPU_DATA_FUNCNAME(VectorHandle, vectorObjectGetAt)(JNIEnv *env, jclass cls, jlong handle, jint idx)
{
	jobject result;

	if (PTR_HAS_MARK(handle))
	{
		STARPU_ASSERT(0 <= idx && idx < STARPUJNI_JOBJECT_VECTOR_GET_NX(handle));
		jobject *array = (jobject *)
			STARPUJNI_JOBJECT_VECTOR_GET_LOCAL_PTR(handle);
		if (array == NULL)
			return NULL;
		result = array[idx];
	}
	else
	{
		jobject *array;
		starpu_data_handle_t hdl = (starpu_data_handle_t) PTR_GET_ADDR(handle);
		if (starpu_data_test_if_allocated_on_node(hdl, s_memory_node()))
		{
			STARPU_ASSERT(0 <= idx && idx < starpujni_jobject_vector_get_nx(hdl));
			array = (jobject *) starpujni_jobject_vector_get_local_ptr(hdl);
			result = array[idx];
		}
		else
		{
			result = NULL;
		}
	}
	return result;
}

JNIEXPORT void JNICALL STARPU_DATA_FUNCNAME(VectorHandle, vectorObjectSetAt)(JNIEnv *env, jclass cls, jlong handle, jint idx, jobject value)
{
	jobject old = STARPU_DATA_FUNCNAME(VectorHandle, vectorObjectGetAt)(env, cls, handle, idx);
	if (old != NULL)
		(*env)->DeleteGlobalRef(env, old);
	value = (*env)->NewGlobalRef(env, value);

	if (PTR_HAS_MARK(handle))
	{
		STARPU_ASSERT(0 <= idx && idx < STARPUJNI_JOBJECT_VECTOR_GET_NX(handle));
		jobject *array = (jobject *) STARPUJNI_JOBJECT_VECTOR_GET_LOCAL_PTR(handle);
		array[idx] = value;
	}
	else
	{
		jobject *array;
		starpu_data_handle_t hdl = (starpu_data_handle_t) PTR_GET_ADDR(handle);

		STARPU_ASSERT(0 <= idx && idx < starpujni_jobject_vector_get_nx(hdl));
		array = (jobject *) starpujni_jobject_vector_get_local_ptr(hdl);
		array[idx] = value;
	}
}

/* copy/paste from *PU */
static void s_compute_chunk_size_and_offset(unsigned n, unsigned nparts, size_t elemsize, unsigned i, unsigned ld, unsigned *chunk_size, size_t *offset)
{
	*chunk_size = n / nparts;
	unsigned remainder = n % nparts;
	if (i < remainder)
		(*chunk_size)++;
	if (offset != NULL)
		*offset = (i * (n / nparts) + STARPU_MIN(remainder, i)) * ld * elemsize;
}

static void s_jobject_vector_filter_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	uint32_t child_nx;
	size_t offset;
	struct jobject_vector_interface *vector_father = father_interface;
	struct jobject_vector_interface *vector_child = child_interface;
	uint32_t nx = vector_father->nx;

	STARPU_ASSERT_MSG(nchunks <= nx, "%u parts for %u elements", nchunks, nx);

	s_compute_chunk_size_and_offset(nx, nchunks, sizeof(jobject), id, 1, &child_nx, &offset);

	STARPU_ASSERT_MSG(vector_father->id == STARPUJNI_JOBJECT_VECTOR_INTERFACE_ID,
			  "%s can only be applied on a jobject-vector data", __func__);
	vector_child->id = vector_father->id;
	vector_child->nx = child_nx;
	vector_child->ptr = vector_father->ptr + offset;
}

JNIEXPORT void JNICALL STARPU_DATA_FUNCNAME(VectorHandle, partition)(JNIEnv *env, jclass cls, jlong handle, jint nbParts)
{
	struct starpu_data_filter f =
	{
		.filter_func = s_jobject_vector_filter_block,
		.nchildren = nbParts
	};

	assert(nbParts > 0);
	assert(!PTR_HAS_MARK(handle));

	starpu_data_partition((starpu_data_handle_t) handle, &f);
}
