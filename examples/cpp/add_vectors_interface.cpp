/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This is a small example of a C++ program using STL and starpu.  We here just
 * add two std::vector with duplicating vectors. StarPU achieves data
 * transfers between objects.
 */

#if defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNU_MINOR < 9))
int main(int argc, char **argv)
{
	return 77;
}
#else
#include <cassert>
#include <vector>

#ifdef PRINT_OUTPUT
#include <iostream>
#endif

#include <starpu.h>

#define MY_TYPE char, my_allocator<char>

/* create an allocator to put data on the correct NUMA node */
template <class T>
class my_allocator
{
	public:

	typedef size_t    size_type;
	typedef ptrdiff_t difference_type;
	typedef T*        pointer;
	typedef const T*  const_pointer;
	typedef T&        reference;
	typedef const T&  const_reference;
	typedef T         value_type;

	my_allocator()
	{
		this->node = STARPU_MAIN_RAM;
	}

	my_allocator(const my_allocator& a)
	{
		node = a.get_node();
	}

	explicit my_allocator(const unsigned node)
	{
		this->node = node;
	}

	pointer allocate(size_type n, const void * = 0)
	{
		T* t = (T*) starpu_malloc_on_node(this->node, n * sizeof(T));
		return t;
	}

	void      deallocate(void* p, size_type n)
	{
		if (p)
		{
			starpu_free_on_node(this->node, (uintptr_t) p, n * sizeof(T));
		}
	}

	unsigned get_node() const
	{
		return node;
	}

	pointer address(reference x) const
	{
		return &x;
	}

	const_pointer address(const_reference x) const
	{
		return &x;
	}

	my_allocator<T>&  operator=(const my_allocator&ref)
	{
		node = ref.node;
		return *this;
	}

	void construct(pointer p, const T& val)
	{
		new ((T*) p) T(val);
	}

	void destroy(pointer p)
	{
		p->~T();
	}

	size_type max_size() const
	{
		return size_type(-1);
	}


	template <class U>
		struct rebind
		{
			typedef my_allocator<U> other;
		};

	template <class U>
		explicit my_allocator(const my_allocator<U>&ref)
		{
			node = ref.node;
		}

	template <class U>
		my_allocator<U>& operator=(const my_allocator<U>&ref)
		{
			node = ref.node;
			return *this;
		}

	private:
	unsigned node;
};

/*
 * Create a new interface to catch C++ vector and make appropriate data transfers
 */
struct vector_cpp_interface
{
	enum starpu_data_interface_id id;

	uintptr_t ptr;
	uintptr_t dev_handle;
	size_t offset;
	uint32_t nx;
	size_t elemsize;
	std::vector<MY_TYPE>* vec;

	uint32_t slice_base;
};

#define VECTOR_CPP_GET_VEC(interface)	({ (((struct vector_cpp_interface *)(interface))->vec); })

static int vector_interface_copy_any_to_any(void *src_interface, unsigned src_node,
                           void *dst_interface, unsigned dst_node, void *async_data);

#if __cplusplus >= 201103L
static const struct starpu_data_copy_methods vector_cpp_copy_data_methods_s =
{

	.can_copy = NULL,

	.ram_to_ram = NULL,
	.ram_to_cuda = NULL,
	.ram_to_opencl = NULL,
	.ram_to_mic = NULL,

	.cuda_to_ram = NULL,
	.cuda_to_cuda = NULL,
	.cuda_to_opencl = NULL,

	.opencl_to_ram = NULL,
	.opencl_to_cuda = NULL,
	.opencl_to_opencl = NULL,

	.mic_to_ram = NULL,

	.ram_to_mpi_ms = NULL,
	.mpi_ms_to_ram = NULL,
	.mpi_ms_to_mpi_ms = NULL,

	.ram_to_cuda_async = NULL,
	.cuda_to_ram_async = NULL,
	.cuda_to_cuda_async = NULL,

	.ram_to_opencl_async = NULL,
	.opencl_to_ram_async = NULL,
	.opencl_to_opencl_async = NULL,

	.ram_to_mpi_ms_async = NULL,
	.mpi_ms_to_ram_async = NULL,
	.mpi_ms_to_mpi_ms_async = NULL,

	.ram_to_mic_async = NULL,
	.mic_to_ram_async = NULL,

	.any_to_any = vector_interface_copy_any_to_any,
};
#else
static const struct starpu_data_copy_methods vector_cpp_copy_data_methods_s =
{
	NULL,

	NULL,
	NULL,
	NULL,
	NULL,

	NULL,
	NULL,
	NULL,

	NULL,
	NULL,
	NULL,

	NULL,

	NULL,
	NULL,
	NULL,

	NULL,
	NULL,
	NULL,

	NULL,
	NULL,
	NULL,

	NULL,
	NULL,
	NULL,

	NULL,
	NULL,
	NULL,

	NULL,
	NULL,

	vector_interface_copy_any_to_any,
};
#endif

static void register_vector_cpp_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface);
static starpu_ssize_t allocate_vector_cpp_buffer_on_node(void *data_interface_, unsigned dst_node);
static void *vector_cpp_to_pointer(void *data_interface, unsigned node);
static int vector_cpp_pointer_is_inside(void *data_interface, unsigned node, void *ptr);
static void free_vector_cpp_buffer_on_node(void *data_interface, unsigned node);
static void free_vector_cpp_buffer_on_node(void *data_interface, unsigned node);
static size_t vector_cpp_interface_get_size(starpu_data_handle_t handle);
static uint32_t footprint_vector_cpp_interface_crc32(starpu_data_handle_t handle);
static int vector_cpp_compare(void *data_interface_a, void *data_interface_b);
static void display_vector_cpp_interface(starpu_data_handle_t handle, FILE *f);
static int pack_vector_cpp_handle(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count);
static int unpack_vector_cpp_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);
static starpu_ssize_t vector_cpp_describe(void *data_interface, char *buf, size_t size);

#if __cplusplus >= 201103L
static struct starpu_data_interface_ops interface_vector_cpp_ops =
{
	.register_data_handle = register_vector_cpp_handle,
	.allocate_data_on_node = allocate_vector_cpp_buffer_on_node,
	.free_data_on_node = free_vector_cpp_buffer_on_node,
	.init = NULL,
	.copy_methods = &vector_cpp_copy_data_methods_s,
	.handle_to_pointer = NULL,
	.to_pointer = vector_cpp_to_pointer,
	.pointer_is_inside = vector_cpp_pointer_is_inside,
	.get_size = vector_cpp_interface_get_size,
	.get_alloc_size = NULL,
	.footprint = footprint_vector_cpp_interface_crc32,
	.alloc_footprint = NULL,
	.compare = vector_cpp_compare,
	.alloc_compare = NULL,
	.display = display_vector_cpp_interface,
	.describe = vector_cpp_describe,
	.interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
	.interface_size = sizeof(struct vector_cpp_interface),
	.is_multiformat = 0,
	.dontcache = 0,
	.get_mf_ops = NULL,
	.pack_data = pack_vector_cpp_handle,
	.unpack_data = unpack_vector_cpp_handle,
	.name = (char *) "VECTOR_CPP_INTERFACE"
};
#else
static struct starpu_data_interface_ops interface_vector_cpp_ops =
{
	register_vector_cpp_handle,
	allocate_vector_cpp_buffer_on_node,
	free_vector_cpp_buffer_on_node,
	NULL,
	&vector_cpp_copy_data_methods_s,
	vector_cpp_to_pointer,
	vector_cpp_pointer_is_inside,
	vector_cpp_interface_get_size,
	NULL,
	footprint_vector_cpp_interface_crc32,
	NULL,
	vector_cpp_compare,
	NULL,
	display_vector_cpp_interface,
	vector_cpp_describe,
	STARPU_UNKNOWN_INTERFACE_ID,
	sizeof(struct vector_cpp_interface),
	0,
	0,
	NULL,
	pack_vector_cpp_handle,
	unpack_vector_cpp_handle,
	(char *) "VECTOR_CPP_INTERFACE"
};
#endif

static void *vector_cpp_to_pointer(void *data_interface, unsigned node)
{
	(void) node;
	struct vector_cpp_interface *vector_interface = (struct vector_cpp_interface *) data_interface;

	return (void*) vector_interface->ptr;
}

static int vector_cpp_pointer_is_inside(void *data_interface, unsigned int node, void *ptr)
{
	(void) node;
	struct vector_cpp_interface *vector_interface = (struct vector_cpp_interface *) data_interface;

	return (char*) ptr >= (char*) vector_interface->ptr &&
		(char*) ptr < (char*) vector_interface->ptr + vector_interface->nx*vector_interface->elemsize;
}

static void register_vector_cpp_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface)
{
	struct vector_cpp_interface *vector_interface = (struct vector_cpp_interface *) data_interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct vector_cpp_interface *local_interface = (struct vector_cpp_interface *)
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
		{
			local_interface->ptr = vector_interface->ptr;
                        local_interface->dev_handle = vector_interface->dev_handle;
                        local_interface->offset = vector_interface->offset;
			local_interface->vec = vector_interface->vec;
		}
		else
		{
			local_interface->ptr = 0;
                        local_interface->dev_handle = 0;
                        local_interface->offset = 0;
			local_interface->vec = NULL;
		}

		local_interface->id = vector_interface->id;
		local_interface->nx = vector_interface->nx;
		local_interface->elemsize = vector_interface->elemsize;
		local_interface->slice_base = vector_interface->slice_base;
	}
}

/* declare a new data with the vector interface */
void vector_cpp_data_register(starpu_data_handle_t *handleptr, int home_node,
                        std::vector<MY_TYPE>* vec, uint32_t nx, size_t elemsize)
{
#if __cplusplus >= 201103L
	struct vector_cpp_interface vector =
	{
		.id = STARPU_UNKNOWN_INTERFACE_ID,
		.ptr = (uintptr_t) &(*vec)[0],
                .dev_handle = (uintptr_t) &(*vec)[0],
                .offset = 0,
		.nx = nx,
		.elemsize = elemsize,
		.vec = vec,
		.slice_base = 0
	};
#else
	struct vector_cpp_interface vector =
	{
		STARPU_UNKNOWN_INTERFACE_ID,
		(uintptr_t) &(*vec)[0],
                (uintptr_t) &(*vec)[0],
                0,
		nx,
		elemsize,
		vec,
		0
	};
#endif

	starpu_data_register(handleptr, home_node, &vector, &interface_vector_cpp_ops);
}

/* offer an access to the data parameters */
uint32_t vector_cpp_get_nx(starpu_data_handle_t handle)
{
	struct vector_cpp_interface *vector_interface = (struct vector_cpp_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return vector_interface->nx;
}


static uint32_t footprint_vector_cpp_interface_crc32(starpu_data_handle_t handle)
{
	return starpu_hash_crc32c_be(vector_cpp_get_nx(handle), 0);
}

static int vector_cpp_compare(void *data_interface_a, void *data_interface_b)
{
	struct vector_cpp_interface *vector_a = (struct vector_cpp_interface *) data_interface_a;
	struct vector_cpp_interface *vector_b = (struct vector_cpp_interface *) data_interface_b;

	/* Two vectors are considered compatible if they have the same size */
	return ((vector_a->nx == vector_b->nx)
			&& (vector_a->elemsize == vector_b->elemsize));
}

static void display_vector_cpp_interface(starpu_data_handle_t handle, FILE *f)
{
	struct vector_cpp_interface *vector_interface = (struct vector_cpp_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	fprintf(f, "%u\t", vector_interface->nx);
}

static int pack_vector_cpp_handle(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct vector_cpp_interface *vector_interface = (struct vector_cpp_interface *)
		starpu_data_get_interface_on_node(handle, node);

	*count = vector_interface->nx*vector_interface->elemsize;

	if (ptr != NULL)
	{
		*ptr = (void*) starpu_malloc_on_node_flags(node, *count, 0);
		memcpy(*ptr, (void*)vector_interface->ptr, vector_interface->elemsize*vector_interface->nx);
	}

	return 0;
}

static int unpack_vector_cpp_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct vector_cpp_interface *vector_interface = (struct vector_cpp_interface *)
		starpu_data_get_interface_on_node(handle, node);

	STARPU_ASSERT(count == vector_interface->elemsize * vector_interface->nx);
	memcpy((void*)vector_interface->ptr, ptr, count);

	starpu_free_on_node_flags(node, (uintptr_t)ptr, count, 0);

	return 0;
}

static size_t vector_cpp_interface_get_size(starpu_data_handle_t handle)
{
	size_t size;
	struct vector_cpp_interface *vector_interface = (struct vector_cpp_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	size = vector_interface->nx*vector_interface->elemsize;

	return size;
}

size_t vector_cpp_get_elemsize(starpu_data_handle_t handle)
{
	struct vector_cpp_interface *vector_interface = (struct vector_cpp_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return vector_interface->elemsize;
}

/* memory allocation/deallocation primitives for the vector interface */

/* returns the size of the allocated area */
static starpu_ssize_t allocate_vector_cpp_buffer_on_node(void *data_interface_, unsigned dst_node)
{
	struct vector_cpp_interface *vector_interface = (struct vector_cpp_interface *) data_interface_;

	uint32_t nx = vector_interface->nx;
	size_t elemsize = vector_interface->elemsize;

	starpu_ssize_t allocated_memory;

	const my_allocator<char> allocator(dst_node);
	std::vector<MY_TYPE> * vec = new std::vector<MY_TYPE>(nx, 0, allocator);

	vector_interface->vec = vec;
	if (!vector_interface->vec)
		return -ENOMEM;

	allocated_memory = nx*elemsize;

	/* update the data properly in consequence */
	vector_interface->ptr = (uintptr_t) &((*vec)[0]);
	vector_interface->dev_handle = (uintptr_t) &((*vec)[0]);
        vector_interface->offset = 0;

	return allocated_memory;
}

static void free_vector_cpp_buffer_on_node(void *data_interface, unsigned node)
{
	struct vector_cpp_interface *vector_interface = (struct vector_cpp_interface *) data_interface;

	delete vector_interface->vec;
}

static int vector_interface_copy_any_to_any(void *src_interface, unsigned src_node,
                           void *dst_interface, unsigned dst_node, void *async_data)
{
	struct vector_cpp_interface *src_vector = (struct vector_cpp_interface *) src_interface;
	struct vector_cpp_interface *dst_vector = (struct vector_cpp_interface *) dst_interface;
	int ret;

	ret = starpu_interface_copy(src_vector->dev_handle, src_vector->offset, src_node,
				    dst_vector->dev_handle, dst_vector->offset, dst_node,
				    src_vector->nx*src_vector->elemsize, async_data);

	return ret;
}

static starpu_ssize_t vector_cpp_describe(void *data_interface, char *buf, size_t size)
{
	struct vector_cpp_interface *vector = (struct vector_cpp_interface *) data_interface;
	return snprintf(buf, size, "V%ux%u",
			(unsigned) vector->nx,
			(unsigned) vector->elemsize);
}

/*
 * End of interface
 */



/* Kernel using STL objects */

void cpu_kernel_add_vectors(void *buffers[], void *cl_arg)
{
	std::vector<MY_TYPE>* vec_A = VECTOR_CPP_GET_VEC(buffers[0]);
	std::vector<MY_TYPE>* vec_B = VECTOR_CPP_GET_VEC(buffers[1]);
	std::vector<MY_TYPE>* vec_C = VECTOR_CPP_GET_VEC(buffers[2]);

	// all the std::vector have to have the same size
	assert(vec_A->size() == vec_B->size() && vec_B->size() == vec_C->size());

	// performs the vector addition (vec_C[] = vec_A[] + vec_B[])
	for (size_t i = 0; i < vec_C->size(); i++)
		(*vec_C)[i] = (*vec_A)[i] + (*vec_B)[i];
}

#define VEC_SIZE 1024

int main(int argc, char **argv)
{
	struct starpu_conf conf;
	bool fail;

	starpu_conf_init(&conf);
	conf.nmic = 0;
	conf.nmpi_ms = 0;

	// initialize StarPU with default configuration
	int ret = starpu_init(&conf);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	{
		/* Test data transfers between NUMA nodes if available */
		unsigned last_numa_node = starpu_memory_nodes_get_numa_count() - 1;

		const my_allocator<char> allocator_main_ram(STARPU_MAIN_RAM);
		const my_allocator<char> allocator_last_numa(last_numa_node);
		std::vector<MY_TYPE> vec_A(VEC_SIZE, 2, allocator_main_ram); // all the vector is initialized to 2
		std::vector<MY_TYPE> vec_B(VEC_SIZE, 3, allocator_main_ram); // all the vector is initialized to 3
		std::vector<MY_TYPE> vec_C(VEC_SIZE, 0, allocator_last_numa); // all the vector is initialized to 0

		// StarPU data registering
		starpu_data_handle_t spu_vec_A;
		starpu_data_handle_t spu_vec_B;
		starpu_data_handle_t spu_vec_C;

		// give the data of the vector to StarPU (C array)
		vector_cpp_data_register(&spu_vec_A, STARPU_MAIN_RAM, &vec_A, vec_A.size(), sizeof(char));
		vector_cpp_data_register(&spu_vec_B, STARPU_MAIN_RAM, &vec_B, vec_B.size(), sizeof(char));
		vector_cpp_data_register(&spu_vec_C, last_numa_node, &vec_C, vec_C.size(), sizeof(char));

		// create the StarPU codelet
		starpu_codelet cl;
		starpu_codelet_init(&cl);
		cl.cpu_funcs     [0] = cpu_kernel_add_vectors;
		cl.cpu_funcs_name[0] = "cpu_kernel_add_vectors";
		cl.nbuffers          = 3;
		cl.modes         [0] = STARPU_R;
		cl.modes         [1] = STARPU_R;
		cl.modes         [2] = STARPU_W;
		cl.name              = "add_vectors";

		// submit a new StarPU task to execute
		ret = starpu_task_insert(&cl,
					 STARPU_R, spu_vec_A,
					 STARPU_R, spu_vec_B,
					 STARPU_W, spu_vec_C,
					 0);
		if (ret == -ENODEV)
		{
			// StarPU data unregistering
			starpu_data_unregister(spu_vec_C);
			starpu_data_unregister(spu_vec_B);
			starpu_data_unregister(spu_vec_A);

			// terminate StarPU, no task can be submitted after
			starpu_shutdown();

			return 77;
		}

		STARPU_CHECK_RETURN_VALUE(ret, "task_submit::add_vectors");

		// wait the task
		starpu_task_wait_for_all();

		// StarPU data unregistering
		starpu_data_unregister(spu_vec_C);
		starpu_data_unregister(spu_vec_B);
		starpu_data_unregister(spu_vec_A);

		// check results
		fail = false;
		int i = 0;
		while (!fail && i < VEC_SIZE)
			fail = vec_C[i++] != 5;
	}

	// terminate StarPU, no task can be submitted after
	starpu_shutdown();

	if (fail)
	{
#ifdef PRINT_OUTPUT
		std::cout << "Example failed..." << std::endl;
#endif
		return EXIT_FAILURE;
	}
	else
	{
#ifdef PRINT_OUTPUT
		std::cout << "Example successfully passed!" << std::endl;
#endif
		return EXIT_SUCCESS;
	}
}
#endif
