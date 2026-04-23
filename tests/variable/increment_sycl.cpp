/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../helper.h"
#include <sycl/sycl.hpp>

void _increment_sycl(unsigned *val)
{
	sycl::queue q;
        sycl::buffer<unsigned, 1> buf(val, sycl::range<1>(1));

        q.submit([&](sycl::handler& h)
	{
		auto val = buf.get_access<sycl::access::mode::read_write>(h);
		h.single_task([=]()
		{
			val[0] += 1;
		});
        });
}

extern "C" void increment_sycl(void *descr[], void *cl_arg)
{
	STARPU_SKIP_IF_VALGRIND;
	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	_increment_sycl(dst);
}

void _redux_sycl(unsigned *dst, unsigned *src)
{
	sycl::queue q;
	sycl::buffer<unsigned, 1> dst_buf(dst, sycl::range<1>(1));
	sycl::buffer<unsigned, 1> src_buf(src, sycl::range<1>(1));
	q.submit([&](sycl::handler& h)
	{
		auto dst_acc = dst_buf.get_access<sycl::access::mode::read_write>(h);
		auto src_acc = src_buf.get_access<sycl::access::mode::read>(h);
		h.single_task([=]()
		{
			dst_acc[0] += src_acc[0];
		});
	});
}

extern "C" void redux_sycl(void *descr[], void *arg)
{
	STARPU_SKIP_IF_VALGRIND;
	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *src = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);
	_redux_sycl(dst, src);
}

void _neutral_sycl(unsigned *val)
{
	sycl::queue q;
	sycl::buffer<unsigned, 1> buf(val, sycl::range<1>(1));
	q.submit([&](sycl::handler& h)
	{
		auto acc = buf.get_access<sycl::access::mode::write>(h);
		h.single_task([=]()
		{
			acc[0] = 0u;
		});
	});
}

extern "C" void neutral_sycl(void *descr[], void *arg)
{
	STARPU_SKIP_IF_VALGRIND;

	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	_neutral_sycl(dst);
}
