/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Centre National de la Recherche Scientifique
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

#include <errno.h>

#define STARPU_TEST_SKIPPED 77

#define STARPU_CHECK_RETURN_VALUE(err, message) {if (err < 0) { \
			fprintf(stderr, "StarPU function <%s> returned unexpected value: <%s>\n", message, strerror(err)); \
			STARPU_ASSERT(0); }}
#define STARPU_CHECK_RETURN_VALUE_IS(err, value, message) {if (err != value) { \
			fprintf(stderr, "StarPU function <%s> returned unexpected value: <%s>\n", message, strerror(err)); \
			STARPU_ASSERT(0); }}

#define FPRINTF(ofile, fmt, args ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ##args); }} while(0)

