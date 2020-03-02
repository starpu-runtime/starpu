/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_hash.h>
#include <stdlib.h>
#include <string.h>

#define _STARPU_CRC32C_POLY_BE 0x1EDC6F41

static inline uint32_t STARPU_ATTRIBUTE_PURE starpu_crc32c_be_8(uint8_t inputbyte, uint32_t inputcrc)
{
	unsigned i;
	uint32_t crc;

	crc = inputcrc ^ (((uint32_t) inputbyte) << 24);
	for (i = 0; i < 8; i++)
		crc = (crc << 1) ^ ((crc & 0x80000000) ? _STARPU_CRC32C_POLY_BE : 0);

	return crc;
}

uint32_t starpu_hash_crc32c_be_n(const void *input, size_t n, uint32_t inputcrc)
{
	uint8_t *p = (uint8_t *)input;
	size_t i;

	uint32_t crc = inputcrc;

	for (i = 0; i < n; i++)
		crc = starpu_crc32c_be_8(p[i], crc);

	return crc;
}

uint32_t starpu_hash_crc32c_be(uint32_t input, uint32_t inputcrc)
{
	uint8_t *p = (uint8_t *)&input;

	uint32_t crc = inputcrc;

	crc = starpu_crc32c_be_8(p[0], crc);
	crc = starpu_crc32c_be_8(p[1], crc);
	crc = starpu_crc32c_be_8(p[2], crc);
	crc = starpu_crc32c_be_8(p[3], crc);

	return crc;
}

uint32_t starpu_hash_crc32c_string(const char *str, uint32_t inputcrc)
{
	uint32_t hash = inputcrc;

	size_t len = strlen(str);

	unsigned i;
	for (i = 0; i < len; i++)
	{
		hash = starpu_crc32c_be_8((uint8_t)str[i], hash);
	}

	return hash;
}
