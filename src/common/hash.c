/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <common/hash.h>
#include <stdlib.h>
#include <string.h>

#define CRC32C_POLY_BE 0x1EDC6F41

static inline uint32_t __attribute__ ((pure)) crc32_be_8(uint8_t inputbyte, uint32_t inputcrc)
{
	unsigned i;
	uint32_t crc;

	crc = inputcrc ^ (inputbyte << 24);
	for (i = 0; i < 8; i++)
		crc = (crc << 1) ^ ((crc & 0x80000000) ? CRC32C_POLY_BE : 0);

	return crc;
}

uint32_t _starpu_crc32_be(uint32_t input, uint32_t inputcrc)
{
	uint8_t *p = (uint8_t *)&input;

	uint32_t crc = inputcrc;

	crc = crc32_be_8(p[0], crc);
	crc = crc32_be_8(p[1], crc);
	crc = crc32_be_8(p[2], crc);
	crc = crc32_be_8(p[3], crc);

	return crc;
}

uint32_t _starpu_crc32_string(char *str, uint32_t inputcrc)
{
	uint32_t hash = inputcrc;

	size_t len = strlen(str);

	unsigned i;
	for (i = 0; i < len; i++)
	{
		hash = crc32_be_8((uint8_t)str[i], hash);
	}

	return hash;
}
