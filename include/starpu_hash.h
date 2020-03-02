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

#ifndef __STARPU_HASH_H__
#define __STARPU_HASH_H__

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @ingroup API_Data_Interfaces
   @{
*/

/**
   Compute the CRC of a byte buffer seeded by the \p inputcrc
   <em>current state</em>. The return value should be considered as the new
   <em>current state</em> for future CRC computation. This is used for computing
   data size footprint.
*/
uint32_t starpu_hash_crc32c_be_n(const void *input, size_t n, uint32_t inputcrc);

/**
   Compute the CRC of a 32bit number seeded by the \p inputcrc
   <em>current state</em>. The return value should be considered as the new
   <em>current state</em> for future CRC computation. This is used for computing
   data size footprint.
*/
uint32_t starpu_hash_crc32c_be(uint32_t input, uint32_t inputcrc);

/**
   Compute the CRC of a string seeded by the \p inputcrc <em>current
   state</em>. The return value should be considered as the new <em>current
   state</em> for future CRC computation. This is used for computing data
   size footprint.
*/
uint32_t starpu_hash_crc32c_string(const char *str, uint32_t inputcrc);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_HASH_H__ */
