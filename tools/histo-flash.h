/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

#ifndef __HISTO_FLASH_H__
#define __HISTO_FLASH_H__

#include <stdint.h>
#include <stdlib.h>
#include <ming.h>
#include <math.h>

#include "fxt-tool.h"

#define WIDTH	800
#define HEIGHT	600

#define THICKNESS	50
#define GAP		10

#define BORDERX		100
#define BORDERY		100

void flash_engine_init(void);

#endif // __HISTO_FLASH_H__
