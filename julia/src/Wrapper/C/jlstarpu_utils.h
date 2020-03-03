/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2018-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2018       Alexis Juven
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
 * jlstarpu_utils.h
 *
 *  Created on: 27 juin 2018
 *      Author: ajuven
 */

#ifndef JLSTARPU_UTILS_H_
#define JLSTARPU_UTILS_H_

#include "jlstarpu.h"


#define TYPE_MALLOC(ptr, nb_elements) \
		do {\
			if ((nb_elements) == 0){ \
				ptr = NULL; \
			} else { \
				ptr = malloc((nb_elements) * sizeof(*(ptr))); \
				if (ptr == NULL){ \
					fprintf(stderr, "\033[31mCRITICAL : MALLOC HAS RETURNED NULL\n\033[0m");\
					fflush(stderr);\
					exit(1);\
				} \
			} \
		} while(0)



//#define DEBUG
#ifdef DEBUG

#define DEBUG_PRINT(...)\
		do {\
			fprintf(stderr, "\x1B[34m%s : \x1B[0m", __FUNCTION__);\
			fprintf(stderr, __VA_ARGS__);\
			fprintf(stderr, "\n");\
			fflush(stderr);\
		} while (0)




#else

#define DEBUG_PRINT(...)

#endif



#endif /* JLSTARPU_UTILS_H_ */
