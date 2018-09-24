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
