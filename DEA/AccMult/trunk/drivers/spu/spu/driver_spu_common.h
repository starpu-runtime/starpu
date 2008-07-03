#ifndef __DRIVER_SPU_COMMON_H__
#define __DRIVER_SPU_COMMON_H__

#include <stdint.h>

typedef struct {
	uint32_t *ea_ready_flag;
	uint32_t deviceid;
	uint8_t pad[16-sizeof(uint32_t *)-sizeof(uint32_t)];
} spu_init_arguments;

#endif // __DRIVER_SPU_COMMON_H__

