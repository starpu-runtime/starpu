#ifndef __FUNC_SGEMM_IBM_H__
#define __FUNC_SGEMM_IBM_H__

#include <stdint.h>

struct ibm_sgemm_block_conf {
	uint32_t m;
	uint32_t n;
	uint32_t k;
	uint32_t pad;
};

#endif // __FUNC_SGEMM_IBM_H__
