#ifndef __FXT_TOOL_H__
#define __FXT_TOOL_H__

#include <search.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <common/fxt.h>
#include <common/list.h>

#define MAXWORKERS      32
#define FACTOR  100

typedef enum {
	WORKING,
	IDLE
} worker_mode;

LIST_TYPE(event,
        uint64_t time;
	worker_mode mode;
);

LIST_TYPE(workq,
	uint64_t time;
	int diff;
	int current_size;
);

#endif // __FXT_TOOL_H__
