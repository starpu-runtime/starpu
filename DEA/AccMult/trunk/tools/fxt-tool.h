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

#include "histo-flash.h"

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

extern void flash_engine_generate_output(event_list_t *events, workq_list_t taskq, char **worker_name,
                      unsigned nworkers, unsigned maxq_size, 
                      uint64_t _start_time, uint64_t _end_time, char *path);



#endif // __FXT_TOOL_H__
