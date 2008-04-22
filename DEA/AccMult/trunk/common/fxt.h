#ifndef __FXT_H__
#define __FXT_H__

#define _GNU_SOURCE  /* ou _BSD_SOURCE ou _SVID_SOURCE */
#include <unistd.h>
#include <sys/syscall.h> /* pour les définitions de SYS_xxx */

#include <string.h>
#include <sys/types.h>
#include <stdlib.h>
#include <common/util.h>

#include <fxt/fxt.h>
#include <fxt/fut.h>

//FUT_DO_PROBE2(0x7337, 1, syscall(SYS_gettid));

void start_fxt_profiling(void);
void fxt_register_thread(unsigned);

#endif // __FXT_H__
