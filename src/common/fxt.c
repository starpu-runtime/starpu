#include <common/fxt.h>

#define PROF_BUFFER_SIZE  (8*1024*1024)

static char PROF_FILE_USER[128];
static int fxt_started = 0;
static fxt_t fut;

void profile_stop(void)
{
	fut_endup(PROF_FILE_USER);
}

void profile_set_tracefile(char *fmt, ...)
{
	va_list vl;
	
	va_start(vl, fmt);
	vsprintf(PROF_FILE_USER, fmt, vl);
	va_end(vl);
	strcat(PROF_FILE_USER, "_user_");
}


void start_fxt_profiling(void)
{
	unsigned threadid;

	if (!fxt_started) {
		fxt_started = 1;
		profile_set_tracefile("/tmp/prof_file");
	}

	threadid = syscall(SYS_gettid);

	atexit(profile_stop);

	if(fut_setup(PROF_BUFFER_SIZE, FUT_KEYMASKALL, threadid) < 0) {
		perror("fut_setup");
		STARPU_ASSERT(0);
	}

	//fxt_register_thread(-1);

	fut_get_mysymbols(fut);
	
	fut_keychange(FUT_ENABLE, FUT_KEYMASKALL, threadid);

	return;
}

void fxt_register_thread(unsigned coreid)
{
	FUT_DO_PROBE2(FUT_NEW_LWP_CODE, coreid, syscall(SYS_gettid));
}
