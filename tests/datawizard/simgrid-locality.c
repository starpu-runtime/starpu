/* Check that defining a main makes starpu use MSG_process_attach. */
#include "locality.c"
#include <config.h>
#if defined(HAVE_MSG_PROCESS_ATTACH) && SIMGRID_VERSION_MAJOR > 3 || (SIMGRID_VERSION_MAJOR == 3 && SIMGRID_VERSION_MINOR >= 14)
#undef main
int main(int argc, char *argv[]) {
	return starpu_main(argc, argv);
}
#endif
