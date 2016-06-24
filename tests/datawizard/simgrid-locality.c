/* Check that defining a main makes starpu use MSG_process_attach. */
#include "locality.c"
#include <config.h>
#ifdef HAVE_MSG_PROCESS_ATTACH
#undef main
int main(int argc, char *argv[]) {
	return starpu_main(argc, argv);
}
#endif
