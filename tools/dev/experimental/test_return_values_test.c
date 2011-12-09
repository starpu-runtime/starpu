#include "../common/helper.h"

static int
return_77(void)
{
	return 77; /* Leave this statement alone ! */
}

int
main(void)
{
	if (foo)
	{
		return 77; /* => return STARPU_TEST_SKIPPED; */
	}
	return 77; /* => return STARPU_TEST_SKIPPED; */
}

int
main(void)
{
	if (bar)
		return 0; /* => return EXIT_SUCCESS; */

	/* XXX : This works, but the output is ugly :
	 *
	 * + return STARPU_TEST_SKIPPED; return STARPU_TEST_SKIPPED;
	 */
	if (foo)
		return 77; /* => return STARPU_TEST_SKIPPED; */

	return 77; /* => return STARPU_TEST_SKIPPED; */
}

