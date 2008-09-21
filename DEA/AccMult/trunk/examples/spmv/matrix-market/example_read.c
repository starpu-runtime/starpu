#include "mm_to_bcsr.h"

int main(int argc, char *argv[])
{
	unsigned c, r;

	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename] [c] [r]\n", argv[0]);
		exit(1);
	}

	c = 64;
	r = 64;

	bcsr_t *bcsr;
	bcsr = mm_file_to_bcsr(argv[1], c, r);

	return 0;
}
