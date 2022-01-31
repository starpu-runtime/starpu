#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>

#include "StreamFMA.h"
#include "MaxSLiCInterface.h"

int main()
{
	const int size = 400;
	int sizeBytes = size * sizeof(int32_t);
	int32_t *a = (int32_t*) malloc(sizeBytes);
	int32_t *b = (int32_t*) malloc(sizeBytes);
	int32_t *c = (int32_t*) malloc(sizeBytes);

	// TODO Generate input data
	for(int i = 0; i < size; ++i)
	{
		a[i] = random() % 100;
		b[i] = random() % 100;
	}
	max_file_t *maxfile = StreamFMA_init();
	max_engine_t *engine = max_load(maxfile, "*");

	max_actions_t* act = max_actions_init(maxfile, NULL);

	max_set_ticks  (act, "StreamFMAKernel", size);
	max_queue_input(act, "a", a, size * sizeof(int32_t));
	max_queue_input(act, "b", b, size * sizeof(int32_t));
	max_queue_output(act, "output", c, size * sizeof(int32_t));
	max_run(engine, act);

	max_actions_free(act);
	max_unload(engine);

	int ret = 0;
	// TODO Use result data
	for(std::size_t i = 0; i < size; ++i)
	{
		int32_t ref =a[i] + b[i];
		if (c[i] != ref)
		{
			std::cout << "Invalid Output at index " << i << ": " << std::endl;
			std::cout << " reference: " << ref << std::endl;
			std::cout << " value:     " << c[i] << std::endl;
			ret = 1;
			break;
		}
	}

	if(0 == ret)
	{
		std::cout << "All " << size << " values calculated correctly on the DFE!" << std::endl;
	}

	std::cout << "Done." << std::endl;
	return ret;
}
