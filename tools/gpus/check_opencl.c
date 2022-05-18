#include <CL/cl.h>
#include <stdio.h>
#include <assert.h>

int main(void)
{
	cl_device_id did[16];
	cl_int err;
	cl_platform_id pid, pids[16];
	cl_uint nbplat, nb;
	char buf[128];
	size_t size;
	int i, j;

	err = clGetPlatformIDs(sizeof(pids)/sizeof(pids[0]), pids, &nbplat);
	assert(err == CL_SUCCESS);
	printf("%u platforms\n", nbplat);
	for (j = 0; j < nbplat; j++)
	{
		pid = pids[j];
		printf("    platform %d\n", j);
		err = clGetPlatformInfo(pid, CL_PLATFORM_VERSION, sizeof(buf)-1, buf, &size);
		assert(err == CL_SUCCESS);
		buf[size] = 0;
		printf("        platform version %s\n", buf);
		err = clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, sizeof(did)/sizeof(did[0]), did, &nb);
		if (err == CL_DEVICE_NOT_FOUND)
			nb = 0;
		else
			assert(err == CL_SUCCESS);
		printf("%d devices\n", nb);
		for (i = 0; i < nb; i++)
		{
			err = clGetDeviceInfo(did[i], CL_DEVICE_VERSION, sizeof(buf)-1, buf, &size);
			buf[size] = 0;
			printf("    device %d version %s\n", i, buf);
		}
	}
	return 0;
}
