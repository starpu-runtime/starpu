/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <common/utils.h>
#include "../helper.h"
#include <unistd.h>

int do_test(char *(*func)(char *tmpl))
{
	int ret;
	char *path;
	char dirname[128];
	char *ptr;
	struct stat sb;

	path = starpu_getenv("TMPDIR");
	if (!path)
		path = starpu_getenv("TEMP");
	if (!path)
		path = starpu_getenv("TMP");
	if (!path)
		path = "/tmp";
	snprintf(dirname, sizeof(dirname), "%s/abcdef_XXXXXX", path);
	ptr = func(dirname);
	FPRINTF(stderr, "Directory '%s' (res '%s')\n", dirname, ptr);

	// use stat
	ret = stat(dirname, &sb);
	if (ret != 0 || !S_ISDIR(sb.st_mode))
	{
		FPRINTF(stderr, "Directory '%s' has not been created\n", dirname);
		return 1;
	}

	ret = rmdir(dirname);
	STARPU_CHECK_RETURN_VALUE(ret, "rmdir '%s'\n", dirname);

	return ret;
}

int main(void)
{
	int ret, ret2;

	ret = do_test(_starpu_mkdtemp);
	ret2 = do_test(_starpu_mkdtemp_internal);

	return ret + ret2;
}
