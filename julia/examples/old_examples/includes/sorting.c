/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2019       Mael Keryell
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

#include "sorting.h"

unsigned chose_pivot(int first, int last)
{
	return (rand() % (last - first + 1) + first);
}

int partitionning(double *arr, int first, int last, int pivot)
{
	double tmp;
	int i,j;

	tmp = arr[last];
	arr[last] = arr[pivot];
	arr[pivot] = tmp;

	j = first;

	for (i = first; i < last; i++)
	{
		if (arr[i] <= arr[last])
		{
			tmp = arr[i];

			arr[i] = arr[j];
			arr[j] = tmp;

			j++;
		}
	}

	tmp = arr[j];
	arr[j] = arr[last];
	arr[last] = tmp;

	return j;
}

void quicksort(double *arr, int first, int last)
{
	if (first < last)
	{
		int pivot = chose_pivot(first, last);
		int j;

		j = partitionning(arr, first, last, pivot);

		quicksort(arr, first, j - 1);
		quicksort(arr, j + 1, last);
	}
}
