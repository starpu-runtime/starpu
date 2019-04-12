#include "sorting.h"

unsigned chose_pivot(int first, int last)
{
	return ( rand() % (last - first + 1) + first );
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

		if (arr[i] <= arr[last]){
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

	if (first < last){
		int pivot = chose_pivot(first, last);
		int j;

		j = partitionning(arr, first, last, pivot);

		quicksort(arr, first, j - 1);
		quicksort(arr, j + 1, last);
	}

}
