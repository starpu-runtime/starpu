#ifndef SORTING_H
#define SORTING_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

unsigned chose_pivot(int first, int last);
int partitionning(double *arr, int first, int last, int pivot);
void quicksort(double *arr, int first, int last);

#endif
