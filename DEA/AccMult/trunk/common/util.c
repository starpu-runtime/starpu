#include <math.h>
#include "util.h"

/* some function useful to manipulate matrices */
void matrix_fill_rand(matrix *m)
{
        unsigned i,j;
        for (i=0; i < m->width; i++) {
                for (j=0; j < m->heigth; j++) {
                        m->data[i+j*m->width] = (float)(drand48());
                        //m->data[i+j*m->width] = (float)(i==j?1.0:0.0);
                }
        }
}

void matrix_fill_zero(matrix *m)
{
        memset(m->data, 0, m->width*m->heigth*sizeof(float));
}

void alloc_matrix(matrix *m, unsigned width, unsigned heigth)
{
        m->width = width;
        m->heigth = heigth;
        m->data = malloc(width*heigth*sizeof(float));
}

void free_matrix(matrix *m)
{
        free(m->data);
}

void display_matrix(matrix *m)
{
        unsigned x,y;

        fprintf(stderr, "****************************\n");
        for (y = 0; y < m->heigth; y++) {
        for (x = 0; x < m->width; x++) {
                fprintf(stderr, "%f\t", m->data[x+y*m->width]);
        }
        fprintf(stderr, "\n");
        }
        fprintf(stderr, "****************************\n");
}

void compare_matrix(matrix *A, matrix *B, float eps) 
{ 
        int isdiff = 0; 
        int ndiff = 0; 
        int ntotal = 0; 
 
        unsigned x,y; 
        for (x = 0; x < A->width; x++)  
        { 
                for (y = 0; y < A->heigth ; y++)  
                { 
                        if (fabs(A->data[x+y*A->width] - B->data[x+y*A->width]) > eps) { 
                                isdiff = 1; 
                                ndiff++; 
                                fprintf(stderr, "(%d,%d) expecting %f got %f\n", x, y,  B->data[x+y*A->width],  A->data[x+y*A->width]); 
                        } 
                        ntotal++; 
                } 
        } 
 
        if (isdiff) { 
                printf("Matrix are DIFFERENT (%d on %d differs ...)!\n", ndiff, ntotal); 
        } else { 
                printf("Matrix are IDENTICAL\n"); 
        } 
}

