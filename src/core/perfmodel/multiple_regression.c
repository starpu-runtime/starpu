/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010, 2011, 2015  Université de Bordeaux
 * Copyright (C) 2010, 2011  CNRS
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

/* Code for computing multiple linear regression */

#include <core/perfmodel/multiple_regression.h>

#ifdef TESTGSL
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_multifit.h>
#endif //TESTGSL

typedef struct { int h, w; double *x;} matrix_t, *matrix;

static long count_file_lines(FILE *f)
{
	int ch, lines=0;
	while(!feof(f))
	{
		ch = fgetc(f);
	    if(ch == '\n')
	    {
		  lines++;
		}
    }
	rewind(f);

	return lines;
}

static void dump_multiple_regression_list(double *mpar, double *my, int start, unsigned nparameters, struct starpu_perfmodel_history_list *list_history)
{
	struct starpu_perfmodel_history_list *ptr = list_history;
	int i = start;
	while (ptr)
	{
		my[i] = ptr->entry->duration;
		for(int j=0; j<nparameters; j++)
			mpar[i*nparameters+j] = ptr->entry->parameters[j];
		ptr = ptr->next;
		i++;
	}

}

static void load_old_calibration(double *mx, double *my, unsigned nparameters, FILE *f)
{
	char buffer[1024];
	char *record,*line;
	int i=0,j=0;

	line=fgets(buffer,sizeof(buffer),f);//skipping first line
	while((line=fgets(buffer,sizeof(buffer),f))!=NULL)
	{
		record = strtok(line,",");
		my[i] = atof(record);
		record = strtok(NULL,",");
		j=0;
		while(record != NULL)
		{
			mx[i*nparameters+j] = atof(record) ;
			++j;
			record = strtok(NULL,",");
		}
		++i ;
	}
}

static long find_long_list_size(struct starpu_perfmodel_history_list *list_history)
{
	long cnt = 0;

	struct starpu_perfmodel_history_list *ptr = list_history;
	while (ptr)
	{
		cnt++;
		ptr = ptr->next;
	}

	return cnt;
}

#ifdef TESTGSL
void gsl_multiple_reg_coeff(double *mpar, double *my, long n, unsigned ncoeff, unsigned nparameters, double *coeff, unsigned **combinations)
{
	double coefficient;
	gsl_matrix *X = gsl_matrix_calloc(n, ncoeff);
	gsl_vector *Y = gsl_vector_alloc(n);
	gsl_vector *beta = gsl_vector_alloc(ncoeff);

	for (int i = 0; i < n; i++) {
		gsl_vector_set(Y, i, my[i]);
		gsl_matrix_set(X, i, 0, 1.);
		for (int j = 1; j < ncoeff; j++)
		{
			coefficient = 1.;
			for(int k=0; k < nparameters; k++)
			{
				coefficient *= pow(mpar[i*nparameters+k],combinations[j-1][k]);
			}
			gsl_matrix_set(X, i, j, coefficient);
		}
	}

	double chisq;
	gsl_matrix *cov = gsl_matrix_alloc(ncoeff, ncoeff);
	gsl_multifit_linear_workspace * wspc = gsl_multifit_linear_alloc(n, ncoeff);
	gsl_multifit_linear(X, Y, beta, cov, &chisq, wspc);

	for(int i=0; i<ncoeff; i++)
		coeff[i] = gsl_vector_get(beta, i);

	gsl_matrix_free(X);
	gsl_matrix_free(cov);
	gsl_vector_free(Y);
	gsl_vector_free(beta);
	gsl_multifit_linear_free(wspc);
}
#endif //TESTGSL

void dgels_multiple_reg_coeff(double *mpar, double *my, long nn, unsigned ncoeff, unsigned nparameters, double *coeff, unsigned **combinations)
{	

	char trans = 'N';
	integer m = nn;
	integer n = ncoeff;
	integer nrhs = 1; // number of columns of B and X (wich are vectors therefore nrhs=1)
	doublereal *X = malloc(sizeof(double)*n*m); // (/!\ modified at the output) contain the model and the different values of pararmters
	doublereal *Y = malloc(sizeof(double)*m);

	double coefficient;
	for (int i=0; i < m; i++)
	{
		Y[i] = my[i];
		X[i*n] = 1.;
		for (int j=1; j < n; j++)
			coefficient = 1.;
			for(int k=0; k < nparameters; k++)
			{
				coefficient *= pow(mpar[i*nparameters+k],combinations[j-1][k]);
			}			
			X[i*n+j] = coefficient;
	}

	integer lda = m; 
	integer ldb = m; //
	integer info;

	integer lwork = n*2;
	doublereal *work = malloc(sizeof(double)*lwork); // (output)

	/* // Running CLAPACK */
	dgels_(&trans, &m, &n, &nrhs, X, &lda, Y, &ldb, work, &lwork, &info);

	/* Check for the full rank */
	if( info != 0 )
	{
		printf( "Problems with DGELS; info=%i\n");
		exit(1);
	}
	
	for(int i=0; i<ncoeff; i++)
		coeff[i] = Y[i];

	free(X);
	free(Y);
	free(work);
}

int _starpu_multiple_regression(struct starpu_perfmodel_history_list *ptr, double *coeff, unsigned ncoeff, unsigned nparameters, unsigned **combinations, char *codelet_name)
{
	// Computing number of rows
	long n=find_long_list_size(ptr);
	STARPU_ASSERT(n);
	
        // Reading old calibrations if necessary
	FILE *f;
	char filepath[50];
	snprintf(filepath, 50, "/tmp/%s.out", codelet_name);
	long old_lines=0;
	int calibrate = starpu_get_env_number("STARPU_CALIBRATE");	
	if (calibrate==1)
	{
		f = fopen(filepath, "a+");
		STARPU_ASSERT_MSG(f, "Could not save performance model %s\n", filepath);
		
		old_lines=count_file_lines(f);
		STARPU_ASSERT(old_lines);

		n+=old_lines;
	}

	// Allocating X and Y matrices
	double *mpar = (double *) malloc(nparameters*n*sizeof(double));
	STARPU_ASSERT(mpar);
	double *my = (double *) malloc(n*sizeof(double));
	STARPU_ASSERT(my);

	// Loading old calibration
	if (calibrate==1)
		load_old_calibration(mpar, my, nparameters, f);

	// Filling X and Y matrices with measured values
	dump_multiple_regression_list(mpar, my, old_lines, nparameters, ptr);
	
	// Computing coefficients using multiple linear regression
#ifdef TESTGSL
	gsl_multiple_reg_coeff(mpar, my, n, ncoeff, nparameters, coeff, combinations);
#elseif	
	dgels_multiple_reg_coeff(mpar, my, n, ncoeff, nparameters, coeff, combinations);
#endif
	
	// Preparing new output calibration file
	if (calibrate==2)
	{
		f = fopen(filepath, "w+");
		STARPU_ASSERT_MSG(f, "Could not save performance model %s\n", filepath);
		fprintf(f, "Duration");
		for(int k=0; k < nparameters; k++)
		{
			fprintf(f, ", P%d", k);
		}
	}
	
	// Writing parameters to calibration file
	if (calibrate>0)
	{
		for(int i=old_lines; i<n; i++)
		{
			fprintf(f, "\n%f", my[i]);
			for(int j=0; j<nparameters;j++)
				fprintf(f, ", %f", mpar[i*nparameters+j]);
		}
		fclose(f);
	}

	// Cleanup
	free(mpar);
	free(my);

	return 0;
}
