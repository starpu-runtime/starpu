/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2018       Umeà University
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

typedef long int integer;
typedef double doublereal;

#ifdef STARPU_MLR_MODEL
#ifdef STARPU_BUILT_IN_MIN_DGELS
int _starpu_dgels_(char *trans, integer *m, integer *n, integer *nrhs, doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *work, integer *lwork, integer *info);
#else
int dgels_(char *trans, integer *m, integer *n, integer *nrhs, doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *work, integer *lwork, integer *info);
#endif
#endif //STARPU_MLR_MODEL

static unsigned long count_file_lines(FILE *f)
{
	unsigned long  lines=0;
	while(!feof(f))
	{
		int ch = fgetc(f);
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
	unsigned j;
	while (ptr)
	{
		my[i] = ptr->entry->duration;
		for(j=0; j<nparameters; j++)
			mpar[i*nparameters+j] = ptr->entry->parameters[j];
		ptr = ptr->next;
		i++;
	}

}

static void load_old_calibration(double *mx, double *my, unsigned nparameters, char *filepath)
{
	char buffer[1024];
	char *line;
	int i=0;

	FILE *f = fopen(filepath, "a+");
	STARPU_ASSERT_MSG(f, "Could not load performance model from file %s\n", filepath);

	line = fgets(buffer,sizeof(buffer),f);//skipping first line
	STARPU_ASSERT(line);
	while((line=fgets(buffer,sizeof(buffer),f))!=NULL)
	{
		char *record = strtok(line,",");
		STARPU_ASSERT_MSG(record, "Could not load performance model from file %s\n", filepath);
		my[i] = atof(record);
		record = strtok(NULL,",");
		int j=0;
		while(record != NULL)
		{
			mx[i*nparameters+j] = atof(record) ;
			++j;
			record = strtok(NULL,",");
		}
		++i ;
	}

	fclose(f);
}

static unsigned long find_long_list_size(struct starpu_perfmodel_history_list *list_history)
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

#ifdef STARPU_MLR_MODEL
int dgels_multiple_reg_coeff(double *mpar, double *my, unsigned long nn, unsigned ncoeff, unsigned nparameters, double *coeff, unsigned **combinations)
{
 /*  Arguments */
/*  ========= */

/*  TRANS   (input) CHARACTER*1 */
/*          = 'N': the linear system involves A; */
/*          = 'T': the linear system involves A**T. */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix A.  M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix A.  N >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of */
/*          columns of the matrices B and X. NRHS >=0. */

/*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N) */
/*          On entry, the M-by-N matrix A. */
/*          On exit, */
/*            if M >= N, A is overwritten by details of its QR */
/*                       factorization as returned by DGEQRF; */
/*            if M <  N, A is overwritten by details of its LQ */
/*                       factorization as returned by DGELQF. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS) */
/*          On entry, the matrix B of right hand side vectors, stored */
/*          columnwise; B is M-by-NRHS if TRANS = 'N', or N-by-NRHS */
/*          if TRANS = 'T'. */
/*          On exit, if INFO = 0, B is overwritten by the solution */
/*          vectors, stored columnwise: */
/*          if TRANS = 'N' and m >= n, rows 1 to n of B contain the least */
/*          squares solution vectors; the residual sum of squares for the */
/*          solution in each column is given by the sum of squares of */
/*          elements N+1 to M in that column; */
/*          if TRANS = 'N' and m < n, rows 1 to N of B contain the */
/*          minimum norm solution vectors; */
/*          if TRANS = 'T' and m >= n, rows 1 to M of B contain the */
/*          minimum norm solution vectors; */
/*          if TRANS = 'T' and m < n, rows 1 to M of B contain the */
/*          least squares solution vectors; the residual sum of squares */
/*          for the solution in each column is given by the sum of */
/*          squares of elements M+1 to N in that column. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B. LDB >= MAX(1,M,N). */

/*  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK. */
/*          LWORK >= max( 1, MN + max( MN, NRHS ) ). */
/*          For optimal performance, */
/*          LWORK >= max( 1, MN + max( MN, NRHS )*NB ). */
/*          where MN = min(M,N) and NB is the optimum block size. */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal size of the WORK array, returns */
/*          this value as the first entry of the WORK array, and no error */
/*          message related to LWORK is issued by XERBLA. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  if INFO =  i, the i-th diagonal element of the */
/*                triangular factor of A is zero, so that A does not have */
/*                full rank; the least squares solution could not be */
/*                computed. */

/*  ===================================================================== */

	if(nn <= ncoeff)
	{
		_STARPU_DISP("Warning: This function is not intended for the use when number of parameters is larger than the number of observations. Check how your matrices A and B were allocated or simply add more benchmarks.\n Multiple linear regression model will not be written into perfmodel file.\n");
		return 1;
	}

	char trans = 'N';
	integer m = nn;
	integer n = ncoeff;
	integer nrhs = 1; // number of columns of B and X (wich are vectors therefore nrhs=1)
	doublereal *X;
	_STARPU_MALLOC(X, sizeof(double)*n*m); // (/!\ modified at the output) contain the model and the different values of pararmters
	doublereal *Y;
	_STARPU_MALLOC(Y, sizeof(double)*m);

	double coefficient;
	int i, j;
	unsigned k;
	for (i=0; i < m; i++)
	{
		Y[i] = my[i];
		X[i] = 1.;
		for (j=1; j < n; j++)
		{
			coefficient = 1.;
			for(k=0; k < nparameters; k++)
			{
				coefficient *= pow(mpar[i*nparameters+k],combinations[j-1][k]);
			}
			X[i+j*m] = coefficient;
		}
	}

	integer lda = m;
	integer ldb = m; //
	integer info = 0;

	integer lwork = n*2;
	doublereal *work; // (output)
	_STARPU_MALLOC(work, sizeof(double)*lwork);

	/* // Running LAPACK dgels_ */
#ifdef STARPU_BUILT_IN_MIN_DGELS
	_starpu_dgels_(&trans, &m, &n, &nrhs, X, &lda, Y, &ldb, work, &lwork, &info);
#else
	dgels_(&trans, &m, &n, &nrhs, X, &lda, Y, &ldb, work, &lwork, &info);
#endif

	/* Check for the full rank */
	if( info != 0 )
	{
		_STARPU_DISP("Warning: Problems when executing dgels_ function. It seems like the diagonal element %ld is zero.\n Multiple linear regression model will not be written into perfmodel file.\n", info);
		free(X);
		free(Y);
		free(work);
		return 1;
	}

	/* Copy computed coefficients */
	for(i=0; i<(int) ncoeff; i++)
		coeff[i] = Y[i];

	free(X);
	free(Y);
	free(work);

	return 0;
}
#endif //STARPU_MLR_MODEL

/*
   Validating the accuracy of the coefficients.
   For the the validation is extremely basic, but it should be improved.
 */
void starpu_validate_mlr(double *coeff, unsigned ncoeff, const char *codelet_name)
{
	unsigned i;
	if (coeff[0] < 0)
		_STARPU_DISP("Warning: Constant computed by least square method is negative (%f). The model %s is likely to be inaccurate.\n", coeff[0], codelet_name);

	for(i=1; i<ncoeff; i++)
		if(fabs(coeff[i]) < 1E-10)
			_STARPU_DISP("Warning: Coefficient computed by least square method is extremelly small (%f). The model %s is likely to be inaccurate.\n", coeff[i], codelet_name);
}

int _starpu_multiple_regression(struct starpu_perfmodel_history_list *ptr, double *coeff, unsigned ncoeff, unsigned nparameters, const char **parameters_names, unsigned **combinations, const char *codelet_name)
{
        unsigned long i;
	unsigned j;

	/* Computing number of rows */
	unsigned n=find_long_list_size(ptr);

        /* Reading old calibrations if necessary */
	FILE *f=NULL;

	char directory[300];
	snprintf(directory, sizeof(directory), "%s/.starpu/sampling/codelets/tmp", _starpu_get_home_path());
	_starpu_mkpath_and_check(directory, S_IRWXU);

	char filepath[400];
	snprintf(filepath, sizeof(filepath), "%s/%s.out", directory,codelet_name);

	unsigned long old_lines=0;
	int calibrate = _starpu_get_calibrate_flag();
	if (calibrate==1)
	{
		f = fopen(filepath, "a+");
		STARPU_ASSERT_MSG(f, "Could not save performance model into the file %s\n", filepath);

		old_lines=count_file_lines(f);
		/* If the program is run for the first time the old_lines will be 0 */
		//STARPU_ASSERT(old_lines);

		n+=old_lines;

		fclose(f);
	}

	/* Allocating X and Y matrices */
	double *mpar;
	_STARPU_MALLOC(mpar, nparameters*n*sizeof(double));
	double *my;
	_STARPU_MALLOC(my, n*sizeof(double));

	/* Loading old calibration */
	if (calibrate==1 && old_lines > 0)
		load_old_calibration(mpar, my, nparameters, filepath);

	/* Filling X and Y matrices with measured values */
	dump_multiple_regression_list(mpar, my, old_lines, nparameters, ptr);

	if (ncoeff!=0 && combinations!=NULL)
	{
#ifdef STARPU_MLR_MODEL
		/* Computing coefficients using multiple linear regression */
		if(dgels_multiple_reg_coeff(mpar, my, n, ncoeff, nparameters, coeff, combinations))
		{
			free(mpar);
			free(my);
			return 1;
		}
		/* Basic validation of the model accuracy */
		starpu_validate_mlr(coeff, ncoeff, codelet_name);
#else
		_STARPU_DISP("Warning: StarPU was compiled without '--enable-mlr' option, thus multiple linear regression model will not be computed.\n");
		for(i=0; i<ncoeff; i++)
			coeff[i] = 0.;
#endif //STARPU_MLR_MODEL
	}

	/* Preparing new output calibration file */
	if (calibrate==1 || calibrate==2)
	{
		if (old_lines > 0)
		{
			f = fopen(filepath, "a+");
			STARPU_ASSERT_MSG(f, "Could not save performance model into the file %s\n", filepath);
		}
		else
		{
			f = fopen(filepath, "w+");
			STARPU_ASSERT_MSG(f, "Could not save performance model into the file %s\n", filepath);
			fprintf(f, "Duration");
			for(j=0; j<nparameters; j++)
			{
				if(parameters_names != NULL && parameters_names[j]!= NULL)
					fprintf(f, ", %s", parameters_names[j]);
				else
					fprintf(f, ", P%u", j);
			}
		}
	}

	/* Writing parameters to calibration file */
	if (calibrate==1 || calibrate==2)
	{
		for(i=old_lines; i<n; i++)
		{
			fprintf(f, "\n%f", my[i]);
			for(j=0; j<nparameters; j++)
				fprintf(f, ", %f", mpar[i*nparameters+j]);
		}
		fclose(f);
	}

	/* Cleanup */
	free(mpar);
	free(my);

	return 0;
}
