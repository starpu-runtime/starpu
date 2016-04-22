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

// Headers needed for gsl
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_multifit.h>

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

static void dump_multiple_regression_list(double *mx, double *my, int start, unsigned ncoeff, unsigned nparameters, unsigned **combinations, struct starpu_perfmodel_history_list *list_history, FILE *f)
{
	struct starpu_perfmodel_history_list *ptr = list_history;
	int i = start;
	while (ptr)
	{
		my[i] = ptr->entry->duration;
		mx[i*ncoeff] = 1.;
		for(int j=0; j<ncoeff-1; j++)
		{
			mx[i*ncoeff+j+1] = 1.;
			for(int k=0; k < nparameters; k++)
			{
				mx[i*ncoeff+j+1] *= pow(ptr->entry->parameters[k],combinations[j][k]);
			}
		}
		ptr = ptr->next;
		i++;
	}

}

static void load_old_calibration(double *mx, double *my, unsigned ncoeff, FILE *f)
{
	char buffer[1024];
	char *record,*line;
	int i=0,j=0;

	line=fgets(buffer,sizeof(buffer),f);//skipping first line
	while((line=fgets(buffer,sizeof(buffer),f))!=NULL)
	{
		record = strtok(line,",");
		my[i] = atoi(record);
		record = strtok(NULL,",");
		j=0;
		while(record != NULL)
		{
			mx[i*ncoeff+j] = atoi(record) ;
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

//double dot(double *a, double *b, int len, int step)
//{
//	double r = 0;
//	while (len--) {
//		r += *a++ * *b;
//		b += step;
//	}
//	return r;
//}
//
//matrix mat_new(int h, int w)
//{
//	matrix r = malloc(sizeof(matrix_t) + sizeof(double) * w * h);
//	r->h = h, r->w = w;
//	r->x = (double*)(r + 1);
//	return r;
//}
//
//void mat_free(matrix a)
//{
//	/* Freeing needs to be fixed
//	free(a->x);
//	a->x=NULL;
//	free(a);
//	a=NULL;
//	*/
//}
//
//matrix mat_mul(matrix a, matrix b)
//{
//	matrix r;
//	double *p, *pa;
//	int i, j;
//	if (a->w != b->h) return 0;
//
//	r = mat_new(a->h, b->w);
//	p = r->x;
//	for (pa = a->x, i = 0; i < a->h; i++, pa += a->w)
//		for (j = 0; j < b->w; j++)
//			*p++ = dot(pa, b->x + j, a->w, b->w);
//	return r;
//}
//
//void mat_show(matrix a)
//{
//	int i, j;
//	double *p = a->x;
//	for (i = 0; i < a->h; i++, putchar('\n'))
//		for (j = 0; j < a->w; j++)
//			printf("\t%7.3f", *p++);
//	putchar('\n');
//}
//
//// Inspired from: https://rosettacode.org/wiki/Matrix_transposition#C
//matrix transpose(matrix src)
//{
//	int i, j;
//	matrix dst;
//	dst = mat_new(src->w, src->h);
//
//	for (i = 0; i < src->h; i++)
//	  for (j = 0; j < src->w; j++)
//		dst->x[j * dst->w + i] = src->x[i * src->w + j];
//
//	return dst;
//}
//
//// Inspired from: http://www.programming-techniques.com/2011/09/numerical-methods-inverse-of-nxn-matrix.html
//matrix mat_inv(matrix src)
//{
//	int n = src->h;
//	int n2=2*n;
//    int i,j, k;
//	double a;
//	matrix r, dst;
//	r = mat_new(n, n2);
//	dst = mat_new(n, n);
//
//	for (i = 0; i < n; i++)
//	  for (j = 0; j < n; j++)
//		r->x[i*n2+j] = src->x[i*n+j];
//
//	for(i = 0; i < n; i++){
//	  for(j = n; j < 2*n; j++){
//            if(i==(j-n))
//                r->x[i*n2+j] = 1.0;
//            else
//                r->x[i*n2+j] = 0.0;
//	  }
//	}
//
//	for(i = 0; i < n; i++){
//	  for(j = 0; j < n; j++){
//	    if(i!=j){
//            for(k = 0; k < 2*n; k++){
//                r->x[j*n2+k] -= (r->x[j*n2+i] / r->x[i*n2+i]) * r->x[i*n2+k];
//
//            }
//        }
//	  }
//	}
//
//	for(i = 0; i < n; i++){
//	  a = r->x[i*n2+i];
//	  for(j = 0; j < 2*n; j++){
//            r->x[i*n2+j] /= a;
//	  }
//	}
//
//	for (i = 0; i < n; i++)
//	  for (j = 0; j < n; j++)
//		dst->x[i*n+j] = r->x[i*n2+n+j];
//
//	return dst;
//}
//
///******************FUNCTION TO FIND THE DETERMINANT OF THE MATRIX************************/
//double detrm(matrix a, int k )
//{
//    double s = 1.;
//    double det = 0.;
//    matrix b;
//    int i, j, m, n, c;
//
//    if(k==1)
//    {
//       return a->x[0];
//    }
//    else
//    {
//    	b = mat_new(k-1, k-1);
//    	det = 0.;
//		for(c=0;c < k;c++)
//		{
//			m = 0;
//			n = 0;
//			for(i=0;i < k;i++)
//			  {
//				for(j=0;j < k;j++)
//				  {
//				b->x[i*k+j] = 0;
//				if(i!=0 && j!=c)
//				  {
//					b->x[m*k+n] = a->x[i*k+j];
//					if(n < (k-2))
//					  n++;
//					else
//					{
//					  n = 0;
//					  m++;
//					}
//				  }
//				  }
//			  }
//			det = det + s*(a->x[c]*detrm(b,k-1));
//			s = -1 * s;
//		}
//		mat_free(b);
//    }
//
//    return det;
//}
//double detrm2(matrix a, int m)
//{
//     if(m==1){
//        return a->x[0];
//     }
//     else if(m==2){
//             return a->x[0]*a->x[3]-a->x[1]*a->x[2];
//     }
//     else{
//             double det = 0;
//             int s=0,t=0,i,j,k;
//             for (i=0;i<m;i++){
//            	 matrix b;
//            	 b = mat_new(m-1, m-1);
//                 for(j=0;j<m;j++){
//                     if(j!=0){
//                         for(k=0;k<m;k++){
//                             if(k!=i){
//                                 b->x[s*(m-1)+t] = a->x[j*m+k];
//                                 t++;
//                             }
//                         }
//                     s++;
//                     }
//                    t = 0;
//                }
//                s = 0;
//                det = det+(int)pow(-1,i)*a->x[i]*detrm2(b, m-1);
//                mat_free(b);
//            }
//            return det;
//     }
//}
//
//// Inspired from:
//// http://thecodecracker.com/c-programming/inverse-of-a-matrix-in-c/comment-page-1/
//matrix mat_inv2(matrix num)
//{
//    int p, q, m, n, i, j;
//    int r = num->h;
//    double d;
//    matrix b, fac, inv;
//    b = mat_new(r, r);
//    fac = mat_new(r, r);
//    inv = mat_new(r, r);
//
//    for(q=0;q < r;q++)
//    {
//       for(p=0;p < r;p++)
//       {
//          m = 0;
//          n = 0;
//          for(i=0;i < r;i++)
//          {
//             for(j=0;j < r;j++)
//             {
//                b->x[i*r+j] = 0;
//                if(i!=q && j!=p)
//                {
//                   b->x[m*r+n] = num->x[i*r+j];
//                   if (n < (r-2))
//                      n++;
//                   else
//                   {
//                      n = 0;
//                      m++;
//                   }
//                }
//             }
//          }
//          d = detrm2(b, r-1);
//          fac->x[q*r+p] = (double) pow((-1),q+p) * d;
//      }
//    }
//
//    for(i= 0;i < r;i++)
//       for(j=0;j < r;j++)
//          b->x[i*r+j] = fac->x[j*r+i];
//
//
//    inv->x[i*r+j] = 0;
//    d = detrm2(num, r);
//
//    for(i=0;i < r;i++)
//       for(j=0;j < r;j++)
//          inv->x[i*r+j] = b->x[i*r+j] / d;
//
//    mat_free(b);
//    mat_free(fac);
//    return inv;
//}
//
//// Inspired from: http://reliawiki.org/index.php/Multiple_Linear_Regression_Analysis#Estimating_Regression_Models_Using_Least_Squares
//void multiple_reg_coeff(double *mx, double *my, int n, int k, double *coeff)
//{
//  	matrix X, Y;
//	X = mat_new(n,k);
//	X->x = mx;
//	Y = mat_new(n,1);
//	Y->x = my;
//
//	matrix mcoeff;
//	matrix Xt, mul1, mul2, inv;
//	Xt = transpose(X);
//	mul1 = mat_mul(Xt, X);
//	inv = mat_inv2(mul1);
//	mul2 = mat_mul(inv, Xt);
//	mcoeff = mat_mul(mul2, Y);
//	/*mcoeff = mat_mul(
//			mat_mul(
//				mat_inv2(
//					mat_mul(transpose(X), X)),
//				transpose(X)),
//			Y);
//*/
//	for(int i=0; i<k; i++)
//		coeff[i] = mcoeff->x[i];
//
//	mat_free(X);
//	mat_free(Y);
//	mat_free(mcoeff);
//}
//
//int test_multiple_regression()
//{
//	double da[] = {	1, 1,  1,   1,
//			2, 4,  8,  16,
//			3, 9, 27,  81,
//			4,16, 64, 256	};
//	double db[] = {     4.0,   -3.0,  4.0/3,
//			-13.0/3, 19.0/4, -7.0/3,
//			  3.0/2,   -2.0,  7.0/6,
//			 -1.0/6,  1.0/4, -1.0/6};
//
//	matrix_t a = { 4, 4, da }, b = { 4, 3, db };
//	matrix at;
//	at = transpose(&a);
//	matrix c = mat_mul(at, &b);
//
//	mat_show(&a), mat_show(at), mat_show(&b);
//	mat_show(c);
//	/* free(c) */
//	printf("\nInverse matrix:\n");
//
//
//	double dA[] = {	1, 2,  0,
//			-1, 1,  1,
//			1, 2, 3	};
//	matrix_t A = { 3, 3, dA };
//	mat_show(&A);
//	matrix Ainv;
//	Ainv = mat_inv(&A);
//	mat_show(Ainv);
//
//	// Multiple regression test: http://www.biddle.com/documents/bcg_comp_chapter4.pdf
//
//	double dX[] = {
//		  1, 153, 1, 153, 1, 151, 1, 151, 1, 151, 1, 152, 1, 155, 1, 152, 1, 152, 1, 151, 1, 154, 1, 53, 1, 154, 1, 156, 1, 154, 1, 153, 1, 153, 1, 153, 1, 156, 1, 156, 1, 153, 1, 153, 1, 154, 1, 151, 1, 154, 1, 153, 1, 151, 1, 153, 1, 155, 1, 151, 1, 152, 1, 158, 1, 156, 1, 154, 1, 151, 1, 158, 1, 152, 1, 154, 1, 150, 1, 153, 1, 151, 1, 153, 1, 152, 1, 153, 1, 156, 1, 152, 1, 154, 1, 153, 1, 152, 1, 154, 1, 152, 1, 152, 1, 150, 1, 152, 1, 154, 1, 151, 1, 151, 1, 150, 1, 155, 1, 157, 1, 154, 1, 153, 1, 152, 1, 151, 1, 151, 1, 156
//		};
//		 double dY[] = {
//		   669.803, 632.467, 633.580, 630.002, 611.578, 628.309, 612.352, 623.060, 621.759, 625.681, 642.941, 196.285, 636.249, 631.336, 648.567, 660.894, 679.917, 647.772, 636.877, 616.426, 632.436, 627.805, 623.983, 631.287, 631.871, 637.453, 652.078, 625.629, 635.099, 613.864, 632.843, 628.665, 630.218, 601.095, 602.018, 615.841, 633.704, 644.564, 630.676, 589.948, 606.154, 610.350, 597.742, 610.014, 614.922, 622.316, 601.779, 602.964, 614.403, 602.782, 608.804, 626.593, 625.762, 602.601, 606.464, 622.250, 592.306, 617.564, 633.847, 610.234, 614.683, 608.472, 615.067, 631.995, 854.267, 638.217
//		 };
//
//	int n = 66;
//	int k = 2;
//	matrix_t X= {n,k, dX};
//	matrix_t Y= {n,1, dY};
//	printf("\nMultiple regression:\n");
//	mat_show(&X);
//	mat_show(&Y);
//
//	matrix coeff;
//	coeff = mat_mul(
//			mat_mul(
//				mat_inv(
//					mat_mul(transpose(&X), &X)
//				        ),
//				transpose(&X)),
//			&Y);
//	mat_show(coeff);
//
//	double *results=NULL;
//	multiple_reg_coeff(dX, dY, n, k, results);
//	printf("\nFinal coefficients:\n");
//	for(int i=0; i<k; i++)
//	  printf("\tcoeff[%d]=%lf\n", i, results[i]);
//	return 0;
//
//}

// Inspired from: https://rosettacode.org/wiki/Multiple_regression#C
void gsl_multiple_reg_coeff(double *mx, double *my, long n, int k, double *coeff)
{
	gsl_matrix *X = gsl_matrix_calloc(n, k);
	gsl_vector *Y = gsl_vector_alloc(n);
	gsl_vector *beta = gsl_vector_alloc(k);

	for (int i = 0; i < n; i++) {
		gsl_vector_set(Y, i, my[i]);
		for (int j = 0; j < k; j++)
			gsl_matrix_set(X, i, j, mx[i*k+j]);
	}

	double chisq;
	gsl_matrix *cov = gsl_matrix_alloc(k, k);
	gsl_multifit_linear_workspace * wspc = gsl_multifit_linear_alloc(n, k);
	gsl_multifit_linear(X, Y, beta, cov, &chisq, wspc);

	for(int i=0; i<k; i++)
		coeff[i] = gsl_vector_get(beta, i);

	gsl_matrix_free(X);
	gsl_matrix_free(cov);
	gsl_vector_free(Y);
	gsl_vector_free(beta);
	gsl_multifit_linear_free(wspc);
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
	double *mx = (double *) malloc(ncoeff*n*sizeof(double));
	STARPU_ASSERT(mx);
	double *my = (double *) malloc(n*sizeof(double));
	STARPU_ASSERT(my);

	// Loading old calibration
	if (calibrate==1)
		load_old_calibration(mx, my, ncoeff, f);

	// Filling X and Y matrices with measured values
	dump_multiple_regression_list(mx, my, old_lines, ncoeff, nparameters, combinations, ptr, f);
	
	// Computing coefficients using multiple linear regression
	//multiple_reg_coeff(mx, my, n, ncoeff, coeff);
	gsl_multiple_reg_coeff(mx, my, n, ncoeff, coeff);

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
			for(int j=1; j<nparameters;j++)
				fprintf(f, ", %f", mx[i*nparameters+j]);			
		}
		fclose(f);
	}

	// Cleanup
	free(mx);
	free(my);

	return 0;
}
