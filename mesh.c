#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <cblas.h>

#define NTHETA	4
#define NTHICK	4

#define DIM	NTHETA*NTHICK

#define RMIN	(0.0f)
#define RMAX	(NTHICK*50.0f)

#define Pi	(3.141592f)


#define NODE_NUMBER(theta, thick)	((thick)+(theta)*NTHICK)
#define NODE_TO_THICK(n)		((n) % NTHICK)
#define NODE_TO_THETA(n)		((n) / NTHICK)

FILE *psfile;

typedef struct point_t {
	float x;
	float y;
} point;

typedef struct triangle_t {
	point *A;
	point *B;
	point *C;
} triangle;

point *pmesh;
float *A;
float *Ares;
float *B;
float *Xres;
float *Yres;
float *LU;
float *L;
float *U;

	/*
	 *   B              C
	 *	**********
	 *	*  0   * *
	 *	*    *   *
	 *	*  *   1 *
	 *	**********
	 *   A             D
	 */


#define X	0
#define Y	1
float diff_psi(unsigned theta_tr, unsigned thick_tr, unsigned side_tr,
		 unsigned theta_psi, unsigned thick_psi, unsigned xy)
{
	float xa,ya,xb,yb,xc,yc;
	float tmp;

	assert(theta_tr + 2 <= NTHETA);
	assert(thick_tr + 2 <= NTHICK);

	/* A */
	xa = pmesh[NODE_NUMBER(theta_tr, thick_tr)].x;
	ya = pmesh[NODE_NUMBER(theta_tr, thick_tr)].y;

	/* B */
	if (side_tr) {
		/* lower D is actually B here */
		xb = pmesh[NODE_NUMBER(theta_tr+1, thick_tr)].x;
		yb = pmesh[NODE_NUMBER(theta_tr+1, thick_tr)].y;
	} else {
		/* upper */
		xb = pmesh[NODE_NUMBER(theta_tr, thick_tr+1)].x;
		yb = pmesh[NODE_NUMBER(theta_tr, thick_tr+1)].y;
	}

	xc = pmesh[NODE_NUMBER(theta_tr+1, thick_tr+1)].x;
	yc = pmesh[NODE_NUMBER(theta_tr+1, thick_tr+1)].y;

	/* now look for the actual psi node */
	if (NODE_NUMBER(theta_tr, thick_tr) == NODE_NUMBER(theta_psi, thick_psi)) {
		/* A nothing to do */
//		printf("case 1\n");
	} else if (NODE_NUMBER(theta_tr+1, thick_tr+1) == NODE_NUMBER(theta_psi, thick_psi)) {
		/* psi matches C */
		/* swap A and C coordinates  */
		tmp = xa; xa = xc; xc = tmp;
		tmp = ya; ya = yc; yc = tmp;
//		printf("case 2\n");

	} else if
		(side_tr && (NODE_NUMBER(theta_tr+1, thick_tr) == NODE_NUMBER(theta_psi, thick_psi))) {
		/* psi is D (that was stored in C) XXX */
		tmp = xa; xa = xb; xb = tmp;
		tmp = ya; ya = yb; yb = tmp;
//		printf("case 3\n");
	} else if
		(!side_tr && (NODE_NUMBER(theta_tr, thick_tr+1) == NODE_NUMBER(theta_psi, thick_psi))) {
//		printf("case 4\n");
//		printf(" (%d,%d)%d <-> (%d,%d)%d\n", theta_tr, thick_tr, NODE_NUMBER(theta_tr, thick_tr), theta_psi, thick_psi, NODE_NUMBER(theta_psi, thick_psi));
		/* psi is C */
		tmp = xa; xa = xb; xb = tmp;
		tmp = ya; ya = yb; yb = tmp;
	} else {
		/* the psi node is not a node of the current triangle */
//		printf("case 5\n");
		return 0.0f;
	}

	/* now the triangle should have A as the psi node */
	float denom;
	float value;

	denom = (xa - xb)*(yc - ya) - (xc - xb)*(ya - yb);

	switch (xy) {
		case X:
		//	printf("X - A (%f,%f) B (%f,%f) C (%f,%f) denom %f\n", xa, ya, xb, yb, xc, yb, denom);
			value = (yc - yb)/denom;
			break;
		case Y:
		//	printf("Y - A (%f,%f) B (%f,%f) C (%f,%f) denom %f\n", xa, ya, xb, yb, xc, yb, denom);
			value = -(xc - xb)/denom;
			break;
		default:
			assert(0);
	}

	return value;
}

float diff_y_psi(unsigned theta_tr, unsigned thick_tr, unsigned side_tr,
		 unsigned theta_psi, unsigned thick_psi)
{
	return diff_psi(theta_tr, thick_tr, side_tr, theta_psi, thick_psi, Y);
}

float diff_x_psi(unsigned theta_tr, unsigned thick_tr, unsigned side_tr,
		 unsigned theta_psi, unsigned thick_psi)
{
	return diff_psi(theta_tr, thick_tr, side_tr, theta_psi, thick_psi, X);
}

float surface_triangle(unsigned theta_tr, unsigned thick_tr, unsigned side_tr)
{
	float surface;
	float tmp;

	float xi, xj, xk, yi, yj, yk;

	assert(theta_tr + 2 <= NTHETA);
	assert(thick_tr + 2 <= NTHICK);

	xi = pmesh[NODE_NUMBER(theta_tr, thick_tr)].x;
	yi = pmesh[NODE_NUMBER(theta_tr, thick_tr)].y;

	xj = pmesh[NODE_NUMBER(theta_tr+1, thick_tr+1)].x;
	yj = pmesh[NODE_NUMBER(theta_tr+1, thick_tr+1)].y;

	if (side_tr) {
		/* lower */
		xk = pmesh[NODE_NUMBER(theta_tr+1, thick_tr)].x;
		yk = pmesh[NODE_NUMBER(theta_tr+1, thick_tr)].y;
	} else {
		xk = pmesh[NODE_NUMBER(theta_tr, thick_tr+1)].x;
		yk = pmesh[NODE_NUMBER(theta_tr, thick_tr+1)].y;
	}

	tmp = (xi - xj)*(yk -yj) - (xk - xj)*(yi -yj);

	surface = 0.5*fabs(tmp);
	printf("surface %f theta_tr %d thick_tr %d side %d xi %f yi %f xj %f yj %f xk %f yk %f\n", surface, theta_tr, thick_tr, side_tr, xi, yi, xj, yj, xk, yk);

	return surface;
}

float integral_triangle(int theta_tr, int thick_tr, unsigned side_tr,
			unsigned theta_i, unsigned thick_i, unsigned theta_j, unsigned thick_j)
{
	float surface;
	float value;

	float dxi, dxj, dyi, dyj;

	if (theta_tr < 0) return 0.0f;
	if (theta_tr > NTHETA-2) return 0.0f;

	if (thick_tr < 0) return 0.0f;
	if (thick_tr > NTHICK-2) return 0.0f;

	dxi = diff_x_psi(theta_tr, thick_tr, side_tr, theta_i, thick_i);
	dyi = diff_y_psi(theta_tr, thick_tr, side_tr, theta_i, thick_i);
	dxj = diff_x_psi(theta_tr, thick_tr, side_tr, theta_j, thick_j);
	dyj = diff_y_psi(theta_tr, thick_tr, side_tr, theta_j, thick_j);

//	if ((theta_i == theta_j) && (thick_i == thick_j) && (thick_i == NTHICK -1) && (theta_i == 0) )
		printf("TOTO | (i %d %d) dxi %f dyi %f  (j %d %d), dxj %f  dyj %f (side %d)\n", theta_i, thick_i, dxi, dyi, theta_j, thick_j, dxj, dyj, side_tr);

	surface = surface_triangle(theta_tr, thick_tr, side_tr);

	value = (dxi*dxj + dyi*dyj)*surface;

	return value;
}

float integrale_sum(unsigned theta_i, unsigned thick_i, unsigned theta_j, unsigned thick_j)
{
	float integral = 0.0f;

	unsigned debug_trace = 0;
//	if ((theta_i == theta_j) && (thick_i == thick_j) && (thick_i == NTHICK -1) && (theta_i == 0) )
		debug_trace = 1;

	if (debug_trace) {
		printf("integrale => += %f\n", integral);
	}
	integral += integral_triangle(theta_i - 1, thick_i - 1, 1, theta_i, thick_i, theta_j, thick_j);
	if (debug_trace) {
		printf("integrale => += %f\n", integral);
	}
	integral += integral_triangle(theta_i - 1, thick_i - 1, 0, theta_i, thick_i, theta_j, thick_j);
	if (debug_trace) {
		printf("integrale => += %f\n", integral);
	}
	integral += integral_triangle(theta_i - 1, thick_i, 1, theta_i, thick_i, theta_j, thick_j);
	if (debug_trace) {
		printf("integrale => += %f\n", integral);
	}
	integral += integral_triangle(theta_i, thick_i, 0, theta_i, thick_i, theta_j, thick_j);
	if (debug_trace) {
		printf("integrale => += %f\n", integral);
	}
	integral += integral_triangle(theta_i, thick_i, 1, theta_i, thick_i, theta_j, thick_j);
	if (debug_trace) {
		printf("integrale => += %f\n", integral);
	}
	integral += integral_triangle(theta_i, thick_i - 1, 0, theta_i, thick_i, theta_j, thick_j);
	if (debug_trace) {
		printf("integrale => += %f\n", integral);
	}

	return integral;
}

void compute_A_value(unsigned i, unsigned j)
{
	float value = 0.0f;

	unsigned thick_i, thick_j;
	unsigned theta_i, theta_j;

	/* add all contributions from all connex triangles  */
	thick_i = NODE_TO_THICK(i);
	thick_j = NODE_TO_THICK(j);

	theta_i = NODE_TO_THETA(i);
	theta_j = NODE_TO_THETA(j);

	/* Compute the Sum of all the integral over all triangles */
	if ( (abs(thick_i - thick_j) <= 1) && (abs(theta_i - theta_j) <= 1) )
	{
		if ( (theta_j == theta_i -1) && (thick_j == thick_i +1))
			goto done;

		if ( (theta_j == theta_i + 1) && (thick_j == thick_i  - 1))
			goto done;

		if (i==j && i == NTHICK-1) {
			printf("PROUT\n");
		}
		/* this may not be a null entry */
		value += integrale_sum(theta_i, thick_i, theta_j, thick_j);
	}

done:

	if (i == j) printf("diag %d -> %f\n", i, value);

	A[i+j*DIM] = value;
}

void postscript_gen()
{
	psfile = fopen("output.ps", "w+");

	int offx, offy;
	unsigned theta, thick;

	offx = RMAX+50;
	offy = 100;

	for (theta = 0; theta < NTHETA-1; theta++)
	{
		for (thick = 0; thick < NTHICK-1; thick++)
		{
			fprintf(psfile, "newpath\n");
			fprintf(psfile, "%d %d moveto\n", (int)pmesh[NODE_NUMBER(theta, thick)].x + offx,
							  (int)pmesh[NODE_NUMBER(theta, thick)].y+ offy);
			fprintf(psfile, "%d %d lineto\n", (int)pmesh[NODE_NUMBER(theta+1, thick)].x + offx,
							  (int)pmesh[NODE_NUMBER(theta+1, thick)].y+ offy);
			fprintf(psfile, "%d %d lineto\n", (int)pmesh[NODE_NUMBER(theta+1, thick+1)].x + offx,
							  (int)pmesh[NODE_NUMBER(theta+1, thick+1)].y+ offy);
			fprintf(psfile, "closepath\n");
			fprintf(psfile, "stroke\n");

			fprintf(psfile, "newpath\n");
			fprintf(psfile, "%d %d moveto\n", (int)pmesh[NODE_NUMBER(theta, thick)].x + offx,
							  (int)pmesh[NODE_NUMBER(theta, thick)].y+ offy);
			fprintf(psfile, "%d %d lineto\n", (int)pmesh[NODE_NUMBER(theta, thick+1)].x + offx,
							  (int)pmesh[NODE_NUMBER(theta, thick+1)].y+ offy);
			fprintf(psfile, "%d %d lineto\n", (int)pmesh[NODE_NUMBER(theta+1, thick+1)].x + offx,
							  (int)pmesh[NODE_NUMBER(theta+1, thick+1)].y+ offy);
			fprintf(psfile, "closepath\n");

			fprintf(psfile, "stroke\n");
		}
	}

	fclose(psfile);

}

void dummy_lu_facto()
{
	int k, i, j;

	memcpy(LU, A, DIM*DIM*sizeof(float));

	float *subLU;
	subLU = malloc(DIM*DIM*sizeof(float));
	memcpy(subLU, A, DIM*DIM*sizeof(float));

	

	for (k = 0; k < DIM; k++) {

			printf("pivot => k %d %f \n", k, LU[k+k*DIM]);
		for (i = k+1; i < DIM ; i++)
		{
		        assert(LU[k+k*DIM] != 0.0);
	//		printf("LU[%d+%d*DIM] = %f -> %f \n", i, k, LU[i+k*DIM], LU[i+k*DIM] / LU[k+k*DIM]);
			LU[i+k*DIM] = LU[i+k*DIM] / LU[k+k*DIM];
		}

		for (j = k+1; j < DIM; j++)
		{
			for (i = k+1; i < DIM; i++)
			{
			        LU[i+j*DIM] -= LU[i+k*DIM]*LU[k+j*DIM];
			}
		}

	}

//	printf("LU\n");
//	for (j = 0; j < DIM; j++)
//	{
//		for (i=0; i < DIM; i++)
//		{
//			printf("%f\t", LU[i+j*DIM]);
//		}
//		printf("\n");
//	}
//

	L = malloc(DIM*DIM*sizeof(float));
	U = malloc(DIM*DIM*sizeof(float));

	for (j = 0; j < DIM*DIM; j++) {
		L[j] = 0.0f;
		U[j] = 0.0f;
	}

        for (j = 0; j < DIM; j++)
        {
                for (i = 0; i < j; i++)
                {
                        L[i+j*DIM] = LU[i+j*DIM];
                }

                /* diag i = j */
                L[j+j*DIM] = LU[j+j*DIM];
                U[j+j*DIM] = 1.0f;

                for (i = j+1; i < DIM; i++)
                {
                        U[i+j*DIM] = LU[i+j*DIM];
                }
        }

	printf("A\n");
	for (j = 0; j < DIM; j++)
	{
		printf("[");
		for (i=0; i < DIM; i++)
		{
			printf("%.2f\t", A[i+j*DIM]);
		}
		printf("[\n");
	}


	printf("L\n");
	for (j = 0; j < DIM; j++)
	{
		for (i=0; i < DIM; i++)
		{
			if (fabs(L[i+j*DIM]) < 0.000000001f) {
				printf(".\t");
			} else {
				printf("%.2f\t", L[i+j*DIM]);
			}
		}
		printf("\n");
	}

	printf("U\n");
	for (j = 0; j < DIM; j++)
	{
		for (i=0; i < DIM; i++)
		{
			if (fabs(U[i+j*DIM]) < 0.0001f) {
				printf(".\t");
			}
			else {
				printf("%.2f\t", U[i+j*DIM]);
			}
		}
		printf("\n");
	}

	/* solve the actual problem LU X = B */
        /* solve LX' = Y with X' = UX */
        /* solve UX = X' */
	Xres = malloc(DIM*sizeof(float));
	memcpy(Xres, B, DIM*sizeof(float));

	Ares = malloc(DIM*DIM*sizeof(float));
	cblas_strsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
			DIM, L, DIM, Xres, 1);


	printf("solution : \n");
	for (i = 0; i< DIM; i++)
	{
		printf("  %d\t->\t%.2f\n", i, Xres[i]);
	}

        cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasUnit,
                        DIM, U, DIM, Xres, 1);

	printf("solution : \n");
	for (i = 0; i< DIM; i++)
	{
		printf("  %d\t->\t%2.f\n", i, Xres[i]);
	}

	Yres = malloc(DIM*sizeof(float));

	for (j = 0; j<DIM; j++)
	{
		for (i = 0; i<DIM; i++)
		{
			Ares[i+j*DIM] = 0.0f;
			for (k = 0; k < DIM; k++){
				Ares[i+j*DIM] += L[k+j*DIM]*U[i+k*DIM];
			}
		}
	}


	for (j = 0; j<DIM; j++)
	{
		Yres[i] = 0.0f;
		for (i = 0; i < DIM; i++){
			Yres[j] += Xres[i]*A[i+j*DIM];
		}
	}

	printf("A Xres = Yres\n");
	for (j = 0; j < DIM; j++)
	{
		for (i=0; i < DIM; i++)
		{
//			fprintf(stderr, "%.1f(%.1f)\t", A[i+j*DIM], Ares[i+j*DIM]);
			if (fabs(Ares[i+j*DIM]) < 0.0001f) {
				fprintf(stdout, ".\t");
			}
			else {
				fprintf(stdout, "%.2f\t", Ares[i+j*DIM]);
			}
		}
		fprintf(stdout, "\t|\t%f\t|\t%f\n", Xres[j], Yres[j]);
	}

	float dx, dy;
	dx = diff_psi(0, NTHICK-2, 0, 0, NTHICK-1,X );
	dy = diff_psi(0, NTHICK-2, 0, 0, NTHICK-1,Y );

	float inte1,inte2;

	printf("EUBGUG\n");
	inte1 = integral_triangle(0, 0, 1, 1, 0, 0, 0);

	printf("EUBGUG\n");
	inte2 =  integrale_sum(0, 0, 1, 0);
	//Ares[3];//integral_triangle(0, 0, 0, 1, 1, 0, 0);

	// case 1
//	inte1 = integral_triangle(0, 0, 0, 1, 1, 0, 1);
//	inte1 += integral_triangle(0, 0, 1, 1, 1, 0, 1);
//	inte1 += integral_triangle(0, 1, 0, 1, 1, 0, 1);
//	inte1 += integral_triangle(0, 1, 1, 1, 1, 0, 1);
//
//	inte2 = integral_triangle(1, 0, 0, 1, 1, 2, 1);
//	inte2 += integral_triangle(1, 0, 1, 1, 1, 2, 1);
//	inte2 += integral_triangle(1, 1, 0, 1, 1, 2, 1);
//	inte2 += integral_triangle(1, 1, 1, 1, 1, 2, 1);

	printf("**********************\n");
	printf("**********************\n");
	printf("A[%d,%d] = %f\n",NTHICK-1, NTHICK-1, A[(NTHICK-1)*DIM + (NTHICK-1)]);
	printf("**********************\n");
	printf("dx = %f dy = %f\n", dx,dy);
	printf("inte1 %f inte2 %f \n", inte1, inte2);
	printf("**********************\n");
	printf("**********************\n");

//	/* first solve LX' = B */
//	float v;
//	float *Xp = malloc(DIM*sizeof(float));
//
//
//	Xp[0] = B[0];
//	for (i = 1; i < DIM; i++)
//	{
//		v = B[i];
//		for (j = 0; j < i-1; j++) {
//			v -= L[i+j*DIM]*Xp[j];
//		}
//		Xp[i] = v;
//	}
//
//
//	/* then solve UX = X' */
//	for (i=DIM-1; i>=0; i--) {
//		v = Xp[i];
//		for (j=i+1; j<DIM; j++) {
//			v = v - U[i+j*DIM]*Xres[j];
//		}
//		Xres[i] = v/U[i+DIM*i];
//	}
//

}

int main(int argc, char **argv)
{
	unsigned theta, thick;


	pmesh = malloc(DIM*sizeof(point));

	/* the stiffness matrix : boundary conditions are known */
	A = malloc(DIM*DIM*sizeof(float));
	B = malloc(DIM*sizeof(float));
	LU = malloc(DIM*DIM*sizeof(float));

	/* first build the mesh by determining all points positions */
	for (theta = 0; theta < NTHETA; theta++)
	{
//		float angle;

//		angle = (NTHETA - 1 - theta) * Pi/(NTHETA-1);

		for (thick = 0; thick < NTHICK; thick++)
		{
//			float r;

//			r = thick * (RMAX - RMIN)/(NTHICK - 1) + RMIN;

//			pmesh[NODE_NUMBER(theta,thick)].x = r*cosf(angle);
//			pmesh[NODE_NUMBER(theta,thick)].y = r*sinf(angle);

			pmesh[NODE_NUMBER(theta,thick)].x = -100 + RMIN+((RMAX-RMIN)*theta)/(NTHETA - 1);//-RMIN + (2*(RMAX - RMIN)*theta)/(NTHETA - 1);
			pmesh[NODE_NUMBER(theta,thick)].y = RMIN+((RMAX-RMIN)*thick)/(NTHICK - 1);//-RMIN + (2*(RMAX - RMIN)*theta)/(NTHETA - 1);
		}
	}

	postscript_gen();


	/* then build the stiffness matrix A */
	unsigned i,j;
	for (j = 0 ; j < DIM ; j++)
	{
		for (i = 0; i < DIM ; i++)
		{
			compute_A_value(i, j);
		}
	}

	for (i = 0; i < DIM; i++)
	{
		B[i] = 100.0f;
	}

	dummy_lu_facto();

	return 0;
}
