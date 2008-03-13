#include "heat.h"

#define NTHETA	180
#define NTHICK	36

#define MIN(a,b)	((a)<(b)?(a):(b))
#define MAX(a,b)	((a)<(b)?(b):(a))

#define DIM	NTHETA*NTHICK

#define RMIN	(150.0f)
#define RMAX	(200.0f)

#define Pi	(3.141592f)


#define NODE_NUMBER(theta, thick)	((thick)+(theta)*NTHICK)
#define NODE_TO_THICK(n)		((n) % NTHICK)
#define NODE_TO_THETA(n)		((n) / NTHICK)

#define UNLIKELY(expr)		(__builtin_expect(!!(expr),0))
#define LIKELY(expr)		(__builtin_expect(!!(expr),1))


typedef struct point_t {
	float x;
	float y;
} point;

//typedef struct triangle_t {
//	point *A;
//	point *B;
//	point *C;
//} triangle;


float minval, maxval;
float *result;
int *RefArray;
point *pmesh;
float *A;
float *subA;
float *Ares;
float *subAres;
float *B;
float *subB;
float *Xres;
float *subXres;
float *Yres;
float *subYres;
float *LU;
float *subLU;
float *L;
float *subL;
float *U;
float *subU;

unsigned printmesh =0;

int argc_;
char **argv_;
	/*
	 *   B              C
	 *	**********
	 *	*  0   * *
	 *	*    *   *
	 *	*  *   1 *
	 *	**********
	 *   A             D
	 */


/*
 * Just some dummy OpenGL code to display our results 
 *
 */

static void generate_graph()
{
	unsigned theta, thick;

	for (theta = 0; theta < NTHETA-1; theta++)
	{
		for (thick = 0; thick < NTHICK-1; thick++)
		{
			unsigned nodeA = NODE_NUMBER(theta, thick);
			unsigned nodeB = NODE_NUMBER(theta, thick+1);
			unsigned nodeC = NODE_NUMBER(theta+1, thick+1);
			unsigned nodeD = NODE_NUMBER(theta+1, thick);

			float colorA_R, colorB_R, colorC_R, colorD_R;
			float colorA_G, colorB_G, colorC_G, colorD_G;
			float colorA_B, colorB_B, colorC_B, colorD_B;

			if (maxval == minval) {
				colorA_R = 1.0f; colorA_G = 1.0f; colorA_B = 1.0f;
				colorB_R = 1.0f; colorB_G = 1.0f; colorB_B = 1.0f;
				colorC_R = 1.0f; colorC_G = 1.0f; colorC_B = 1.0f;
				colorD_R = 1.0f; colorD_G = 1.0f; colorD_B = 1.0f;
			}
			else {
				float amplitude = maxval - minval;

				float coeffA, coeffB, coeffC, coeffD;

				coeffA = (result[nodeA] - minval)/amplitude;
				coeffB = (result[nodeB] - minval)/amplitude;
				coeffC = (result[nodeC] - minval)/amplitude;
				coeffD = (result[nodeD] - minval)/amplitude;

				colorA_R = coeffA>0.5f?1.0f:(2.0*coeffA)*1.0f; 
				colorB_R = coeffB>0.5f?1.0f:(2.0*coeffB)*1.0f; 
				colorC_R = coeffC>0.5f?1.0f:(2.0*coeffC)*1.0f; 
				colorD_R = coeffD>0.5f?1.0f:(2.0*coeffD)*1.0f; 

				colorA_B = 0.0f; 
				colorB_B = 0.0f; 
				colorC_B = 0.0f; 
				colorD_B = 0.0f; 

				colorA_G = coeffA<0.5f?1.0f:2.0*(1 - coeffA)*1.0f;
				colorB_G = coeffB<0.5f?1.0f:2.0*(1 - coeffB)*1.0f;
				colorC_G = coeffC<0.5f?1.0f:2.0*(1 - coeffC)*1.0f;
				colorD_G = coeffD<0.5f?1.0f:2.0*(1 - coeffD)*1.0f;
			}

			if (printmesh) {
   			glColor3f (0.0f, 0.0f, 0.0f);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glLineWidth(3.0f);
			glBegin(GL_POLYGON);
				glVertex3f(pmesh[nodeA].x, pmesh[nodeA].y, 2.0f);
				glVertex3f(pmesh[nodeD].x, pmesh[nodeD].y, 2.0f);
				glVertex3f(pmesh[nodeC].x, pmesh[nodeC].y, 2.0f);
				glVertex3f(pmesh[nodeA].x, pmesh[nodeA].y, 2.0f);
			glEnd();

			glBegin(GL_POLYGON);
				glVertex3f(pmesh[nodeA].x, pmesh[nodeA].y, 1.0f);
				glVertex3f(pmesh[nodeC].x, pmesh[nodeC].y, 1.0f);
				glVertex3f(pmesh[nodeB].x, pmesh[nodeB].y, 1.0f);
				glVertex3f(pmesh[nodeA].x, pmesh[nodeA].y, 1.0f);
			glEnd();
			}

			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glBegin(GL_POLYGON);
   				glColor3f (colorA_R, colorA_G, colorA_B);
				glVertex3f(pmesh[nodeA].x, pmesh[nodeA].y, 0.0f);
   				glColor3f (colorD_R, colorD_G, colorD_B);
				glVertex3f(pmesh[nodeD].x, pmesh[nodeD].y, 0.0f);
   				glColor3f (colorC_R, colorC_G, colorC_B);
				glVertex3f(pmesh[nodeC].x, pmesh[nodeC].y, 0.0f);
			glEnd();

			glBegin(GL_POLYGON);
   				glColor3f (colorA_R, colorA_G, colorA_B);
				glVertex3f(pmesh[nodeA].x, pmesh[nodeA].y, 0.0f);
   				glColor3f (colorC_R, colorC_G, colorC_B);
				glVertex3f(pmesh[nodeC].x, pmesh[nodeC].y, 0.0f);
   				glColor3f (colorB_R, colorB_G, colorB_B);
				glVertex3f(pmesh[nodeB].x, pmesh[nodeB].y, 0.0f);
			glEnd();
		}
	}
}

static void display(void)
{
   glClear (GL_COLOR_BUFFER_BIT);
   glLoadIdentity ();             /* clear the matrix */
   gluLookAt (0.0, 0.0, 15.0f, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
   float factor = 10.0/(RMIN+RMAX);
   glScalef (factor, factor, factor);      /* modeling transformation */
   glRotatef(-0,0.0,0.0,0.0);
   generate_graph();
   glFlush ();
}


static void pressKey(unsigned char key, int x, int y)
{
	switch (key) {
		case 'q':
			exit(0);
		default:
			printmesh = !printmesh;
			display();
			break;
	}
}



static void reshape (int w, int h)
{
   glViewport (0, 0, (GLsizei) w, (GLsizei) h);
   glMatrixMode (GL_PROJECTION);
   glLoadIdentity ();
   glFrustum (-1.0, 1.0, -1.0, 1.0, 1.5, 20.0);
   glMatrixMode (GL_MODELVIEW);
}



static void opengl_render(void)
{
	unsigned i;
	printf("OpenGL rendering ... \n");

	minval = 100000000.0f;
	maxval = -10000000.0f;

	for (i = 0; i < DIM; i++)
	{

		/* find min */
		minval = MIN(result[i], minval);

		/* find max */
		maxval = MAX(result[i], maxval);
	}



	glutInit(&argc_, argv_);
	glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize (800, 800);
	glutInitWindowPosition (100, 100);
	glutCreateWindow ("Temperature");

	/* init */
	glClearColor (0.0, 0.0, 0.0, 0.0);
	glShadeModel (GL_MODELVIEW);

	glutKeyboardFunc(pressKey);
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMainLoop();
}

/*
 *
 * The Finite element method code 
 *
 */

#define X	0
#define Y	1
static inline float diff_psi(unsigned theta_tr, unsigned thick_tr, unsigned side_tr,
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
	} else if (NODE_NUMBER(theta_tr+1, thick_tr+1) == NODE_NUMBER(theta_psi, thick_psi)) {
		/* psi matches C */
		/* swap A and C coordinates  */
		tmp = xa; xa = xc; xc = tmp;
		tmp = ya; ya = yc; yc = tmp;
	} else if
		(side_tr && (NODE_NUMBER(theta_tr+1, thick_tr) == NODE_NUMBER(theta_psi, thick_psi))) {
		/* psi is D (that was stored in C) XXX */
		tmp = xa; xa = xb; xb = tmp;
		tmp = ya; ya = yb; yb = tmp;
	} else if
		(!side_tr && (NODE_NUMBER(theta_tr, thick_tr+1) == NODE_NUMBER(theta_psi, thick_psi))) {
		/* psi is C */
		tmp = xa; xa = xb; xb = tmp;
		tmp = ya; ya = yb; yb = tmp;
	} else {
		/* the psi node is not a node of the current triangle */
		return 0.0f;
	}

	/* now the triangle should have A as the psi node */
	float denom;
	float value;

	denom = (xa - xb)*(yc - ya) - (xc - xb)*(ya - yb);

	switch (xy) {
		case X:
			value = (yc - yb)/denom;
			break;
		case Y:
			value = -(xc - xb)/denom;
			break;
		default:
			assert(0);
	}

	return value;
}

static inline float diff_y_psi(unsigned theta_tr, unsigned thick_tr, unsigned side_tr,
		 unsigned theta_psi, unsigned thick_psi)
{
	return diff_psi(theta_tr, thick_tr, side_tr, theta_psi, thick_psi, Y);
}

static inline float diff_x_psi(unsigned theta_tr, unsigned thick_tr, unsigned side_tr,
		 unsigned theta_psi, unsigned thick_psi)
{
	return diff_psi(theta_tr, thick_tr, side_tr, theta_psi, thick_psi, X);
}

static inline float surface_triangle(unsigned theta_tr, unsigned thick_tr, unsigned side_tr)
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

	return surface;
}

static inline float integral_triangle(int theta_tr, int thick_tr, unsigned side_tr,
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

	surface = surface_triangle(theta_tr, thick_tr, side_tr);

	value = (dxi*dxj + dyi*dyj)*surface;

	return value;
}

static inline float integrale_sum(unsigned theta_i, unsigned thick_i, unsigned theta_j, unsigned thick_j)
{
	float integral = 0.0f;

	integral += integral_triangle(theta_i - 1, thick_i - 1, 1, theta_i, thick_i, theta_j, thick_j);
	integral += integral_triangle(theta_i - 1, thick_i - 1, 0, theta_i, thick_i, theta_j, thick_j);
	integral += integral_triangle(theta_i - 1, thick_i, 1, theta_i, thick_i, theta_j, thick_j);
	integral += integral_triangle(theta_i, thick_i, 0, theta_i, thick_i, theta_j, thick_j);
	integral += integral_triangle(theta_i, thick_i, 1, theta_i, thick_i, theta_j, thick_j);
	integral += integral_triangle(theta_i, thick_i - 1, 0, theta_i, thick_i, theta_j, thick_j);

	return integral;
}

#define TRANSLATE(k)	(RefArray[(k)])

static void compute_A_value(unsigned i, unsigned j)
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

		/* this may not be a null entry */
		value += integrale_sum(theta_i, thick_i, theta_j, thick_j);
	}

done:
	A[i+j*DIM] = value;
}

#ifdef USE_POSTSCRIPT
static void postscript_gen()
{
	FILE *psfile;
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
#endif

static void dummy_lu_facto(int subsize)
{
	int k, i, j;

//	memcpy(LU, A, DIM*DIM*sizeof(float));

	memcpy(subLU, subA, DIM*DIM*sizeof(float));


	/* LU factorisation of the stifness matrix */
	printf("LU Factorization\n");

	for (k = 0; k < subsize; k++) {
		for (i = k+1; LIKELY(i < subsize) ; i++)
		{
		        assert(subLU[k+k*subsize] != 0.0);
			subLU[i+k*subsize] = subLU[i+k*subsize] / subLU[k+k*subsize];
		}

		for (j = k+1; j < subsize; j++)
		{
			for (i = k+1; LIKELY(i < subsize); i++)
			{
			        subLU[i+j*subsize] -= subLU[i+k*subsize]*subLU[k+j*subsize];
			}
		}

	}

//
//	for (k = 0; k < DIM; k++) {
//
//		for (i = k+1; i < DIM ; i++)
//		{
//		        assert(LU[k+k*DIM] != 0.0);
//			LU[i+k*DIM] = LU[i+k*DIM] / LU[k+k*DIM];
//		}
//
//		for (j = k+1; j < DIM; j++)
//		{
//			for (i = k+1; i < DIM; i++)
//			{
//			        LU[i+j*DIM] -= LU[i+k*DIM]*LU[k+j*DIM];
//			}
//		}
//
//	}

}

static void solve_system(int subsize)
{
	unsigned i,j;

	//L = malloc(DIM*DIM*sizeof(float));
	//U = malloc(DIM*DIM*sizeof(float));

	for (j = 0; j < subsize*subsize; j++) {
		subL[j] = 0.0f;
		subU[j] = 0.0f;
	}

        for (j = 0; j < subsize; j++)
        {
                for (i = 0; i < j; i++)
                {
                        subL[i+j*subsize] = subLU[i+j*subsize];
                }

                /* diag i = j */
                subL[j+j*subsize] = subLU[j+j*subsize];
                subU[j+j*subsize] = 1.0f;

                for (i = j+1; i < subsize; i++)
                {
                        subU[i+j*subsize] = subLU[i+j*subsize];
                }
        }


//	for (j = 0; j < DIM*DIM; j++) {
//		L[j] = 0.0f;
//		U[j] = 0.0f;
//	}
//
//        for (j = 0; j < DIM; j++)
//        {
//                for (i = 0; i < j; i++)
//                {
//                        L[i+j*DIM] = LU[i+j*DIM];
//                }
//
//                /* diag i = j */
//                L[j+j*DIM] = LU[j+j*DIM];
//                U[j+j*DIM] = 1.0f;
//
//                for (i = j+1; i < DIM; i++)
//                {
//                        U[i+j*DIM] = LU[i+j*DIM];
//                }
//        }

	/* solve the actual problem LU X = B */
        /* solve LX' = Y with X' = UX */
        /* solve UX = X' */
//	Xres = malloc(DIM*sizeof(float));
	subXres = malloc(subsize*sizeof(float));
//	memcpy(Xres, B, DIM*sizeof(float));
	memcpy(subXres, subB, subsize*sizeof(float));

//	Ares = malloc(DIM*DIM*sizeof(float));
	subAres = malloc(subsize*subsize*sizeof(float));

//	cblas_strsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
//			DIM, L, DIM, Xres, 1);

	printf("Solving the problem ...\n");
	cblas_strsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
			subsize, subL, subsize, subXres, 1);


 //       cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasUnit,
 //                       DIM, U, DIM, Xres, 1);
        cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasUnit,
                        subsize, subU, subsize, subXres, 1);

//	Yres = malloc(DIM*sizeof(float));
	subYres = malloc(subsize*sizeof(float));

//	for (j = 0; j<DIM; j++)
//	{
//		for (i = 0; i<DIM; i++)
//		{
//			Ares[i+j*DIM] = 0.0f;
//			for (k = 0; k < DIM; k++){
//				Ares[i+j*DIM] += L[k+j*DIM]*U[i+k*DIM];
//			}
//		}
//	}
//
//
//	for (j = 0; j<DIM; j++)
//	{
//		Yres[i] = 0.0f;
//		for (i = 0; i < DIM; i++){
//			Yres[j] += Xres[i]*A[i+j*DIM];
//		}
//	}

	//for (j = 0; j<subsize; j++)
	//{
	//	for (i = 0; i<subsize; i++)
	//	{
	//		float val = 0.0f;
	//		for (k = 0; k < subsize; k++){
	//			val += subL[k+j*subsize]*subU[i+k*subsize];
	//		}
	//		subAres[i+j*subsize] = val;
	//	}
	//}

	//for (j = 0; j<subsize; j++)
	//{
	//	subYres[j] = 0.0f;
	//	for (i = 0; i < subsize; i++){
	//		subYres[j] += subXres[i]*subA[i+j*subsize];
	//	}
	//}



//	printf("A Xres = Yres\n");
//	for (j = 0; j < DIM; j++)
//	{
//		for (i=0; i < DIM; i++)
//		{
////			fprintf(stderr, "%.1f(%.1f)\t", A[i+j*DIM], Ares[i+j*DIM]);
//			if (fabs(Ares[i+j*DIM]) < 0.0001f) {
//				fprintf(stdout, ".\t");
//			}
//			else {
//				fprintf(stdout, "%.2f\t", Ares[i+j*DIM]);
//			}
//		}
//		fprintf(stdout, "\t|\t%f\t|\t%f\n", Xres[j], Yres[j]);
//	}
//
//	printf("SUBA SUBXres = SUBYres\n");
//	for (j = 0; j < subsize; j++)
//	{
//		fprintf(stdout, "%d ->\t", TRANSLATE(j));
//		for (i=0; i < subsize; i++)
//		{
////			fprintf(stderr, "%.1f(%.1f)\t", A[i+j*subsize], Ares[i+j*subsize]);
//			if (fabs(subAres[i+j*subsize]) < 0.0001f) {
//				fprintf(stdout, ".\t");
//			}
//			else {
//				fprintf(stdout, "%.2f\t", subAres[i+j*subsize]);
//			}
//		}
//		fprintf(stdout, "\t|\t%f\t|\t%f(%2.f)\n", subXres[j], subYres[j], subB[j]);
//	}
//
	result = malloc(DIM*sizeof(float));

	/* now display back the ACTUAL result */
	for (i = 0; i < subsize; i++)
	{
		/* we computed those value ! */
		result[TRANSLATE(i)] = subXres[i];
	}
	for (i = subsize; i < DIM; i++)
	{
		/* those were the boundaries */
		result[TRANSLATE(i)] = B[TRANSLATE(i)];
	}
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
	argc_ = argc;
	argv_ = argv;

#ifdef USE_MARCEL
	marcel_init(&argc, argv);
#endif


	unsigned theta, thick;

	int newsize;

	pmesh = malloc(DIM*sizeof(point));

	/* the stiffness matrix : boundary conditions are known */
	A = malloc(DIM*DIM*sizeof(float));
	B = malloc(DIM*sizeof(float));
	//LU = malloc(DIM*DIM*sizeof(float));

	/* first build the mesh by determining all points positions */
	for (theta = 0; theta < NTHETA; theta++)
	{
		float angle;

		angle = (NTHETA - 1 - theta) * Pi/(NTHETA-1);

		for (thick = 0; thick < NTHICK; thick++)
		{
			float r;

			r = thick * (RMAX - RMIN)/(NTHICK - 1) + RMIN;

			pmesh[NODE_NUMBER(theta,thick)].x = r*cosf(angle);
			pmesh[NODE_NUMBER(theta,thick)].y = r*sinf(angle);

//			pmesh[NODE_NUMBER(theta,thick)].x = -100 + RMIN+((RMAX-RMIN)*theta)/(NTHETA - 1);//-RMIN + (2*(RMAX - RMIN)*theta)/(NTHETA - 1);
//			pmesh[NODE_NUMBER(theta,thick)].y = RMIN+((RMAX-RMIN)*thick)/(NTHICK - 1);//-RMIN + (2*(RMAX - RMIN)*theta)/(NTHETA - 1);
		}
	}

#ifdef USE_POSTSCRIPT
	postscript_gen();
#endif

	/* then build the stiffness matrix A */
	unsigned i,j;

	printf("Assembling matrix ... \n");

	for (j = 0 ; j < DIM ; j++)
	{
		for (i = 0; i < DIM ; i++)
		{
			compute_A_value(i, j);
		}
	}

	for (i = 0; i < DIM; i++)
	{
		B[i] = 0.0f;
	}

	for (i = 0; i < NTHICK; i++)
	{
		B[i] = 200.0f;
		B[DIM-1-i] = 200.0f;
	}

	for (i = 1; i < NTHETA-1; i++)
	{
		B[i*NTHICK] = 200.0f;
		B[(i+1)*NTHICK-1] = 000.0f;
	}

	/* now simplify that problem given the boundary conditions */ 
	/*
	 * The values at boundaries are well known : 
	 *
	 *	-----------
	 *	|	  |
	 *	|	  |
	 *	|	  |
	 *	-----------
	 *
	 */

	/* first create a reference vector to track pivoting */
	unsigned k;
	unsigned index = 0;
	RefArray = malloc(DIM*sizeof(int));
	for (k = 0; k < DIM; k++)
	{
		RefArray[k] = k;
	}

	/* first inner nodes */
	for (theta = 1; theta < NTHETA - 1 ; theta++)
	{
		for (thick = 1; thick < NTHICK - 1; thick++) 
		{
			/* inner nodes are unknown */
			RefArray[index++] = NODE_NUMBER(theta, thick);
		}
	}

	newsize = index;

	for (theta=0; theta < NTHETA; theta++)
	{
		/* Lower boundary "South" */
		RefArray[index++] = NODE_NUMBER(theta, 0);
		
		/* Upper boundary "North" */
		RefArray[index++] = NODE_NUMBER(theta, NTHICK-1);
	}

	for (thick = 1; thick < NTHICK -1; thick++)
	{
		/* "West "*/
		RefArray[index++] = NODE_NUMBER(0, thick);

		/* "East" */
		RefArray[index++] = NODE_NUMBER(NTHETA-1, thick);
	}

	assert(index == DIM);

	printf("Problem size : %dx%d (%dx%d)\n", newsize, newsize, DIM, DIM);

	subA = malloc(newsize*newsize*sizeof(float));
	subB = malloc(newsize*sizeof(float));
	

	for (j = 0; j < newsize; j++) 
	{
		for (i = 0; i < newsize; i++)
		{
			subA[i+j*newsize] = A[TRANSLATE(i)+DIM*TRANSLATE(j)]; 
		}
	}

	for (j = 0; j < newsize; j++)
	{
		subB[j] = B[TRANSLATE(j)];

		for (i = newsize; i < DIM; i++)
		{
			subB[j] -= B[TRANSLATE(i)]*A[TRANSLATE(i) +TRANSLATE(j)*DIM];
		}
	}


	//dummy_lu_facto(newsize);
	subLU = malloc(newsize*newsize*sizeof(float));
	factoLU(subA, subLU, newsize);

	subL = malloc(newsize*newsize*sizeof(float));
	subU = malloc(newsize*newsize*sizeof(float));
	solve_system(newsize);

	opengl_render();

	return 0;
}
