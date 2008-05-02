#include "heat.h"

/* default values */
unsigned ntheta = 32+2;
unsigned nthick = 32+2;
unsigned nblocks = 16;

#define DIM	ntheta*nthick

#define RMIN	(150.0f)
#define RMAX	(200.0f)

#define Pi	(3.141592f)

#define NODE_NUMBER(theta, thick)	((thick)+(theta)*nthick)
#define NODE_TO_THICK(n)		((n) % nthick)
#define NODE_TO_THETA(n)		((n) / nthick)

typedef struct point_t {
	float x;
	float y;
} point;

float minval, maxval;
float *result;
int *RefArray;
point *pmesh;
float *A;
float *subA;
float *B;
float *subB;
float *subL;
float *subU;

float *Bformer;

unsigned printmesh =0;

int argc_;
char **argv_;


unsigned version = 2;

	/*
	 *   B              C
	 *	**********
	 *	*  0   * *
	 *	*    *   *
	 *	*  *   1 *
	 *	**********
	 *   A             D
	 */


#ifdef OPENGL_RENDER
/*
 * Just some dummy OpenGL code to display our results 
 *
 */

static void generate_graph()
{
	unsigned theta, thick;

	for (theta = 0; theta < ntheta-1; theta++)
	{
		for (thick = 0; thick < nthick-1; thick++)
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
//   glRotatef(-0,0.0,0.0,0.0);
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
	fprintf(stderr, "OpenGL rendering ... \n");

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
#endif // OPENGL_RENDER
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

	assert(theta_tr + 2 <= ntheta);
	assert(thick_tr + 2 <= nthick);

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

	ASSERT(theta_tr + 2 <= ntheta);
	ASSERT(thick_tr + 2 <= nthick);

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
	if (theta_tr + 2  > (int)ntheta) return 0.0f;

	if (thick_tr < 0) return 0.0f;
	if (thick_tr + 2 > (int)nthick) return 0.0f;

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

	for (theta = 0; theta < ntheta-1; theta++)
	{
		for (thick = 0; thick < nthick-1; thick++)
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

static void solve_system(unsigned size, unsigned subsize)
{
	unsigned i;

	/* solve the actual problem LU X = B */
        /* solve LX' = Y with X' = UX */
        /* solve UX = X' */
	fprintf(stderr, "Solving the problem ...\n");

	/* L */
	cblas_strsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
			subsize, A, size, B, 1);

	/* U */
        cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasUnit,
                        subsize, A, size, B, 1);

	ASSERT(DIM == size);

	/* now display back the ACTUAL result */
	for (i = 0; i < subsize; i++)
	{
		result[TRANSLATE(i)] = B[i];
	}
	for (i = subsize ; i < size; i++)
	{
		result[TRANSLATE(i)] = Bformer[TRANSLATE(i)];
	}

}

void reorganize_matrices(float *A, float *B, int *RefArray, unsigned size)
{
	/* only reorganize the newsize*newsize upper left square on A, and the
	 * newsize first items on B */
	int i;
	for (i = 0; i < (int)size; i++)
	{
		if (RefArray[i] > i) {
			/* swap i and RefArray[i] columns on A */
			cblas_sswap (size, &A[i], size, &A[RefArray[i]], size);

			/* swap i and RefArray[i] rows on A */
			cblas_sswap (size, &A[i*size], 1, &A[RefArray[i]*size], 1);

			/* swap i and RefArray[i] rows on B */
			cblas_sswap (1, &B[i], 1, &B[RefArray[i]], 1);
		}
	}
}

unsigned shape = 0;

void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-shape") == 0) {
		        char *argptr;
			shape = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nthick") == 0) {
		        char *argptr;
			nthick = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-ntheta") == 0) {
		        char *argptr;
			ntheta = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks") == 0) {
		        char *argptr;
			nblocks = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-v1") == 0) {
			version = 1;
		}

		if (strcmp(argv[i], "-v2") == 0) {
			version = 2;
		}

		if (strcmp(argv[i], "-h") == 0) {
			/* TODO */
		}
	}
}

int main(int argc, char **argv)
{
	argc_ = argc;
	argv_ = argv;

#ifdef USE_MARCEL
	marcel_init(&argc, argv);
#endif

	parse_args(argc, argv);

	unsigned theta, thick;

	unsigned newsize;

	pmesh = malloc(DIM*sizeof(point));

	/* the stiffness matrix : boundary conditions are known */
	A = malloc(DIM*DIM*sizeof(float));
	B = malloc(DIM*sizeof(float));

	/* first build the mesh by determining all points positions */
	for (theta = 0; theta < ntheta; theta++)
	{
		float angle;

		angle = (ntheta - 1 - theta) * Pi/(ntheta-1);

		for (thick = 0; thick < nthick; thick++)
		{
			float r;

			r = thick * (RMAX - RMIN)/(nthick - 1) + RMIN;

			switch (shape) {
				default:
				case 0:
					pmesh[NODE_NUMBER(theta,thick)].x = r*cosf(angle);
					pmesh[NODE_NUMBER(theta,thick)].y = r*sinf(angle);
					break;
				case 1:
					pmesh[NODE_NUMBER(theta,thick)].x = -100 + RMIN+((RMAX-RMIN)*theta)/(ntheta - 1);
					pmesh[NODE_NUMBER(theta,thick)].y = RMIN+((RMAX-RMIN)*thick)/(nthick - 1);
					break;
				case 2:
					pmesh[NODE_NUMBER(theta,thick)].x = r*(2.0f*theta/(ntheta - 1)- 1.0f);
					pmesh[NODE_NUMBER(theta,thick)].y = r*(2.0f*thick/(nthick - 1)- 1.0f);
					break;
			}
		}
	}

#ifdef USE_POSTSCRIPT
	postscript_gen();
#endif

	/* then build the stiffness matrix A */
	unsigned i,j;

	fprintf(stderr, "Assembling matrix ... \n");

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

	for (i = 0; i < nthick; i++)
	{
		B[i] = 200.0f;
		B[DIM-1-i] = 200.0f;
	}

	for (i = 1; i < ntheta-1; i++)
	{
		B[i*nthick] = 200.0f;
		B[(i+1)*nthick-1] = 100.0f;
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
	for (theta = 1; theta < ntheta - 1 ; theta++)
	{
		for (thick = 1; thick < nthick - 1; thick++) 
		{
			/* inner nodes are unknown */
			RefArray[index++] = NODE_NUMBER(theta, thick);
		}
	}

	newsize = index;

	for (theta=0; theta < ntheta; theta++)
	{
		/* Lower boundary "South" */
		RefArray[index++] = NODE_NUMBER(theta, 0);
		
		/* Upper boundary "North" */
		RefArray[index++] = NODE_NUMBER(theta, nthick-1);
	}

	for (thick = 1; thick < nthick -1; thick++)
	{
		/* "West "*/
		RefArray[index++] = NODE_NUMBER(0, thick);

		/* "East" */
		RefArray[index++] = NODE_NUMBER(ntheta-1, thick);
	}

	assert(index == DIM);

	Bformer = malloc(DIM*sizeof(float));
	memcpy(Bformer, B, DIM*sizeof(float));

	fprintf(stderr, "Problem size : %dx%d (%dx%d)\n", newsize, newsize, DIM, DIM);

	reorganize_matrices(A, B, RefArray, DIM);

	for (j = 0; j < newsize; j++)
	{
		for (i = newsize; i < DIM; i++)
		{
			B[j] -= B[i]*A[i+j*DIM];
		}
	}

	result = malloc(DIM*sizeof(float));

	dw_factoLU(A, newsize, DIM, nblocks, version);

	solve_system(DIM, newsize);

#ifdef OPENGL_RENDER
	opengl_render();
#endif

	free(subA);
	free(subB);
	free(pmesh);
	free(result);
	free(RefArray);

	return 0;
}
