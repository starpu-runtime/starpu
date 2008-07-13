#include <examples/heat.h>

#ifdef OPENGL_RENDER
/*
 * Just some dummy OpenGL code to display our results 
 *
 */

static float minval, maxval;

static unsigned ntheta;
static unsigned nthick;
static float *result;
static unsigned printmesh =0;
static point *pmesh;

static void generate_graph(void)
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


static void pressKey(unsigned char key, int x __attribute__ ((unused)), int y  __attribute__ ((unused)))
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


void opengl_render(unsigned _ntheta, unsigned _nthick, float *_result, point *_pmesh, int argc_, char **argv_)
{
	unsigned i;
	fprintf(stderr, "OpenGL rendering ... \n");

	ntheta = _ntheta;
	nthick = _nthick;
	result = _result;
	printmesh = 0;
	pmesh = _pmesh;

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

#ifdef USE_POSTSCRIPT
static void postscript_gen(void)
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


