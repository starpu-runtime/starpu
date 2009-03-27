#include <core/perfmodel/regression.h>

#define MAXREGITER	1000
#define EPS 1.0e-10

//#define MIN(a,b) ((a)<(b)?(a):(b))

static double compute_b(double c, unsigned n, unsigned *x, double *y)
{
	double b;

	/* X = log (x) , Y = log (y - c) */
	double sumxy = 0.0;
	double sumx = 0.0;
	double sumx2 = 0.0;
	double sumy = 0.0;

	unsigned i;
	for (i = 0; i < n; i++)
	{
		double xi = logl(x[i]);
		double yi = logl(y[i]-c);

		sumxy += xi*yi;
		sumx += xi;
		sumx2 += xi*xi;
		sumy += yi;
	}

	b = (n * sumxy - sumx * sumy) / (n*sumx2 - sumx*sumx);

	return b;
}

static double compute_a(double c, double b, unsigned n, unsigned *x, double *y)
{
	double a;

	/* X = log (x) , Y = log (y - c) */
	double sumx = 0.0;
	double sumy = 0.0;

	unsigned i;
	for (i = 0; i < n; i++)
	{
		double xi = logl(x[i]);
		double yi = logl(y[i]-c);

		sumx += xi;
		sumy += yi;
	}

	a = (sumy - b*sumx) / n;

	return a;
}



/* returns r */
static double test_r(double c, unsigned n, unsigned *x, double *y)
{
	double r;

//	printf("test c = %e\n", c);

	/* X = log (x) , Y = log (y - c) */
	double sumxy = 0.0;
	double sumx = 0.0;
	double sumx2 = 0.0;
	double sumy = 0.0;
	double sumy2 = 0.0;

	unsigned i;
	for (i = 0; i < n; i++)
	{
		double xi = logl(x[i]);
		double yi = logl(y[i]-c);

	//	printf("Xi = %e, Yi = %e\n", xi, yi);

		sumxy += xi*yi;
		sumx += xi;
		sumx2 += xi*xi;
		sumy += yi;
		sumy2 += yi*yi;
	}

	//printf("sumxy %e\n", sumxy);
	//printf("sumx %e\n", sumx);
	//printf("sumx2 %e\n", sumx2);
	//printf("sumy %e\n", sumy);
	//printf("sumy2 %e\n", sumy2);

	r = (n * sumxy - sumx * sumy) / sqrt( (n* sumx2 - sumx*sumx) * (n*sumy2 - sumy*sumy) );

	return r;
}

static unsigned find_list_size(struct history_list_t *list_history)
{
	unsigned cnt = 0;

	struct history_list_t *ptr = list_history;
	while (ptr) {
		cnt++;
		ptr = ptr->next;
	}

	return cnt;
}

static double find_list_min(double *y, unsigned n)
{
	double min = 1.0e30;

	unsigned i;
	for (i = 0; i < n; i++)
	{
		min = STARPU_MIN(min, y[i]);
	}

	return min;
}

static void dump_list(unsigned *x, double *y, struct history_list_t *list_history)
{
	struct history_list_t *ptr = list_history;
	unsigned i = 0;

	while (ptr) {
		x[i] = ptr->entry->size;
		y[i] = ptr->entry->mean;

		ptr = ptr->next;
		i++;
	}
}


/* y = ax^b + c 
 * 	return 0 if success, -1 otherwise
 * 	if success, a, b and c are modified
 * */
int regression_non_linear_power(struct history_list_t *ptr, double *a, double *b, double *c)
{
	unsigned n = find_list_size(ptr);

	unsigned *x = malloc(n*sizeof(unsigned));
	STARPU_ASSERT(x);

	double *y = malloc(n*sizeof(double));
	STARPU_ASSERT(y);

	dump_list(x, y, ptr);

	double cmin = 0.0;
	double cmax = find_list_min(y, n);
	
	unsigned iter;

	double err = 100000.0;

	for (iter = 0; iter < MAXREGITER; iter++)
	{
		double c1, c2;
		double r1, r2;
		
		double radius = 0.01;

		c1 = cmin + (0.5-radius)*(cmax - cmin);
		c2 = cmin + (0.5+radius)*(cmax - cmin);

		r1 = test_r(c1, n, x, y);
		r2 = test_r(c2, n, x, y);

		double err1, err2;
		err1 = fabsl(1.0 - r1);
		err2 = fabsl(1.0 - r2);

		if (err1 < err2)
		{
			cmax = (cmin + cmax)/2;
		}
		else {
			/* 2 is better */
			cmin = (cmin + cmax)/2;
		}

		if (fabsl(err - STARPU_MIN(err1, err2)) < EPS)
		{
			err = STARPU_MIN(err1, err2);
			break;
		}

		err = STARPU_MIN(err1, err2);
	}

	*c = (cmin + cmax)/2;

	*b = compute_b(*c, n, x, y); 
	*a = expl(compute_a(*c, *b, n, x, y));

	free(x);
	free(y);

	return 0;
}

