/* GCC-StarPU
   Copyright (C) 2012 Institut National de Recherche en Informatique et Automatique

   GCC-StarPU is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   GCC-StarPU is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GCC-StarPU.  If not, see <http://www.gnu.org/licenses/>.  */

/* (instructions compile) */

/* Make sure warnings get raised when pointer variables are likely never
   registered.  */

extern void my_task (double *x, double *y)
  __attribute__ ((task));

extern double *global1;
double global2[123];

void
parm_decl_no_warn (double *parm1, double *parm2)
{
  my_task (parm1, parm2);	  /* no warning, because these are parameters
				     so we cannot tell anything */
}

void
global_decl_no_warn (void)
{
  my_task (global1, global2);
}

void
two_unregistered_pointers (void)
{
  double *p, *q;

  p = malloc (12 * sizeof *p);
  q = malloc (23 * sizeof *q);

  my_task (p, q); /* (warning "p.* used unregistered") *//* (warning "q.* used unregistered") */
}

void
one_unregistered_pointer (void)
{
  double *p, *q;

  p = malloc (12 * sizeof *p);
  q = malloc (23 * sizeof *q);

#pragma starpu register p 12
  my_task (p, q);		      /* (if optimizing?
					     (warning "q.* used unregistered")) */
}

void
another_unregistered_pointer (void)
{
  double X[] = { 1, 2, 3, 4 };
  double *Y;

  Y = malloc (123 * sizeof *Y);
  if (Y == NULL)
    return;
  else
    {
      extern void frob (double *);
      frob (Y);
    }
  X[0] = 42;

#pragma starpu register Y 123
  my_task (X, Y);		      /* (warning "X.* used unregistered") */
}

void
zero_unregistered_pointers (void)
{
  double *p, *q;

  p = malloc (12 * sizeof *p);
  q = malloc (23 * sizeof *q);

#pragma starpu register p 12
#pragma starpu register q 23
  my_task (p, q);				  /* no warning */
}

void
two_pointers_unregistered_before_call (void)
{
  double *p, *q;

  p = malloc (12 * sizeof *p);
  q = malloc (23 * sizeof *q);

  my_task (p, q); /* (warning "p.* used unregistered") *//* (warning "q.* used unregistered") */

#pragma starpu register p 12
#pragma starpu register q 23
}

void
one_unregistered_array (void)
{
  double PPP[12], QQQ[23];

#pragma starpu register PPP	      /* (warning "on-stack .* unsafe") */
  my_task (PPP, QQQ);		      /* (warning "QQQ.* used unregistered") */
}

void
not_the_ones_registered (void)
{
  double a[12], b[23], p[12], q[23];

#pragma starpu register a		 /* (warning "on-stack .* unsafe") */
#pragma starpu register b		 /* (warning "on-stack .* unsafe") */

  my_task (p, q); /* (warning "p.* used unregistered") */ /* (warning "q.* used unregistered") */
}

void
registered_pointers_with_aliases (void)
{
  double *a, *b, *p, *q;

  a = malloc (123 * sizeof *a);
  b = malloc (234 * sizeof *b);

#pragma starpu register a 123
#pragma starpu register b 234
  p = a;
  q = b;
  my_task (p, q);				  /* no warning */
}

void
one_unregistered_array_attrs (void)
{
  double p[12];
  double q[23] __attribute__ ((heap_allocated, registered));

  my_task (p, q);		      /* (warning "p.* used unregistered") */
}

void
unregistered_on_one_path (int x)
{
  double p[12], q[34];

  if (x > 42)
    {
#pragma starpu register p	      /* (warning "on-stack .* unsafe") */
    }

#pragma starpu register q	      /* (warning "on-stack .* unsafe") */

  my_task (p, q);		      /* (warning "p.* used unregistered") */
}

void
registered_via_two_paths (int x)
{
  double p[12], q[34];

  if (x > 42)
    {
#pragma starpu register p	      /* (warning "on-stack .* unsafe") */
    }
  else
    {
#pragma starpu register p	      /* (warning "on-stack .* unsafe") */
    }

#pragma starpu register q	      /* (warning "on-stack .* unsafe") */

  my_task (p, q);				  /* no warning */
}

#if 0

/* FIXME: This case currently triggers a false positives.  */

void
registered_and_used_in_loop (void)
{
  int i;
  double *p[123];
  static double q[234];

  for (i = 0; i < 123; i++)
    {
      p[i] = malloc (123 * sizeof *p[i]);
#pragma starpu register p[i] 123
    }

  for (i = 0; i < 123; i++)
    my_task (p[i], q);
}

#endif
