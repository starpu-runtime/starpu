/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2013,2015                           Inria
 * Copyright (C) 2009-2017, 2019                                Université de Bordeaux
 * Copyright (C) 2010-2013,2015-2017                      CNRS
 * Copyright (C) 2013                                     Thibaut Lambert
 * Copyright (C) 2010                                     Mehdi Juhoor
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

	/* This is the base code, just like can be read in Chameleon */
	/* A source-to-source compiler can very easily produce this kind of
	code, with rewritten loops etc */

	unsigned k, m, n;

	for (k = 0; k < nblocks; k++)
	{
		POTRF(A(k,k), (2*nblocks - 2*k));

		for (m = k+1; m < nblocks; m++)
			TRSM(A(k,k), A(m,k), (2*nblocks - 2*k - m));

		for (n = k+1; n < nblocks; n++)
		{
			SYRK(A(n,k), A(n, n), (2*nblocks - 2*k - n));
			for (m = n+1; m < nblocks; m++)
				GEMM(A(m,k), A(n,k), A(m,n), (2*nblocks - 2*k - n - m));
		}
	}
