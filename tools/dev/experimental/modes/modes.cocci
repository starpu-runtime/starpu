/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
/*
 * KNOWN BUGS :
 *
 * - Undefined behaviour if a codelet is used by more than one task
 * - buffers[x].mode must be STARPU_{R,W,RW} : cant be a variable
 */
@r@
identifier c;
identifier t;
expression id;
constant MODE;
expression H;
@@
 t->cl = &c;
<...
- t->buffers[id].handle = H;
+ t->handles[id] = H;
- t->buffers[id].mode = MODE;
...>

@s depends on r@
identifier r.c;
expression r.id;
constant r.MODE;
@@
struct starpu_codelet c = {
++	.modes[id] = MODE,
};

