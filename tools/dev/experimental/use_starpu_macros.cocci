// StarPU --- Runtime system for heterogeneous multicore architectures.
//
// Copyright (C) 2011 Institut National de Recherche en Informatique et Automatique
//
// StarPU is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or (at
// your option) any later version.
//
// StarPU is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
// See the GNU Lesser General Public License in COPYING.LGPL for more details.

@@
@@
-	abort();
+	STARPU_ABORT();


@@
@@
-	assert(
+	STARPU_ASSERT(
...)


@min_max@
identifier i;
expression E1, E2;
@@
(
- 	return E1<E2?E1:E2;
+ 	return STARPU_MIN(E1, E2);
|
-	i =  E1<E2?E1:E2            // No semi-colon at the end, so that it
+	i = STARPU_MIN(E1, E2)      // matches both "i = ..." and "t i = ..."
|
-	return E1>E2?E1:E2;
+	return STAPU_MAX(E1, E2);
|
-	i = E1>E2?E1:E2             // No semi-colon at the end, so that it
+	i = STARPU_MAX(E1, E2)      // matches both "i = ..." and "t i = ..."
)
