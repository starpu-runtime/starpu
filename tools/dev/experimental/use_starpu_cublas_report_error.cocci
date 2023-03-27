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
// When using the CUBLAS library, we call cublasGetError() from times to times.
// If it does not return CUBLAS_STATUS_SUCCESS, we should not immediatly abort:
// first, we need to print a useful piece of information to make it easier to
// debug the program.

virtual context
virtual org
virtual patch
virtual report


// Pattern:
//	status = cubalsGetError();
// 	if (status != CUBLAS_STATUS_SUCCESS)
// 		STARPU_ABORT();
//
@starpu_abort@
cublasStatus_t status;
position p;
@@
status = cublasGetError();
(
if (status != CUBLAS_STATUS_SUCCESS)
  STARPU_ABORT@p();
|
if (STARPU_UNLIKELY(status != CUBLAS_STATUS_SUCCESS))
  STARPU_ABORT@p();
)

@depends on starpu_abort && context@
position starpu_abort.p;
@@
* STARPU_ABORT@p();

@script:python depends on starpu_abort && org@
p << starpu_abort.p;
@@
coccilib.org.print_todo(p[0], "Use STARPU_CUBLAS_REPORT_ERROR() instead of STARPU_ABORT().")

@depends on starpu_abort && patch@
cublasStatus_t starpu_abort.status;
position starpu_abort.p;
@@
- STARPU_ABORT@p();
+ STARPU_CUBLAS_REPORT_ERROR(status);

@script:python depends on starpu_abort && report@
p << starpu_abort.p;
@@
coccilib.report.print_report(p[0], "Use STARPU_CUBLAS_REPORT_ERROR() instead of STARPU_ABORT().")




// Pattern:
// 	status = cublasGetError();
// 	STARPU_ASSERT(!status);
@starpu_assert@
cublasStatus_t status;
position p;
@@
status = cublasGetError();
STARPU_ASSERT@p(!status);

@depends on starpu_assert && context@
position starpu_assert.p;
@@
* STARPU_ASSERT@p(...);

@script:python depends on starpu_assert && org@
p << starpu_assert.p;
@@
coccilib.org.print_todo(p[0], "Use STARPU_CUBLAS_REPORT_ERROR() instead of STARPU_ASSERT().")


@depends on starpu_assert && patch@
position starpu_assert.p;
cublasStatus_t starpu_assert.status;
@@
- STARPU_ASSERT@p(!status);
+ if (STARPU_UNLIKELY(status != CUBLAS_STATUS_SUCCESS))
+ 	STARPU_CUBLAS_REPORT_ERROR(status);

@script:python depends on starpu_assert && report@
p << starpu_assert.p;
@@
coccilib.report.print_report(p[0], "Use STARPU_CUBLAS_REPORT_ERROR() instead of STARPU_ASSERT().")
