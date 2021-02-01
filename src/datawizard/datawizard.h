/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DATAWIZARD_H__
#define __DATAWIZARD_H__

/** @file */

#include <starpu.h>
#include <common/config.h>

#include <common/utils.h>

#include <datawizard/coherency.h>
#include <datawizard/filters.h>
#include <datawizard/copy_driver.h>
#include <datawizard/footprint.h>

#include <datawizard/data_request.h>
#include <datawizard/interfaces/data_interface.h>

#include <core/dependencies/implicit_data_deps.h>

/** Make data transfers progress on node \p memory_node.
 *
 * If \p push_requests is 1, it can start new transfers
 *
 * If \p may_alloc is 1, it can allocate destination data for transfers
 * (this is not possible e.g. when spinning for a handle lock)
 */
int ___starpu_datawizard_progress(unsigned memory_node, unsigned may_alloc, unsigned push_requests);
/** Call ___starpu_datawizard_progress() for all memory nodes driven by the
 * current worker */
int __starpu_datawizard_progress(unsigned may_alloc, unsigned push_requests);
/** Call __starpu_datawizard_progress with push_requests = 1 */
void _starpu_datawizard_progress(unsigned may_alloc);

#endif // __DATAWIZARD_H__
