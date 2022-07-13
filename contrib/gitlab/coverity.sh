#!/bin/sh
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2021-2022   Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#

BRANCH="unknown"
if test -n "$CI_COMMIT_BRANCH"
then
    BRANCH=$CI_COMMIT_BRANCH
fi
if test -n "$CI_MERGE_REQUEST_SOURCE_BRANCH_NAME"
then
    BRANCH=$CI_MERGE_REQUEST_SOURCE_BRANCH_NAME
fi

./contrib/ci.inria.fr/job-1-check.sh -coverity $BRANCH

