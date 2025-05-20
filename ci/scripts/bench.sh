#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2025-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

set -x
set -e
dir=$(realpath $(dirname $0))

SCRIPT_NAME="$HOME/scratch/starpu/benchmarks/scripts/uploadBenchs.sh"
scriptExists=$(ssh luckyluke ls $SCRIPT_NAME 2>/dev/null)
if test -z "$scriptExists"
then
    echo This runner is not eligible to deploy new benchmarks for StarPU
    ssh luckyluke ls $SCRIPT_NAME
    ssh luckyluke ls $(dirname $SCRIPT_NAME)
    exit 1
fi

ARTIFACTS=$1
if test ! -d "$ARTIFACTS"
then
    echo "Error. Directory <$ARTIFACTS> not found"
    exit 1
fi

RELEASE_STAMPFILE=$ARTIFACTS/latest_release
if test ! -f "$RELEASE_STAMPFILE"
then
    echo "Error. File <$RELEASE_STAMPFILE> not found"
    exit 1
fi

RELEASE_VERSION=$(cat $RELEASE_STAMPFILE)
RELEASE_DIR=$(dirname $RELEASE_STAMPFILE)/$RELEASE_VERSION
if test ! -d "$RELEASE_DIR"
then
    echo "Error. Directory $RELEASE_DIR not found"
    exit 1
fi

BRANCH=$(cat $RELEASE_DIR/branch_name)
if test -z "$BRANCH"
then
    echo "Error. Branch not defined. File <$RELEASE_DIR/branch_name> missing or empty"
    exit 1
fi

TMP_DIR=$(ssh luckyluke mktemp -d 2>/dev/null)
# copy files on the frontal node
scp -pr $RELEASE_STAMPFILE $RELEASE_DIR luckyluke:$TMP_DIR/$(dirname $RELEASE_STAMPFILE)/

# execute on the frontal node to upload benchmarks on the web
ssh luckyluke $SCRIPT_NAME $BRANCH $TMP_DIR/$RELEASE_STAMPFILE

