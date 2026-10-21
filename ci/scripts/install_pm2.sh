#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017-2022, 2026  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

cur_dir=${PWD}
XROOT=$(realpath ${starpu_build_dir}/../pm2)
mkdir -p ${XROOT}
cd ${XROOT}

git clone https://gitlab.inria.fr/pm2/pm2.git
cd pm2
cd scripts
if test ! -f $1
then
    echo "error. configuration file $1 is not available"
    exit 1
fi

XPREFIX=$XROOT/install
if test "$2" != ""
then
    XPREFIX=$XROOT/install/$2
fi
./pm2-build-packages --purge $1 --prefix=$XPREFIX

(
cat <<EOF
#!/bin/bash
exec $XPREFIX/bin/mpiexec --padico-binding=none "$@"
EOF
) > $XPREFIX/bin/mpiexec_nobind
chmod +x $XPREFIX/bin/mpiexec_nobind

. $XPREFIX/share/${1}.sh
cd $cur_dir
