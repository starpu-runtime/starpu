#!/bin/sh
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

set -e
set -x

export LC_ALL=C
oldPATH=$PATH
# Add both PATHS for msys and cygdrive
export PATH="/c/Program Files/Microsoft Visual Studio/2022/Community/VC/bin":"/c/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE":"/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64":$oldPATH
export PATH="/cygdrive/c/Program Files/Microsoft Visual Studio/2022/Community/VC/bin":"/cygdrive/c/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE":"/cygdrive/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64":$PATH

cd artifacts
zipdir=$PWD
zipball=$(ls -tr starpu*.zip | tail -1)
if test -z "$zipball" ; then
    echo Zipball not available
    exit 2
fi

basename=$(basename $zipball .zip)
test -d $basename && chmod -R u+rwX $basename && rm -rf $basename
unzip $zipball

cd $basename
. ./bin/starpu_env -d $PWD
cd share/doc/starpu/tutorial/
make
./hello_world.exe
./vector_scal.exe

version=$(echo $zipball | sed 's/.*-//' | sed 's/.zip//')
# get the package id
package_id=$(curl --header "JOB-TOKEN: ${CI_JOB_TOKEN}" --url https://gitlab.inria.fr/api/v4/projects/${CI_PROJECT_ID}/packages | jq '.[] | select(.version == "'$version'") | .id')

# first try to remove the file if it already exists
file_ids=$(curl --header "JOB-TOKEN: ${CI_JOB_TOKEN}" --url https://gitlab.inria.fr/api/v4/projects/${CI_PROJECT_ID}/packages/$package_id/package_files|jq '.[] | select(.file_name == "'$zipball'") | .id')
for x in $file_ids
do
    xx=(echo $x | tr -d '\r')
    curl --request DELETE --header "JOB-TOKEN: ${CI_JOB_TOKEN}" --url https://gitlab.inria.fr/api/v4/projects/${CI_PROJECT_ID}/packages/$package_id/package_files/$xx
done

# upload zipball in package registry
curl --header "JOB-TOKEN: ${CI_JOB_TOKEN}" --upload-file $zipdir/$zipball "https://gitlab.inria.fr/api/v4/projects/${CI_PROJECT_ID}/packages/generic/starpu-windows/${version}/${zipball}"
