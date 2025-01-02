#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2024-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

scriptExists=$(ssh luckyluke ls $HOME/softs/starpu/starpu-scripts/mirror/scripts/uploadRelease.sh 2>/dev/null)
if test -z "$scriptExists"
then
    echo This runner is not eligible to deploy new releases for StarPU
    ssh luckyluke ls $HOME/softs/starpu/starpu-scripts/mirror/scripts/uploadRelease.sh
    ssh luckyluke ls $HOME/softs/starpu/starpu-scripts/mirror/scripts
    exit 1
fi


echo '--------------------------------'
env
echo '--------------------------------'

DATE=$(echo $CI_COMMIT_TIMESTAMP | sed 's/T.*//')

# Get the release name through the branch name, and through the ChangeLog file.
# Both have to match to be correct
RELEASE_NAME=`echo ${CI_COMMIT_TAG} | sed 's/starpu-//'`
firstline=$(grep -n "^StarPU" ChangeLog | head -n 1 | cut -d ':' -f 1)
release=$(head -n $firstline ChangeLog | tail -n 1 | sed 's/StarPU//' | tr -d ' ')

if [ -z "${RELEASE_NAME}" -o -z "${release}" -o "${RELEASE_NAME}" != "${release}" ]
then
    echo "Commit name ${RELEASE_NAME} is different from ChangeLog name ${release}"
    exit 1
fi

# compile and deploy
set +e
mpiexec -oversubscribe pwd 2>/dev/null
ret=$?
set -e
ARGS=""
if test "$ret" = "0"
then
    ARGS="--with-mpiexec-args=-oversubscribe"
fi

CONFIGURE_OPTIONS="--enable-quick-check --enable-mpi-minimal-tests $ARGS --enable-verbose --enable-debug"

export STARPU_MICROBENCHS_DISABLED=1
export STARPU_TIMEOUT_ENV=3600
export MPIEXEC_TIMEOUT=3600
export STARPU_NCUDA=2
export STARPU_OPENCL_PROGRAM_DIR=$PWD

# using a separate build directory messes up with the coverage output which extracts files both from the source and build directory
./autogen.sh
./configure --enable-build-doc-pdf --enable-coverage $CONFIGURE_OPTIONS $STARPU_USER_CONFIGURE_OPTIONS
make -j4
make check
make dist
rm -rf coverage_$$ && mkdir coverage_$$
lcov --directory . --capture --output coverage_$$/coverage.info
genhtml --output-directory coverage_$$/coverage coverage_$$/coverage.info

# set up artifacts
ARTIFACTS=./artifacts
mkdir ${ARTIFACTS}

## main file
(
    cat ${dir}/head.org
    echo "#+TITLE: StarPU Release starpu-${release}"
    cat <<EOF

* Informations
- The latest nightly tarball successfully passing 'make check' is
  available at [[./starpu-${release}.tar.gz][starpu-${release}.tar.gz]] (produced on $DATE).

- The coverage report is available [[./starpu-${release}.lcov][as a single file]] or [[./coverage][as a HTML page]].

- The StarPU full documentation is available in [[./starpu.pdf][PDF]] and [[./html/index.html][HTML]].

- The StarPU developers documentation is available in [[./starpu_dev.pdf][PDF]] and in [[./html_dev/index.html][HTML]].

- Other StarPU documentations are available [[./doc.html][here]].
EOF
) > ${ARTIFACTS}/index.org

## log.txt
firstline=$(grep -n "^StarPU" ./ChangeLog | head -n 1 | cut -d ':' -f 1)
lastline=$(grep -n "^StarPU" ./ChangeLog | head -n 2 | tail -n 1 | cut -d ':' -f 1)
lastline=$((lastline - 1))
(
    for i in `seq $firstline $lastline`
    do
        head -n $i ./ChangeLog | tail -n 1
    done
)  | tr '*' '-' > ${ARTIFACTS}/log.txt

## coverage
cp coverage_$$/coverage.info ${ARTIFACTS}/starpu-${release}.lcov
cp -rp coverage_$$/coverage ${ARTIFACTS}/

## source .gz
cp starpu-${release}.tar.gz ${ARTIFACTS}/
md5sum starpu-${release}.tar.gz | awk '{print $1}' > ${ARTIFACTS}/starpu-${release}.tar.gz.md5

## documentation
mkdir -p ${ARTIFACTS}/doc
if test -f doc/README.org
then
    (
	cat ${dir}/head.org
	echo "#+TITLE: StarPU Release starpu-${release} - Documentation"
	echo "* Documentation produced on ${DATE}"
	cat doc/README.org
    ) > ${ARTIFACTS}/doc/doc.org
fi

for doc in "" _dev _web_basics _web_extensions _web_faq _web_installation _web_introduction _web_languages _web_performances _web_applications
do
    if test -f doc/doxygen${doc}/starpu${doc}.pdf
    then
	cp -p doc/doxygen${doc}/starpu${doc}.pdf ${ARTIFACTS}/doc/
    fi
    if test -d doc/doxygen${doc}/html${doc}
    then
	cp -rp doc/doxygen${doc}/html${doc} ${ARTIFACTS}/doc/
    fi
done

make clean

# execute on the frontal node to upload release files on files.inria.fr/starpu
ssh luckyluke $HOME/softs/starpu/starpu-scripts/mirror/scripts/uploadRelease.sh ${PWD}/artifacts ${release} ${DATE}

