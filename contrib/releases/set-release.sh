#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2024-2024   Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

# download the change log
wget https://files.inria.fr/starpu/starpu-${release}/log.txt -O log-${release}.txt 2>/dev/null
changelog=$(cat log-${release}.txt | tr '*' '-')
echo ${changelog}

# Try to remove the release if it already exists
curl --request DELETE --header "JOB-TOKEN: ${CI_JOB_TOKEN}" https://gitlab.inria.fr/api/v4/projects/${CI_PROJECT_ID}/releases/starpu-${release}

# copy the source archive and the documentation to the Gitlab's Package registry
wget https://files.inria.fr/starpu/starpu-${release}/starpu-${release}.tar.gz -O starpu-${release}.tar.gz  2>/dev/null
curl --header "JOB-TOKEN: ${CI_JOB_TOKEN}" --upload-file ./starpu-${release}.tar.gz "https://gitlab.inria.fr/api/v4/projects/${CI_PROJECT_ID}/packages/generic/starpu/${release}/starpu-${release}.tar.gz"

wget https://files.inria.fr/starpu/starpu-${release}/starpu.pdf -O starpu-${release}.pdf  2>/dev/null
curl --header "JOB-TOKEN: ${CI_JOB_TOKEN}" --upload-file ./starpu-${release}.pdf "https://gitlab.inria.fr/api/v4/projects/${CI_PROJECT_ID}/packages/generic/starpu/${release}/starpu-${release}.pdf"

# Create a file with the CLI to create the release
(
    echo "release-cli create --name \"Release ${release}\" --tag-name ${CI_COMMIT_TAG} --ref \"${CI_COMMIT_REF_NAME}\" --description \"${changelog}\" \\"
    echo "--assets-link \"{\\\"name\\\":\\\"Release download page\\\",\\\"url\\\":\\\"https://files.inria.fr/starpu/starpu-${release}\\\"}\" \\"
    for asset in starpu-${release}.pdf starpu-${release}.tar.gz
    do
        echo "--assets-link \"{\\\"name\\\":\\\"${asset}\\\",\\\"url\\\":\\\"https://gitlab.inria.fr/api/v4/projects/${CI_PROJECT_ID}/packages/generic/starpu/${release}/${asset}\\\"}\" \\"
    done
    echo ";"
) > ./release-cli.txt
