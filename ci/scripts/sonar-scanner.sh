#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017-2026   Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
export SOURCES_TO_ANALYZE="include src mpi/src tools tests examples mpi/tests mpi/examples"
export SONAR_SOURCES=$(echo $SOURCES_TO_ANALYZE | tr ' ' ',')
export SOURCES_TO_EXCLUDE="-itools/dev"

#export SONAR_BRANCH=$(git status|head -1|awk '{print $NF}')

if test -z "$SONAR_BRANCH"
then
    echo "Error. Environment variable SONAR_BRANCH missing"
    exit 1
fi
if test -z "$SONAR_TOKEN"
then
    echo "Error. Environment variable SONAR_TOKEN missing"
    exit 1
fi
export SONAR_PROJECT_KEY=$(echo "storm:starpu:git")

if test -f config.log
then
    echo Error no config.log file found
    exit 1
fi
eval $(grep ^STARPU_SRC_DIR config.log )
export STARPU_SRC_DIR

# clean valgrind core files
rm -f $(find . -name "vgcore*")

export CPPCHECK_INCLUDES="-I$STARPU_SRC_DIR/include -I$STARPU_SRC_DIR/mpi/include -I$STARPU_SRC_DIR/src -I./src"
# The gcc command allows to include the system paths for gcc, which implies the availability of libc6-dev-i386
export SONAR_INC=$(echo $STARPU_SRC_DIR/include,$STARPU_SRC_DIR/mpi/include,$STARPU_SRC_DIR/src,./src,$STARPU_SRC_DIR/examples,$STARPU_SRC_DIR/mpi/examples)
#,$(echo | gcc -E -Wp,-v - 2>&1 | grep "^ " | tr '\n' ','))
export SONAR_INCLUDES=$(echo $SONAR_INC | tr -d ' ')

export DEFINITIONS_LOG=$(grep "^#define" ./config.log | sed -e "s#\#define #-D#g" | sed -e "s# #=#g" | xargs)
export DEFINITIONS_SRC=$(grep "^#undef" $STARPU_SRC_DIR/src/common/config.h.in| sed -e "s#\#undef #-D#g" | sed -e "s#\$#=1#g" | xargs)
export DEFINITIONS_LOCAL=$(grep -rs "#ifdef" $STARPU_SRC_DIR/src/ $STARPU_SRC_DIR/mpi/src |awk -F':' '{print $2}' | awk '{print "-D"$2"=1"}' |sort|uniq)
export DEFINITIONS=$(echo ${DEFINITIONS_LOG} ${DEFINITIONS_SRC} ${DEFINITIONS_LOCAL} | tr ' ' '\012'|grep -v STARPU_NO_ASSERT|sort|uniq|tr '\012' ' ')

# run scan-build make to generate clang sa reports
#make clean
#scan-build -plist --intercept-first --analyze-headers -o analyzer_reports make V=1 2>&1 > starpu_make.log

# Disable builtin_expect and __attribute calls, sonar-scanner does not support them
#sed -e 's/#  define STARPU_UNLIKELY(expr)          (__builtin_expect(!!(expr),0))/#  define STARPU_UNLIKELY(expr) (expr)/' -i include/starpu_util.h
#sed -e 's/#  define STARPU_LIKELY(expr)            (__builtin_expect(!!(expr),1))/#  define STARPU_LIKELY(expr) (expr)/' -i include/starpu_util.h
#sed -e 's/__attribute__.*//' -i include/starpu_util.h

# Run rats analysis
#echo "Run rats analysis..."
#rats -w 3 --xml ${SOURCES_TO_ANALYZE} > starpu-rats.xml
#RATS_XMLS=$(ls -m starpu-rats*.xml)

# Run cppcheck analysis
#echo "Run cppcheck analysis..."
#cppcheck -j 64 --language=c --platform=unix64 --force -v --enable=all --inline-suppr --xml --xml-version=2 --suppress=purgedConfiguration ${DEFINITIONS} -USTARPU_NO_ASSERT ${CPPCHECK_INCLUDES} ${SOURCES_TO_EXCLUDE} ${SOURCES_TO_ANALYZE} >starpu-cppcheck.log 2>starpu-cppcheck.xml
#--suppressions-list=tools/cppcheck/suppressions.txt
#--template="[{file}:{line}]: {id} ({severity}) {message}"
#tail -10 starpu-cppcheck.log

# Retrieve valgrind report files
# VALGRIND_XMLS=$(ls -m valgrind/*xml)

# Run pylint
echo "Run pylint analysis..."
(pylint $(find . -not -path "*dev*" -type f -name "*.py") || true) > ./starpu-pylint.log

# Run gcov
# echo "Generate coverage report..."
# lcov --directory . --capture --output starpu.lcov
# lcov --summary starpu.lcov
# genhtml -o coverage starpu.lcov
# gcovr --xml-pretty --exclude-unreachable-branches --print-summary -o coverage.xml --root .
# source $HOME/.venv/bin/activate
# lcov_cobertura starpu.lcov --output starpu_coverage.xml

# generate gcc log file
#echo "Generate gcc log file..."
#make clean
#make > ./starpu-gcc.log 2>&1

# get version id
if test "$SONAR_BRANCH" == "master"
then
    PROJECT_VERSION="master"
else
    PROJECT_VERSION=$(grep AC_INIT configure.ac | sed 's/AC_INIT(\[StarPU\], \[//' | sed 's/\].*//')
    #xbranch=$(grep STARPU_EFFECTIVE_VERSION STARPU-VERSION | sed 's/STARPU_EFFECTIVE_VERSION=//')
    #xversion=$(git tag | grep starpu-$xbranch | sed 's/starpu-'$xbranch'.//' | sort -nr | head -1)
    #xnext_version=$(( xversion + 1 ))
    #PROJECT_VERSION=$xbranch.$xnext_version
fi

# Create the config for sonar-scanner
cat > sonar-project.properties << EOF
sonar.host.url=https://sonarqube.inria.fr/sonarqube
sonar.token=${SONAR_TOKEN}
sonar.links.homepage=http://starpu.gitlabpages.inria.fr/
sonar.links.ci=https://ci.inria.fr/starpu/
sonar.links.scm=https://gitlab.inria.fr/starpu/starpu
sonar.projectKey=${SONAR_PROJECT_KEY}
sonar.projectDescription=StarPU
sonar.branch.name=${SONAR_BRANCH}
sonar.projectVersion=${PROJECT_VERSION}
sonar.scm.disabled=false
sonar.cfamily.threads=16
sonar.sourceEncoding=UTF-8
sonar.sources=${SONAR_SOURCES}
sonar.exclusions=tools/dev/**,examples/pi/**,**/loader.c,**/*.sh,**/*.html,examples/cholesky/cholesky_compiled.c
sonar.cxx.includeDirectories=${SONAR_INCLUDES}
sonar.cxx.file.suffixes=.cpp,.c,.h
sonar.cxx.errorRecoveryEnabled=true
sonar.cxx.gcc.encoding=UTF-8
sonar.cxx.gcc.regex=(?<file>.*):(?<line>[0-9]+):[0-9]+:\\\x20warning:\\\x20(?<message>.*)\\\x20\\\[(?<id>.*)\\\]
sonar.cxx.gcc.reportPaths=starpu-gcc.log
sonar.cxx.cppcheck.reportPaths=starpu-cppcheck.xml
#sonar.cxx.cobertura.reportPaths=starpu_coverage.xml
sonar.cxx.rats.reportPaths=${RATS_XMLS}
sonar.python.pylint.reportPaths=starpu-pylint.log
sonar.cxx.valgrind.reportPaths=${VALGRIND_XMLS}
#sonar.cxx.clangsa.reportPaths=analyzer_reports/*/*.plist
EOF

cat >> sonar-project.properties << EOF
sonar.issue.ignore.multicriteria=e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19
# Reserved names should not be used for preprocessor macros
sonar.issue.ignore.multicriteria.e1.ruleKey=cxx:ReservedNames
sonar.issue.ignore.multicriteria.e1.resourceKey=**
# Split this 161 characters long line (which is greater than 160 authorized).
sonar.issue.ignore.multicriteria.e2.ruleKey=cxx:TooLongLine
sonar.issue.ignore.multicriteria.e2.resourceKey=**
# 196 more comment lines need to be written to reach the minimum threshold of 25.0% comment density.
# BUG: doesn't seem to match properly, even with * or so on instead of ++
sonar.issue.ignore.multicriteria.e3.ruleKey=common-c++:InsufficientCommentDensity
sonar.issue.ignore.multicriteria.e3.resourceKey=**
# Complete the task associated to this TODO comment.
sonar.issue.ignore.multicriteria.e4.ruleKey=cxx:TodoTagPresence
sonar.issue.ignore.multicriteria.e4.resourceKey=**
# Missing curly brace.
sonar.issue.ignore.multicriteria.e5.ruleKey=cxx:MissingCurlyBraces
sonar.issue.ignore.multicriteria.e5.resourceKey=**
# Control flow statements "if", "switch", "try" and iterators should not be nested too deeply
sonar.issue.ignore.multicriteria.e6.ruleKey=cxx:NestedStatements
sonar.issue.ignore.multicriteria.e6.resourceKey=**
# Undocumented API
sonar.issue.ignore.multicriteria.e7.ruleKey=cxx:UndocumentedApi
sonar.issue.ignore.multicriteria.e7.resourceKey=**
# Functions should not be too complex
sonar.issue.ignore.multicriteria.e8.ruleKey=cxx:FunctionComplexity
sonar.issue.ignore.multicriteria.e8.resourceKey=**
# Functions, methods and lambdas should not have too many parameters
sonar.issue.ignore.multicriteria.e9.ruleKey= cxx:TooManyParameters
sonar.issue.ignore.multicriteria.e9.resourceKey=**
# Extra care should be taken to ensure that character arrays that are allocated on the stack are used safely. They are prime targets for buffer overflow attacks.
sonar.issue.ignore.multicriteria.e10.ruleKey=rats:fixed size global buffer
sonar.issue.ignore.multicriteria.e10.resourceKey=**
# Double check that your buffer is as big as you specify. When using functions that accept a number n of bytes to copy, such as strncpy, be aware that if the dest buffer size = n it may not NULL-terminate the string.
sonar.issue.ignore.multicriteria.e11.ruleKey=rats:snprintf
sonar.issue.ignore.multicriteria.e11.resourceKey=**
# Double check that your buffer is as big as you specify. When using functions that accept a number n of bytes to copy, such as strncpy, be aware that if the dest buffer size = n it may not NULL-terminate the string.
sonar.issue.ignore.multicriteria.e12.ruleKey=rats:memcpy
sonar.issue.ignore.multicriteria.e12.resourceKey=**
# Source files should have a sufficient density of comment lines
sonar.issue.ignore.multicriteria.e13.ruleKey=common-c++:InsufficientCommentDensity
sonar.issue.ignore.multicriteria.e13.resourceKey=**
# Files should not be too complex
sonar.issue.ignore.multicriteria.e14.ruleKey=cxx:FileComplexity
sonar.issue.ignore.multicriteria.e14.resourceKey=**
# Standard random number generators should not be used to generate randomness used for security reasons. For security sensitive randomness a cryptographic randomness generator that provides sufficient entropy should be used.
sonar.issue.ignore.multicriteria.e15.ruleKey=rats:random
sonar.issue.ignore.multicriteria.e15.resourceKey=**
# Statements should be on separate lines
sonar.issue.ignore.multicriteria.e16.ruleKey=cxx:TooManyStatementsPerLine
sonar.issue.ignore.multicriteria.e16.resourceKey=**
# Tabulation characters should not be used
sonar.issue.ignore.multicriteria.e17.ruleKey=cxx:TabCharacter
sonar.issue.ignore.multicriteria.e17.resourceKey=**
# Function names should comply with a naming convention
sonar.issue.ignore.multicriteria.e18.ruleKey=cxx:FunctionName
sonar.issue.ignore.multicriteria.e18.resourceKey=**
# C++ Parser can't read code. Declaration is skipped
sonar.issue.ignore.multicriteria.e19.ruleKey=cxx:ParsingErrorRecovery
sonar.issue.ignore.multicriteria.e19.resourceKey=**
EOF

# Run the sonar-scanner analysis and submit to SonarQube server
echo "Run sonar-scanner ..."
time sonar-scanner -X > sonar.log
tail -20 sonar.log

#x=$(grep -c "ANALYSIS SUCCESSFUL" sonar.log)
#if test "$x" == "0"
#then
#    exit 1
#fi
