#!/bin/bash

export CPPCHECK_INCLUDES="-Iinclude -Impi/include"
export SOURCES_TO_ANALYZE="include src mpi/src tools tests examples mpi/tests mpi/examples"
export SONAR_SOURCES=$(echo $SOURCES_TO_ANALYZE | tr ' ' ',')
export SOURCES_TO_EXCLUDE="-itools/dev"

if test -z "$SONAR_LOGIN_FILE"
then
    echo "Error. Environment variable SONAR_LOGIN_FILE missing"
    exit 1
fi

SONAR_PROJECT=storm:starpu_trunk
SONAR_LOGIN=$(grep ${SONAR_PROJECT} ${SONAR_LOGIN_FILE} | awk '{print $2}')
if test -z "$SONAR_LOGIN"
then
    echo "Error. Project ${SONAR_PROJECT} not available in file ${SONAR_LOGIN_FILE}"
    exit 1
fi

if test -f build/config.log
then
    CONFIG_LOG=build/config.log
elif test -f config.log
then
    CONFIG_LOG=config.log
else
    echo Error no config.log file found
    exit 1
fi

export DEFINITIONS_LOG=$(grep "^#define" ${CONFIG_LOG} | sed -e "s#\#define #-D#g" | sed -e "s# #=#g" | xargs)
export DEFINITIONS_SRC=$(grep "^#undef" src/common/config.h.in| sed -e "s#\#undef #-D#g" | sed -e "s#\$#=1#g" | xargs)
export DEFINITIONS_LOCAL=$(grep -rs "#ifdef" src/ mpi/src |awk -F':' '{print $2}' | awk '{print "-D"$2"=1"}' |sort|uniq)
export DEFINITIONS="${DEFINITIONS_LOG} ${DEFINITIONS_SRC} ${DEFINITIONS_LOCAL}"

cppcheck --max-configs=1000 --language=c++ --platform=unix64 --force -v --enable=all --inline-suppr --xml --xml-version=2 ${DEFINITIONS} ${CPPCHECK_INCLUDES} ${SOURCES_TO_EXCLUDE} ${SOURCES_TO_ANALYZE} 2> cppcheck.xml
#--suppressions-list=tools/cppcheck/suppressions.txt

sonar-scanner -Dsonar.login=${SONAR_LOGIN} -Dsonar.projectKey=${SONAR_PROJECT} -Dsonar.projectName=StarPU_trunk -Dsonar.projectVersion=HEAD -Dsonar.sources=${SONAR_SOURCES} -Dsonar.cxx.cppcheck.reportPath=cppcheck.xml
