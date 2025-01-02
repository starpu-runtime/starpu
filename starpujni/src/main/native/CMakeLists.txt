# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2022-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#
# COMPILATION OF STARPU BINDINGS
#
include(CheckIncludeFile)

find_package(PkgConfig REQUIRED)
pkg_check_modules(STARPU_LIB REQUIRED starpu-1.3)

if(NOT DEFINED HDFS_INCLUDE_DIR)
    if(DEFINED ENV{HDFS_INCLUDE_DIR})
        set(HDFS_INCLUDE_DIR $ENV{HDFS_INCLUDE_DIR})
    else()
        set(HDFS_INCLUDE_DIR "${HADOOP_COMMON_HOME}/include")
    endif()
endif()

if(NOT DEFINED HDFS_LIBRARY_DIR)
    if(DEFINED ENV{HDFS_LIBRARY_DIR})
        set(HDFS_LIBRARY_DIR $ENV{HDFS_LIBRARY_DIR})
    else()
        set(HDFS_LIBRARY_DIR "${HADOOP_COMMON_HOME}/${HADOOP_COMMON_LIB_NATIVE_DIR}")
    endif()
endif()

set(CMAKE_REQUIRED_INCLUDES "${HDFS_INCLUDE_DIR}")
CHECK_INCLUDE_FILE("hdfs.h" HAVE_HDFS_H) # "-I${HDFS_INCLUDE_DIR}")

if(NOT HAVE_HDFS_H)
    unset(HAVE_HDFS_H CACHE)
    message( FATAL_ERROR "hdfs.h is not found" )
endif()

find_library(LIBHDFS NAMES hdfs PATHS "${HDFS_LIBRARY_DIR}")
if(${LIBHDFS} STREQUAL "LIBHDFS-NOTFOUND")
    message(FATAL_ERROR "cannot find HDFS library")
endif()

message("LIBHDFS PATH=${HDFS_LIBRARY_DIR}")
message("LIBHDFS=${LIBHDFS}")

set(starpujni_SOURCES
        starpujni.c
        starpujni-codelet.c starpujni-codelet.h
        starpujni-data.c starpujni-data.h
        starpujni-common.c starpujni-common.h
        starpujni-native-tests.c
        starpujni-jobject-vector.c starpujni-jobject-vector.h
        starpujni-hdfs.c starpujni-hdfs.h
        )


add_library(starpujni MODULE "${starpujni_SOURCES}")
add_dependencies(build-all starpujni)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_target_properties(starpujni PROPERTIES SUFFIX ".jnilib")
endif ()

set_target_properties(starpujni PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        "${STARPUJNI_LIBRARY_OUTPUT_DIR}")
target_link_libraries(starpujni ${JNI_LIBRARIES})
target_link_libraries(starpujni ${STARPU_LIB_LDFLAGS})
target_link_libraries(starpujni ${LIBHDFS})

target_include_directories(starpujni PRIVATE
        "${JNI_INCLUDE_DIRS}"
        "${STARPUJNI_HEADERS_DIR}"
        "${STARPU_LIB_INCLUDE_DIRS}"
        "${HDFS_INCLUDE_DIR}"
        )

target_compile_options(starpujni PRIVATE ${STARPU_LIB_CFLAGS})

set(NATIVELIBINFO_INPUT
    "${STARPUJNI_NATIVELIBINFO_SRCDIR}/${STARPUJNI_NATIVELIBINFO_JAVA}.in")
set(NATIVELIBINFO_OUTPUT
    "${STARPUJNI_NATIVELIBINFO_DESTDIR}/${STARPUJNI_NATIVELIBINFO_JAVA}")

file(GENERATE OUTPUT "${NATIVELIBINFO_OUTPUT}" INPUT "${NATIVELIBINFO_INPUT}")
