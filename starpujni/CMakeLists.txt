# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2022-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
cmake_minimum_required(VERSION 3.8)
project(STARPUJNI Java C)

add_custom_target(build-all)

include(${CMAKE_CURRENT_BINARY_DIR}/MavenSettings.cmake)

set(CMAKE_MODULE_PATH ${STARPUJNI_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

foreach(var JARFILE HEADERS_DIR LIBRARY_OUTPUT_DIR LAUNCHER_CLASS)
    if(NOT DEFINED STARPUJNI_${var})
        message(FATAL_ERROR "missing variable STARPUJNI_${var}")
    endif()
endforeach()


foreach(var HADOOP_COMMON_HOME HADOOP_COMMON_LIB_NATIVE_DIR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "missing variable ${var}")
    endif()
endforeach()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if (NOT DEFINED HADOOP_VERSION)
    set (STARPUJNI_HADOOP_MAJOR 3)
    set (STARPUJNI_HADOOP_MINOR 1)
else()
    string(REPLACE "." ";" VERSION_LIST ${HADOOP_VERSION})
    list(GET VERSION_LIST 0 STARPUJNI_HADOOP_MAJOR)
    list(GET VERSION_LIST 1 STARPUJNI_HADOOP_MINOR)
endif()
add_compile_definitions(HADOOP_MAJOR=${HADOOP_MAJOR}
                        HADOOP_MINOR=${HADOOP_MINOR})

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS_DEBUG "-ggdb -Wall")
#-pedantic")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-gnu-statement-expression")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-gnu-zero-variadic-macro-arguments")
message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message("CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")
message("CMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}")

find_package(JNI REQUIRED)

enable_testing()
add_subdirectory(src/main/native)
add_subdirectory(src/test)

