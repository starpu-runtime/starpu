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
set(jvm_launcher_SOURCES jvm-launcher.c)

add_custom_target(native-testers DEPENDS jvm-launcher)
add_dependencies(build-all native-testers)

add_executable(jvm-launcher ${jvm_launcher_SOURCES})

string(REPLACE "." "/" STARPUJNI_LAUNCHER_CLASS "${STARPUJNI_LAUNCHER_CLASS}")

target_compile_definitions(jvm-launcher PUBLIC
        "STARPUJNI_JARFILE=\"${STARPUJNI_JARFILE}\""
        "STARPUJNI_LAUNCHER_CLASS=\"${STARPUJNI_LAUNCHER_CLASS}\"")

include_directories(jvm-launcher "${JNI_INCLUDE_DIRS}")
target_link_libraries(jvm-launcher "${JNI_LIBRARIES}")
