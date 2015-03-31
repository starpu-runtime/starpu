# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2013  Centre National de la Recherche Scientifique
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

s/\bstarpu_canonical_block_filter_bcsr\b/starpu_bcsr_filter_canonical_block/g
s/\bstarpu_vertical_block_filter_func_csr\b/starpu_csr_filter_vertical_block/g

s/\bstarpu_block_filter_func\b/starpu_matrix_filter_block/g
s/\bstarpu_block_shadow_filter_func\b/starpu_matrix_filter_block_shadow/g
s/\bstarpu_vertical_block_filter_func\b/starpu_matrix_filter_vertical_block/g
s/\bstarpu_vertical_block_shadow_filter_func\b/starpu_matrix_filter_vertical_block_shadow/g

s/\bstarpu_block_filter_func_vector\b/starpu_vector_filter_block/g
s/\bstarpu_block_shadow_filter_func_vector\b/starpu_vector_filter_block_shadow/g
s/\bstarpu_vector_list_filter_func\b/starpu_vector_filter_list/g
s/\bstarpu_vector_divide_in_2_filter_func\b/starpu_vector_filter_divide_in_2/g

s/\bstarpu_block_filter_func_block\b/starpu_block_filter_block/g
s/\bstarpu_block_shadow_filter_func_block\b/starpu_block_filter_block_shadow/g
s/\bstarpu_vertical_block_filter_func_block\b/starpu_block_filter_vertical_block/g
s/\bstarpu_vertical_block_shadow_filter_func_block\b/starpu_block_filter_vertical_block_shadow/g
s/\bstarpu_depth_block_filter_func_block\b/starpu_block_filter_depth_block/g
s/\bstarpu_depth_block_shadow_filter_func_block\b/starpu_block_filter_depth_block_shadow/g
