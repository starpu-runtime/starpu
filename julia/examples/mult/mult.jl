import Libdl
using StarPU
using LinearAlgebra

@target STARPU_CPU+STARPU_CUDA
@codelet function matrix_mult(m1 :: Matrix{Float32}, m2 :: Matrix{Float32}, m3 :: Matrix{Float32}) :: Nothing

    width_m2 :: Int32 = width(m2)
    height_m1 :: Int32 = height(m1)
    width_m1 :: Int32 = width(m1)
    # Naive version
    @parallel for j in (1 : width_m2)
       @parallel for i in (1 : height_m1)
    
             sum :: Float32 = 0.

             for k in (1 : width_m1)
                 sum = sum + m1[i, k] * m2[k, j]
             end
    
             m3[i, j] = sum
         end
     end
    # ##### Tiled and unrolled version 
    # for l in (1 : width_m2)
    #     for m in (1 : height_m1)
    #         m3[m,l] = 0
    #     end
    # end
    # @parallel for i in (1 : STRIDE : height_m1)
    #     for k in (1 : STRIDE : width_m1 )
    #         for j in (1 : STRIDE : width_m2  )
    #             for kk in (k : 4 : k+STRIDE-1)
    #                 for jj in (j : 2 : j+STRIDE-1)
    #                     alpha00 :: Float32 =m2[kk,jj]
    #                     alpha01 :: Float32 =m2[kk,jj+1]
    #                     alpha10 :: Float32 =m2[kk+1,jj]
    #                     alpha11 :: Float32 =m2[kk+1,jj+1]
    #                     alpha20 :: Float32 =m2[kk+2,jj]
    #                     alpha21 :: Float32 =m2[kk+2,jj+1]
    #                     alpha30 :: Float32 =m2[kk+3,jj]
    #                     alpha31 :: Float32 =m2[kk+3,jj+1]
    #                     for ii in (i : 1 : i+STRIDE-1) 
    #                         m3[ii, jj] = m3[ii, jj] + m1[ii, kk] * alpha00 + m1[ii, kk+1] * alpha10 + m1[ii, kk+2] * alpha20 + m1[ii,kk+3]*alpha30
    #                         m3[ii, jj+1] = m3[ii, jj+1] + m1[ii, kk] * alpha01 + m1[ii, kk+1] * alpha11 + m1[ii, kk+2]*alpha21 + m1[ii,kk+3]*alpha31 
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end

    return
end


starpu_init()

function multiply_with_starpu(A :: Matrix{Float32}, B :: Matrix{Float32}, C :: Matrix{Float32}, nslicesx, nslicesy, stride)
    scale= 3
    tmin=0
    vert = StarpuDataFilter(STARPU_MATRIX_FILTER_VERTICAL_BLOCK, nslicesx)
    horiz = StarpuDataFilter(STARPU_MATRIX_FILTER_BLOCK, nslicesy)
    @starpu_block let
        hA,hB,hC = starpu_data_register(A, B, C)
        starpu_data_partition(hB, vert)
        starpu_data_partition(hA, horiz)
        starpu_data_map_filters(hC, vert, horiz)
        tmin=0
        perfmodel = StarpuPerfmodel(
            perf_type = STARPU_HISTORY_BASED,
            symbol = "history_perf"
        )
        cl = StarpuCodelet(
            cpu_func = CPU_CODELETS["matrix_mult"],
            # cuda_func = CUDA_CODELETS["matrix_mult"],
            #opencl_func="ocl_matrix_mult",
            modes = [STARPU_R, STARPU_R, STARPU_W],
            perfmodel = perfmodel
        )

        for i in (1 : 10 )
            t=time_ns()
            @starpu_sync_tasks begin
                for taskx in (1 : nslicesx)
                    for tasky in (1 : nslicesy)
                        handles = [hA[tasky], hB[taskx], hC[taskx, tasky]]
                        task = StarpuTask(cl = cl, handles = handles, cl_arg=(stride))
                        starpu_task_submit(task)
                        #@starpu_async_cl matrix_mult(hA[tasky], hB[taskx], hC[taskx, tasky])
                    end
                end
            end
            t=time_ns()-t
            if (tmin==0 || tmin>t)
                tmin=t
            end
        end
    end
    return tmin
end


function approximately_equals(
    A :: Matrix{Cfloat},
    B :: Matrix{Cfloat},
    eps = 1e-2
)
    (height, width) = size(A)

    for j in (1 : width)
        for i in (1 : height)
            if (abs(A[i,j] - B[i,j]) > eps * max(abs(B[i,j]), abs(A[i,j])))
                println("A[$i,$j] : $(A[i,j]), B[$i,$j] : $(B[i,j])")
                return false
            end
        end
    end

    return true
end

function compute_times(io,start_dim, step_dim, stop_dim, nslicesx, nslicesy, stride)
    for dim in (start_dim : step_dim : stop_dim)
        A = Array(rand(Cfloat, dim, dim))
        B = Array(rand(Cfloat, dim, dim))
        C = zeros(Float32, dim, dim)
        mt =  multiply_with_starpu(A, B, C, nslicesx, nslicesy, stride)
        flops = (2*dim-1)*dim*dim/mt
        size=dim*dim*4*3/1024/1024
        println(io,"$size $flops")
        println("$size $flops")
    end
end

if size(ARGS, 1) < 2
    stride=4
    filename="x.dat"
else
    stride=parse(Int, ARGS[1])
    filename=ARGS[2]
end
io=open(filename,"w")
compute_times(io,16*stride,4*stride,128*stride,2,2,stride)
close(io)

starpu_shutdown()

