import Libdl
using StarPU
using LinearAlgebra

function multiply_without_starpu(A :: Matrix{Float32}, B :: Matrix{Float32}, C :: Matrix{Float32}, nslicesx, nslicesy, stride)
    tmin = 0
    for i in (1 : 10 )
        t=time_ns()
        C = A * B;
        t=time_ns() - t
        if (tmin==0 || tmin>t)
            tmin=t
        end
    end
    return tmin
end


function compute_times(io,start_dim, step_dim, stop_dim, nslicesx, nslicesy, stride)
    for dim in (start_dim : step_dim : stop_dim)
        A = Array(rand(Cfloat, dim, dim))
        B = Array(rand(Cfloat, dim, dim))
        C = zeros(Float32, dim, dim)
        mt =  multiply_without_starpu(A, B, C, nslicesx, nslicesy, stride)
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

