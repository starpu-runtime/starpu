include("../../src/Compiler/include.jl")

starpu_new_cpu_kernel_file("../build/generated_cpu_nbody.c")
starpu_new_cuda_kernel_file("../build/generated_cuda_nbody.cu")

@cpu_cuda_kernel function nbody_acc(positions ::Matrix{Float64}, accelerations ::Matrix{Float64}, masses ::Vector{Float64}, parameters ::Vector{Float64}, sliceID ::Vector{Int64}) ::Void
    widthp ::Int64 = width(positions)
    widtha ::Int64 = width(accelerations)
    
    @indep for plan = 1:widtha

        sumaccx ::Float64 = 0
        sumaccy ::Float64 = 0

        for oplan = 1:widthp
            eps ::Float64 = parameters[3]
            Id ::Int64 = sliceID[1]*widtha
            G ::Float64 = parameters[1]

            b ::Int64 = ((plan + Id) >= oplan) + ((plan + Id) <= oplan)
            if (b < 2)

                dx ::Float64 = positions[1, oplan] - positions[1, plan + Id]
                dy ::Float64 = positions[2, oplan] - positions[2, plan + Id]
                modul ::Float64= sqrt(dx *dx + dy * dy)

                sumaccx = sumaccx + (G * masses[oplan] * dx) / ((modul + eps) * (modul + eps) * (modul + eps)) 
                sumaccy = sumaccy + (G * masses[oplan] * dy) / ((modul + eps) * (modul + eps) * (modul + eps))

                # sumaccx = sumaccx + (G * masses[oplan]) * (dx / sqrt(dx * dx + dy * dy)) / (dx * dx + dy * dy + eps)
                # sumaccy = sumaccy + (G * masses[oplan]) * (dy / sqrt(dx * dx + dy * dy)) / (dy * dy + dx * dx + eps)
            end
        end
        accelerations[1, plan] = sumaccx
        accelerations[2, plan] = sumaccy
    end
end

@cpu_cuda_kernel function nbody_updt(positions ::Matrix{Float64}, velocities ::Matrix{Float64}, accelerations ::Matrix{Float64}, parameters ::Vector{Float64}) ::Void
    widthp ::Int64 = width(positions)

    @indep for i = 1:widthp

        velocities[1, i] = velocities[1, i] + accelerations[1, i] * parameters[2]
        velocities[2, i] = velocities[2, i] + accelerations[2, i] * parameters[2]

        positions[1, i] = positions[1, i] + velocities[1, i] * parameters[2]
        positions[2, i] = positions[2, i] + velocities[2, i] * parameters[2]
    end

end

compile_cpu_kernels("../build/generated_cpu_nbody.so")
compile_cuda_kernels("../build/generated_cuda_nbody.so")
combine_kernel_files("../build/generated_tasks_nbody.so", ["../build/generated_cpu_nbody.so", "../build/generated_cuda_nbody.so"])