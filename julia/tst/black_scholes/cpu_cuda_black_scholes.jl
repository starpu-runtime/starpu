include("../../src/Compiler/include.jl")

starpu_new_cpu_kernel_file("../build/generated_cpu_black_scholes.c")
starpu_new_cuda_kernel_file("../build/generated_cuda_black_scholes.cu")





@cpu_cuda_kernel function black_scholes(data ::Matrix{Float64}, res ::Matrix{Float64}) ::Void
    
    widthn ::Int64 = width(data)
        
    # data[1,...] -> S
    # data[2,...] -> K
    # data[3,...] -> r
    # data[4,...] -> T
    # data[4,...] -> sig

    p ::Float64 = 0.2316419
    b1 ::Float64 = 0.31938153
    b2 ::Float64 = -0.356563782
    b3 ::Float64 = 1.781477937
    b4 ::Float64 = -1.821255978
    b5 ::Float64 = 1.330274428

    
    @indep for i = 1:widthn
        

        d1 ::Float64 = (log(data[1,i] / data[2,i]) + (data[3,i] + pow(data[5,i], 2.0) * 0.5) * data[4,i]) / (data[5,i] * sqrt(data[4,i]))
        d2 ::Float64 = (log(data[1,i] / data[2,i]) + (data[3,i] - pow(data[5,i], 2.0) * 0.5) * data[4,i]) / (data[5,i] * sqrt(data[4,i]))
        



        f ::Float64 = 0
        ff ::Float64 = 0
        s1 ::Float64 = 0
        s2 ::Float64 = 0
        s3 ::Float64 = 0
        s4 ::Float64 = 0
        s5 ::Float64 = 0
        sz ::Float64 = 0
        


        
        ######## Compute normcdf of d1

        normd1p ::Float64 = 0
        normd1n ::Float64 = 0

        boold1 ::Int64 = (d1 >= 0) + (d1 <= 0)
        
        if (boold1 >= 2)
            normd1p = 0.5
            normd1n = 0.5
        else
            tmp1 ::Float64 = abs(d1)
            f = 1 / sqrt(2 * M_PI)
            ff = exp(-pow(tmp1, 2.0) / 2) * f
            s1 = b1 / (1 + p * tmp1)
            s2 = b2 / pow((1 + p * tmp1), 2.0)
            s3 = b3 / pow((1 + p * tmp1), 3.0)
            s4 = b4 / pow((1 + p * tmp1), 4.0)
            s5 = b5 / pow((1 + p * tmp1), 5.0)
            sz = ff * (s1 + s2 + s3 + s4 + s5)
        
            if (d1 > 0)
                normd1p = 1 - sz # normcdf(d1)
                normd1n = sz # normcdf(-d1)
            else
                normd1p = sz
                normd1n = 1 - sz
            end    
        end
        ########
        

        ######## Compute normcdf of d2
        normd2p ::Float64 = 0
        normd2n ::Float64 = 0

        boold2 ::Int64 = (d2 >= 0) + (d2 <= 0)
        
        if (boold2 >= 2)
            normd2p = 0.5
            normd2n = 0.5
        else
            tmp2 ::Float64 = abs(d2)
            f = 1 / sqrt(2 * M_PI)
            ff = exp(-pow(tmp2, 2.0) / 2) * f
            s1 = b1 / (1 + p * tmp2)
            s2 = b2 / pow((1 + p * tmp2), 2.0)
            s3 = b3 / pow((1 + p * tmp2), 3.0)
            s4 = b4 / pow((1 + p * tmp2), 4.0)
            s5 = b5 / pow((1 + p * tmp2), 5.0)
            sz = ff * (s1 + s2 + s3 + s4 + s5)
        
        
            if (d2 > 0)
                normd2p = 1 - sz # normcdf(d2)
                normd2n = sz # normcdf(-d2)
            else
                normd2p = sz
                normd2n = 1 - sz
            end
        end
        # normd1p = (1 + erf(d1/sqrt(2.0)))/2.0
        # normd1n = (1 + erf(-d1/sqrt(2.0)))/2.0
        
        # normd2p = (1 + erf(d2/sqrt(2.0)))/2.0
        # normd2n = (1 + erf(-d2/sqrt(2.0)))/2.0
        
        res[1,i] = data[1,i] * (normd1p) - data[2,i]*exp(-data[3,i]*data[4,i]) * (normd2p) # S * N(d1) - r*exp(-r*T) * norm(d2)
        res[2,i] = -data[1,i] * (normd1n) + data[2,i]*exp(-data[3,i]*data[4,i]) * (normd2n) # -S * N(-d1) + r*exp(-r*T) * norm(-d2)
        
    end
end

compile_cpu_kernels("../build/generated_cpu_black_scholes.so")
compile_cuda_kernels("../build/generated_cuda_black_scholes.so")
combine_kernel_files("../build/generated_tasks_black_scholes.so", ["../build/generated_cpu_black_scholes.so", "../build/generated_cuda_black_scholes.so"])