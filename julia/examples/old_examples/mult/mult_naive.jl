function multjl(A ::Matrix{Float32}, B ::Matrix{Float32}, C ::Matrix{Float32})
    heightC, widthC = size(C)
    widthA = size(A)[2]
    for i = 1:heightC
        for j = 1:widthC
            sum = 0
            for k = 1:widthA
                sum = sum + A[i, k] * B[k, j]
            end
            C[i,j] = sum
        end
    end
end