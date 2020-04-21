function variable(val ::Ref{Float32}) :: Nothing
    val[] = val[] + 1

    return
end

function variable_without_starpu(val ::Ref{Float32}, niter)
    for i = 1:niter
        variable(val)
    end
end

function display(niter)
    foo = Ref(0.0f0)

    variable_without_starpu(foo, niter)

    println("variable -> ", foo[])
    if foo[] == niter
        println("result is correct")
    else
        println("result is incorret")
    end
end

display(10)
