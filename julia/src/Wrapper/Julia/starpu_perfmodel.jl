
export StarpuPerfmodelType
export STARPU_PERFMODEL_INVALID, STARPU_PER_ARCH, STARPU_COMMON
export STARPU_HISTORY_BASED, STARPU_REGRESSION_BASED
export STARPU_NL_REGRESSION_BASED, STARPU_MULTIPLE_REGRESSION_BASED

@enum(StarpuPerfmodelType,
    STARPU_PERFMODEL_INVALID = 0,
	STARPU_PER_ARCH = 1,
	STARPU_COMMON = 2,
	STARPU_HISTORY_BASED = 3,
	STARPU_REGRESSION_BASED = 4,
	STARPU_NL_REGRESSION_BASED = 5,
	STARPU_MULTIPLE_REGRESSION_BASED = 6
)


mutable struct StarpuPerfmodel_c

    perf_type :: StarpuPerfmodelType

    cost_function :: Ptr{Void}
    arch_cost_function :: Ptr{Void}

    size_base :: Ptr{Void}
    footprint :: Ptr{Void}

    symbol :: Cstring

    is_loaded :: Cuint
    benchmarking :: Cuint
    is_init :: Cuint

    parameters :: Ptr{Void}
    parameters_names :: Ptr{Void}
    nparameters :: Cuint
    combinations :: Ptr{Void}
    ncombinations :: Cuint

    state :: Ptr{Void}


    function StarpuPerfmodel_c()

        output = new()
        jlstarpu_set_to_zero(output)

        return output
    end

end



export StarpuPerfmodel
struct StarpuPerfmodel

    perf_type :: StarpuPerfmodelType
    symbol :: String

    c_perfmodel :: Ptr{StarpuPerfmodel_c}
end




function StarpuPerfmodel(; perf_type = STARPU_PERFMODEL_INVALID, symbol = "")

    if (perf_type == STARPU_PERFMODEL_INVALID)
        return StarpuPerfmodel(perf_type, symbol, Ptr{StarpuPerfmodel_c}(C_NULL))
    end

    if (isempty(symbol))
        error("Field \"symbol\" can't be empty when creating a StarpuPerfmodel")
    end

    c_perfmodel = StarpuPerfmodel_c()
    c_perfmodel.perf_type = perf_type
    c_perfmodel.symbol = Cstring_from_String(symbol)

    c_perfmodel_ptr = jlstarpu_allocate_and_store(c_perfmodel)

    return StarpuPerfmodel(perf_type, symbol, c_perfmodel_ptr)
end


function show_c_perfmodel(x :: StarpuPerfmodel)
    x_c = unsafe_load(x.c_perfmodel)
    println(x_c)
end
