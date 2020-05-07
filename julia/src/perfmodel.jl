function starpu_perfmodel(; perf_type::starpu_perfmodel_type, symbol::String)
    output = starpu_perfmodel(zero)
    output.type = perf_type
    output.symbol = Cstring_from_String(symbol)

    # Performance models must not be garbage collected before starpu_shutdown
    # is called.
    push!(perfmodels, output)

    return output
end
