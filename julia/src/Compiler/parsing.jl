

#======================================================
                GLOBAL PARSING
======================================================#



starpu_parse_key_word_parsing_function = Dict{Symbol, Function}()


function starpu_parse(x :: Expr)

    if (x.head == :macrocall)

        if (x.args[1] != Symbol("@indep"))
            error("Only @indep macro, used before a for loop, is allowed ($(x.args[1]) was found)")
        end

        if (length(x.args) != 2)
            error("Invalid usage of @indep macro")
        end

        return starpu_parse_for(x.args[2], is_independant = true)
    end


    if !(x.head in keys(starpu_parse_key_word_parsing_function))
        return StarpuExprInvalid() #TODO error ?
    end

    return starpu_parse_key_word_parsing_function[x.head](x)

end

for kw in (:if, :call, :for, :block, :return, :function, :while, :ref)
    starpu_parse_key_word_parsing_function[kw] = eval(Symbol(:starpu_parse_, kw))
end

starpu_parse_key_word_parsing_function[:(:)] = starpu_parse_interval
starpu_parse_key_word_parsing_function[:(::)] = starpu_parse_typed
starpu_parse_key_word_parsing_function[:(=)] = starpu_parse_affect
starpu_parse_key_word_parsing_function[:(.)] = starpu_parse_field



macro parse(x)
    y = Expr(:quote, x)
    :(starpu_parse($y))
end
