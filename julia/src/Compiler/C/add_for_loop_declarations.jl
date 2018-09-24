


function interval_evaluation_declarations(set :: StarpuExprInterval, for_index_var :: Symbol)

    const decl_pattern = @parse € :: Int64
    const affect_pattern = @parse € :: Int64 = €
    const interv_size_affect_pattern = @parse € :: Int64 = jlstarpu_interval_size(€, €, €)

    id = set.id

    start_var = starpu_parse(Symbol(:start_, id))
    start_decl = replace_pattern(affect_pattern, start_var, set.start)

    step_var = starpu_parse(Symbol(:step_, id))
    step_decl = replace_pattern(affect_pattern, step_var, set.step)

    dim_var = starpu_parse(Symbol(:dim_, id))
    dim_decl = replace_pattern(interv_size_affect_pattern, dim_var, start_var, step_var, set.stop)

    iter_var = starpu_parse(Symbol(:iter_, id))
    iter_decl = replace_pattern(decl_pattern, iter_var)

    index_var = starpu_parse(for_index_var)
    index_decl = replace_pattern(decl_pattern, index_var)


    return StarpuExpr[start_decl, step_decl, dim_decl, iter_decl, index_decl]
end



function add_for_loop_declarations(expr :: StarpuExpr)

    function func_to_apply(x :: StarpuExpr)

        if !isa(x, StarpuExprFor)
            return x
        end

        interval_decl = interval_evaluation_declarations(x.set, x.iter)

        return StarpuExprFor(x.iter, x.set, x.body, x.is_independant, interval_decl)
    end

    return apply(func_to_apply, expr)
end
