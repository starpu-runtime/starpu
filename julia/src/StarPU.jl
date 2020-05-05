"""
__precompile__()
"""
module StarPU
import Libdl

const STARPU_MAXIMPLEMENTATIONS = 1 # TODO : These must be the same values as defined in C macros !
const STARPU_NMAXBUFS = 8 # TODO : find a way to make it automatically match

export STARPU_CPU
export STARPU_CUDA
const  STARPU_CPU = 1 << 1
const  STARPU_CUDA = 1 << 3

const starpu_task_library_name="libjlstarpu_c_wrapper.so"
global starpu_tasks_library_handle = C_NULL
global starpu_target=STARPU_CPU

include("compiler/include.jl")

macro starpufunc(symbol)
    :($symbol, starpu_task_library_name)
end

"""
    Used to call a StarPU function compiled inside "libjlstarpu_c_wrapper.so"
    Works as ccall function
"""
macro starpucall(func, ret_type, arg_types, args...)
    return Expr(:call, :ccall, (func, starpu_task_library_name), esc(ret_type), esc(arg_types), map(esc, args)...)
end

function debug_print(x...)
    println("\x1b[32m", x..., "\x1b[0m")
    flush(stdout)
end

function Cstring_from_String(str :: String)
    return Cstring(pointer(str))
end


function jlstarpu_set_to_zero(x :: T) :: Ptr{Cvoid} where {T}
    @starpucall(memset,
          Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t),
          Ref{T}(x), 0, sizeof(x)
        )
end

tuple_len(::NTuple{N, Any}) where {N} = N

export starpu_init
export starpu_shutdown
export starpu_memory_pin
export starpu_memory_unpin
export starpu_data_unregister
export starpu_data_register
export starpu_data_get_sub_data
export StarpuDataFilterFunc
export STARPU_MATRIX_FILTER_VERTICAL_BLOCK, STARPU_MATRIX_FILTER_BLOCK
export StarpuDataFilter
export starpu_data_partition
export starpu_data_unpartition
export starpu_data_map_filters
export @starpu_sync_tasks
export starpu_task_wait_for_all
export @starpu_async_cl
export starpu_task_submit
export @starpu_block
export StarpuPerfmodel
export @starpu_filter
export StarpuPerfmodelType
export STARPU_PERFMODEL_INVALID, STARPU_PER_ARCH, STARPU_COMMON
export STARPU_HISTORY_BASED, STARPU_REGRESSION_BASED
export STARPU_NL_REGRESSION_BASED, STARPU_MULTIPLE_REGRESSION_BASED
export starpu_task_declare_deps
export starpu_task_wait_for_n_submitted
export starpu_task_destroy
export starpu_tag_wait
export starpu_iteration_pop
export starpu_iteration_push
export starpu_tag_declare_deps
export StarpuTask
export StarpuDataAccessMode
export STARPU_NONE,STARPU_R,STARPU_W,STARPU_RW, STARPU_SCRATCH
export STARPU_REDUX,STARPU_COMMUTE, STARPU_SSEND, STARPU_LOCALITY
export STARPU_ACCESS_MODE_MAX
export StarpuCodelet

@enum(StarpuDataAccessMode,
    STARPU_NONE = 0,
    STARPU_R = (1 << 0),
    STARPU_W = (1 << 1),
    STARPU_RW = ((1 << 0) | (1 << 1)),
    STARPU_SCRATCH = (1 << 2),
    STARPU_REDUX = (1 << 3),
    STARPU_COMMUTE = (1 << 4),
    STARPU_SSEND = (1 << 5),
    STARPU_LOCALITY = (1 << 6),
    STARPU_ACCESS_MODE_MAX = (1 << 7)
      )

const jlstarpu_allocated_structures = Vector{Ptr{Cvoid}}([])
@enum(StarpuPerfmodelType,
      STARPU_PERFMODEL_INVALID = 0,
      STARPU_PER_WORKER = 1,
      STARPU_PER_ARCH = 2,
      STARPU_COMMON = 3,
      STARPU_HISTORY_BASED = 4,
      STARPU_REGRESSION_BASED = 5,
      STARPU_NL_REGRESSION_BASED = 6,
      STARPU_MULTIPLE_REGRESSION_BASED = 7
)
mutable struct StarpuPerfmodel_c

    perf_type :: StarpuPerfmodelType

    cost_function :: Ptr{Cvoid}
    arch_cost_function :: Ptr{Cvoid}
    worker_cost_function :: Ptr{Cvoid}

    size_base :: Ptr{Cvoid}
    footprint :: Ptr{Cvoid}

    symbol :: Cstring

    is_loaded :: Cuint
    benchmarking :: Cuint
    is_init :: Cuint

    parameters :: Ptr{Cvoid}
    parameters_names :: Ptr{Cvoid}
    nparameters :: Cuint
    combinations :: Ptr{Cvoid}
    ncombinations :: Cuint

    state :: Ptr{Cvoid}


    function StarpuPerfmodel_c()

        output = new()
        jlstarpu_set_to_zero(output)

        return output
    end

end
struct StarpuPerfmodel

    perf_type :: StarpuPerfmodelType
    symbol :: String

    c_perfmodel :: Ptr{StarpuPerfmodel_c}
end



"""
        Copies x_c to a new allocated memory zone.
        Returns the pointer toward the copied object. Every pointer
        returned by this function will be freed after a call to
        jlstarpu_free_allocated_structures
    """
function jlstarpu_allocate_and_store(x_c :: T) where {T}

    allocated_ptr = Ptr{T}(Libc.malloc(sizeof(T)))

    if (allocated_ptr == C_NULL)
        error("Base.Libc.malloc returned NULL")
    end

    unsafe_store!(allocated_ptr, x_c)
    push!(jlstarpu_allocated_structures, Ptr{Cvoid}(allocated_ptr))

    return allocated_ptr
end


"""
        Frees every pointer allocated by jlstarpu_allocate_and_store
    """
function jlstarpu_free_allocated_structures()
    map(Libc.free, jlstarpu_allocated_structures)
    empty!(jlstarpu_allocated_structures)
    return nothing
end

struct StarpuCodelet
    where_to_execute :: UInt32

    color :: UInt32

    cpu_func :: String
    cuda_func :: String
    opencl_func :: String

    modes :: Vector{StarpuDataAccessMode}

    perfmodel :: StarpuPerfmodel

    c_codelet :: Ptr{Cvoid}

    function StarpuCodelet(;
                           cpu_func :: String = "",
                           cuda_func :: String = "",
                           opencl_func :: String = "",
                           modes :: Vector{StarpuDataAccessMode} = StarpuDataAccessMode[],
                           perfmodel :: StarpuPerfmodel = StarpuPerfmodel(),
                           where_to_execute :: Union{Cvoid, UInt32} = nothing,
                           color :: UInt32 = 0x00000000
                           )

        if (length(modes) > STARPU_NMAXBUFS)
            error("Codelet has too much buffers ($(length(modes)) but only $STARPU_NMAXBUFS are allowed)")
        end

        real_c_codelet_ptr = @starpucall jlstarpu_new_codelet Ptr{Cvoid} ()
        push!(jlstarpu_allocated_structures, real_c_codelet_ptr)

        if (where_to_execute == nothing)
            real_where = ((cpu_func != "") * STARPU_CPU) | ((cuda_func != "") * STARPU_CUDA)
        else
            real_where = where_to_execute
        end

        output = new(real_where, color, cpu_func, cuda_func, opencl_func,modes, perfmodel, real_c_codelet_ptr)

        starpu_c_codelet_update(output)

        return output
    end
end

export StarpuTag
const StarpuTag = UInt64
const StarpuDataHandlePointer = Ptr{Cvoid}

mutable struct Link{T}
    data :: T
    previous :: Union{Nothing, Link{T}}
    next :: Union{Nothing, Link{T}}
    list
    function Link{T}(x :: T, l) where {T}
        output = new()
        output.data = x
        output.previous = Nothing()
        output.next = Nothing()
        output.list = l
        return output
    end
end

export LinkedList
mutable struct LinkedList{T}
    nelement :: Int64
    first :: Union{Nothing, Link{T}}
    last :: Union{Nothing, Link{T}}
    function LinkedList{T}() where {T}
        output = new()
        output.nelement = 0
        output.first = Nothing()
        output.last = Nothing()
        return output
    end
end

export add_to_head!
function add_to_head!(l :: LinkedList{T}, el :: T) where {T}
    new_first = Link{T}(el, l)
    old_first = l.first
    l.first = new_first
    new_first.next = old_first
    if (isnothing(old_first))
        l.last = new_first
    else
        old_first.previous = new_first
    end
    l.nelement += 1
    return new_first
end

export add_to_tail!
function add_to_tail!(l :: LinkedList{T}, el :: T) where {T}
    new_last = Link{T}(el, l)
    old_last = l.last
    l.last = new_last
    new_last.previous = old_last
    if (isnothing(old_last))
        l.first = new_last
    else
        old_last.next = new_last
    end
    l.nelement += 1
    return new_last
end

function LinkedList(v :: Union{Array{T,N}, NTuple{N,T}}) where {N,T}
    output = LinkedList{T}()
    for x in v
        add_to_tail!(output, x)
    end
    return output
end

export remove_link!
function remove_link!(lnk :: Link{T}) where {T}
    if (lnk.list == nothing)
        return lnk.data
    end
    l = lnk.list
    next = lnk.next
    previous = lnk.previous
    if (isnothing(next))
        l.last = previous
    else
        next.previous = previous
    end
    if (isnothing(previous))
        l.first = next
    else
        previous.next = next
    end
    l.nelement -= 1
    lnk.list = nothing
    return lnk.data
end

export is_linked
function is_linked(lnk :: Link)
    return (lnk.list != nothing)
end

export foreach_asc
macro foreach_asc(list, lnk_iterator, expression)
    quote
        $(esc(lnk_iterator)) = $(esc(list)).first
        while (!isnothing($(esc(lnk_iterator))))
            __next_lnk_iterator = $(esc(lnk_iterator)).next
            $(esc(expression))
            $(esc(lnk_iterator)) = __next_lnk_iterator
        end
    end
end

export foreach_desc
macro foreach_desc(list, lnk_iterator, expression)
    quote
        $(esc(lnk_iterator)) = $(esc(list)).last
        while (!isnothing($(esc(lnk_iterator))))
            __next_lnk_iterator = $(esc(lnk_iterator)).previous
            $(esc(expression))
            $(esc(lnk_iterator)) = __next_lnk_iterator
        end
    end
end

function Base.show(io :: IO, lnk :: Link{T}) where {T}
    print(io, "Link{$T}{data: ")
    print(io, lnk.data)
    print(io, " ; previous: ")
    if (isnothing(lnk.previous))
        print(io, "NONE")
    else
        print(io, lnk.previous.data)
    end
    print(io, " ; next: ")
    if (isnothing(lnk.next))
        print(io, "NONE")
    else
        print(io, lnk.next.data)
    end
    print(io, "}")
end

function Base.show(io :: IO, l :: LinkedList{T}) where {T}
    print(io, "LinkedList{$T}{")
    @foreach_asc l lnk begin
        if (!isnothing(lnk.previous))
            print(io, ", ")
        end
        print(io, lnk.data)
    end
    print(io, "}")
end

#import Base.start
function start(l :: LinkedList)
    return nothing
end

#import Base.done
function done(l :: LinkedList, state)
    if (state == nothing)
        return isnothing(l.first)
    end
    return isnothing(state.next)
end

#import Base.next
function next(l :: LinkedList, state)
    if (state == nothing)
        next_link = l.first
    else
        next_link = state.next
    end
    return (next_link.data, next_link)
end

#import Base.endof
function endof(l :: LinkedList)
    return l.nelement
end

export index_to_link
function index_to_link(l :: LinkedList, ind)
    if (ind > l.nelement || ind <= 0)
        error("Invalid index")
    end
    lnk = l.first
    for i in (1:(ind - 1))
        lnk = lnk.next
    end
    return lnk
end

import Base.getindex
function getindex(l :: LinkedList, ind)
    return index_to_link(l,ind).data
end

import Base.setindex!
function setindex!(l :: LinkedList{T}, ind, value :: T) where T
    lnk = index_to_link(l,ind)
    lnk.data = value
end

import Base.eltype
function eltype(l :: LinkedList{T}) where T
    return T
end

import Base.isempty
function isempty(l :: LinkedList)
    return (l.nelement == 0)
end

import Base.empty!
function empty!(l :: LinkedList)
    @foreach_asc l lnk remove_link!(lnk)
end

import Base.length
function length(l :: LinkedList)
    return l.nelement
end


"""
        Object used to store a lot of function which must
        be applied to and object
    """
mutable struct StarpuDestructible{T}

    object :: T
    destructors :: LinkedList{Function}

end

StarpuDataHandle = StarpuDestructible{StarpuDataHandlePointer}

mutable struct StarpuTask

    cl :: StarpuCodelet
    handles :: Vector{StarpuDataHandle}
    handle_pointers :: Vector{StarpuDataHandlePointer}
    synchronous :: Bool
    cl_arg # type depends on codelet

    c_task :: Ptr{Cvoid}


    """
        StarpuTask(; cl :: StarpuCodelet, handles :: Vector{StarpuDataHandle}, cl_arg :: Ref)

        Creates a new task which will run the specified codelet on handle buffers and cl_args data
    """
    function StarpuTask(; cl :: Union{Cvoid, StarpuCodelet} = nothing, handles :: Vector{StarpuDataHandle} = StarpuDataHandle[], cl_arg = ())

        if (cl == nothing)
            error("\"cl\" field can't be empty when creating a StarpuTask")
        end

        output = new()

        output.cl = cl
        output.handles = handles

        # handle scalar_parameters
        codelet_name = cl.cpu_func
        if isempty(codelet_name)
            codelet_name = cl.cuda_func
        end
        if isempty(codelet_name)
            codelet_name = cl.opencl_func
        end
        if isempty(codelet_name)
            error("No function provided with codelet.")
        end
        scalar_parameters = get(CODELETS_SCALARS, codelet_name, nothing)
        if scalar_parameters != nothing
            nb_scalar_required = length(scalar_parameters)
            nb_scalar_provided = tuple_len(cl_arg)
            if (nb_scalar_provided != nb_scalar_required)
                error("$nb_scalar_provided scalar parameters provided but $nb_scalar_required are required by $codelet_name.")
            end
            output.cl_arg = create_param_struct_from_clarg(codelet_name, cl_arg)
        else
            output.cl_arg = cl_arg
        end

        output.synchronous = false
        output.handle_pointers = StarpuDataHandlePointer[]

        c_task = @starpucall starpu_task_create Ptr{Cvoid} ()

        if (c_task == C_NULL)
            error("Couldn't create new task: starpu_task_create() returned NULL")
        end

        output.c_task = c_task

        starpu_c_task_update(output)

        return output
    end

end

function create_param_struct_from_clarg(name, cl_arg)
    struct_params_name = CODELETS_PARAMS_STRUCT[name]

    if struct_params_name == false
        error("structure name not found in CODELET_PARAMS_STRUCT")
    end

    nb_scalar_provided = length(cl_arg)
    create_struct_param_str = "output = $struct_params_name("
    for i in 1:nb_scalar_provided-1
        arg = cl_arg[i]
        create_struct_param_str *= "$arg, "
        end
    if (nb_scalar_provided > 0)
        arg = cl_arg[nb_scalar_provided]
        create_struct_param_str *= "$arg"
    end
    create_struct_param_str *= ")"
    eval(Meta.parse(create_struct_param_str))
    return output
end

"""
    Structure used to update fields of the real C task structure 
"""
mutable struct StarpuTaskTranslator

    cl :: Ptr{Cvoid}
    handles :: Ptr{Cvoid}
    synchronous :: Cuint

    cl_arg :: Ptr{Cvoid}
    cl_arg_size :: Csize_t

    function StarpuTaskTranslator(task :: StarpuTask)

        output = new()

        output.cl = task.cl.c_codelet

        task.handle_pointers = map((x -> x.object), task.handles)
        output.handles = pointer(task.handle_pointers)
        output.synchronous = Cuint(task.synchronous)

        if (task.cl_arg == nothing)
            output.cl_arg = C_NULL
            output.cl_arg_size = 0
        else
            output.cl_arg = Base.unsafe_convert(Ptr{Cvoid}, Ref(task.cl_arg))
            output.cl_arg_size = sizeof(task.cl_arg)
        end

        return output
    end

end


starpu_block_list = Vector{LinkedList{StarpuDestructible}}()
"""
    Declares a block of code. Every declared StarpuDestructible in this code
    will execute its destructors on its object, once the block is exited
"""
macro starpu_block(expr)
    quote
        starpu_enter_new_block()
        local z=$(esc(expr))
        starpu_exit_block()
        z
    end
end

@enum(StarpuDataFilterFunc,
      STARPU_MATRIX_FILTER_VERTICAL_BLOCK = 0,
      STARPU_MATRIX_FILTER_BLOCK = 1,
      STARPU_VECTOR_FILTER_BLOCK = 2,
)

"""
    Must be called before any other starpu function. Field extern_task_path is the
    shared library path which will be used to find StarpuCodelet
    cpu and gpu function names
"""
function starpu_init()
    debug_print("starpu_init")

    if (get(ENV,"JULIA_TASK_LIB",0)!=0)
        global starpu_tasks_library_handle= Libdl.dlopen(ENV["JULIA_TASK_LIB"])
        debug_print("Loading external codelet library")
        ff = Libdl.dlsym(starpu_tasks_library_handle,:starpu_find_function)
        dump(ff)
        for k in keys(CUDA_CODELETS)
            CPU_CODELETS[k]=unsafe_string(ccall(ff,Cstring, (Cstring,Cstring),Cstring_from_String(string(k)),Cstring_from_String("cpu")))
            print(k,">>>>",CPU_CODELETS[k],"\n")
        end
    else
        debug_print("generating codelet library")
        run(`make generated_tasks.so`);
        global starpu_tasks_library_handle=Libdl.dlopen("generated_tasks.so")
    end
    output = @starpucall jlstarpu_init Cint ()

    starpu_enter_new_block()

    return output
end

"""
    Must be called at the end of the program
"""
function starpu_shutdown()
    debug_print("starpu_shutdown")

    starpu_exit_block()
    @starpucall starpu_shutdown Cvoid ()
    jlstarpu_free_allocated_structures()
    return nothing
end

STARPU_MAIN_RAM = 0 #TODO: ENUM

function starpu_memory_pin(data) :: Nothing
    data_pointer = pointer(data)

    @starpucall(starpu_memory_pin,
                Cvoid, (Ptr{Cvoid}, Csize_t),
                data_pointer,
                sizeof(data))
end

function starpu_memory_unpin(data) :: Nothing
    data_pointer = pointer(data)

    @starpucall(starpu_memory_unpin,
                Cvoid, (Ptr{Cvoid}, Csize_t),
                data_pointer,
                sizeof(data))
end

function StarpuNewDataHandle(ptr :: StarpuDataHandlePointer, destr :: Function...) :: StarpuDataHandle
    return StarpuDestructible(ptr, destr...)
end



function starpu_data_unregister_pointer(ptr :: StarpuDataHandlePointer)
    @starpucall(starpu_data_unregister, Cvoid, (Ptr{Cvoid},), ptr)
end

function starpu_data_unregister(handles :: StarpuDataHandle...)
    for h in handles
        starpu_execute_destructor!(h, starpu_data_unregister_pointer)
    end
end

function starpu_data_register(v :: Vector{T}) where T
    output = Ref{Ptr{Cvoid}}(0)
    data_pointer = pointer(v)

    @starpucall(starpu_vector_data_register,
                Cvoid,
                (Ptr{Cvoid}, Cint, Ptr{Cvoid}, UInt32, Csize_t),
                output, STARPU_MAIN_RAM, data_pointer,
                length(v), sizeof(T)
            )
    return StarpuNewDataHandle(output[], starpu_data_unregister_pointer)#, [starpu_data_unregister_pointer])
end

function starpu_data_register(m :: Matrix{T}) where T

    output = Ref{Ptr{Cvoid}}(0)
    data_pointer = pointer(m)
    (height, width) = size(m)

    @starpucall(starpu_matrix_data_register,
                Cvoid,
                (Ptr{Cvoid}, Cint, Ptr{Cvoid},
                    UInt32, UInt32, UInt32, Csize_t),
                output, STARPU_MAIN_RAM, data_pointer,
                height, height, width, sizeof(T)
            )

    return StarpuNewDataHandle(output[], starpu_data_unregister_pointer)#, [starpu_data_unregister_pointer])
end

function starpu_data_register(block :: Array{T,3}) where T

    output = Ref{Ptr{Cvoid}}(0)
    data_pointer = pointer(block)
    (height, width, depth) = size(block)

    @starpucall(starpu_block_data_register,
                Cvoid,
                (Ptr{Cvoid}, Cint, Ptr{Cvoid},
                    UInt32, UInt32, UInt32, UInt32,
                    UInt32, Csize_t),
                output, STARPU_MAIN_RAM, data_pointer,
                height, height * width,
                height, width, depth,
                sizeof(T)
            )

    return StarpuNewDataHandle(output[], starpu_data_unregister_pointer)
end

function starpu_data_register(ref :: Ref{T}) where T

    output = Ref{Ptr{Cvoid}}(0)

    @starpucall(starpu_variable_data_register,
                Cvoid,
                (Ptr{Cvoid}, Cint, Ptr{Cvoid}, Csize_t),
                output, STARPU_MAIN_RAM, ref, sizeof(T)
            )

    return StarpuNewDataHandle(output[], starpu_data_unregister_pointer)
end

function starpu_data_register(x1, x2, next_args...)

    handle_1 = starpu_data_register(x1)
    handle_2 = starpu_data_register(x2)

    next_handles = map(starpu_data_register, next_args)

    return [handle_1, handle_2, next_handles...]
end

function starpu_data_get_sub_data(root_data :: StarpuDataHandle, id)
    output = @starpucall(starpu_data_get_sub_data,
                        Ptr{Cvoid}, (Ptr{Cvoid}, Cuint, Cuint),
                        root_data.object, 1, id - 1
                    )

    return StarpuNewDataHandle(output)
end

function starpu_data_get_sub_data(root_data :: StarpuDataHandle, idx, idy)
    output = @starpucall(starpu_data_get_sub_data,
                        Ptr{Cvoid}, (Ptr{Cvoid}, Cuint, Cuint, Cuint),
                        root_data.object, 2, idx - 1, idy - 1
                    )

    return StarpuNewDataHandle(output)
end

import Base.getindex
function Base.getindex(handle :: StarpuDataHandle, indexes...)
     starpu_data_get_sub_data(handle, indexes...)
 end


"""
    TODO : use real function pointers loaded from starpu shared library
"""
mutable struct StarpuDataFilter

    filter_func :: StarpuDataFilterFunc
    nchildren :: Cuint

    function StarpuDataFilter(filter_func, nchildren)
        output = new()
        output.filter_func = filter_func
        output.nchildren = Cuint(nchildren)
        return output
    end

end

function starpu_data_unpartition_pointer(ptr :: StarpuDataHandlePointer)
    @starpucall(starpu_data_unpartition, Cvoid, (Ptr{Cvoid}, Cuint), ptr, STARPU_MAIN_RAM)
end

function starpu_data_partition(handle :: StarpuDataHandle, filter :: StarpuDataFilter)

    starpu_add_destructor!(handle, starpu_data_unpartition_pointer)

    @starpucall(jlstarpu_data_partition,
            Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}),
            handle.object, Ref{StarpuDataFilter}(filter)
        )
end

function starpu_data_unpartition(handles :: StarpuDataHandle...)

    for h in handles
        starpu_execute_destructor!(h, starpu_data_unpartition_pointer)
    end

    return nothing
end

function starpu_data_map_filters(handle :: StarpuDataHandle, filter :: StarpuDataFilter)

    starpu_add_destructor!(handle, starpu_data_unpartition_pointer)

    @starpucall(jlstarpu_data_map_filters_1_arg,
            Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}),
            handle.object, Ref{StarpuDataFilter}(filter)
    )
end

function starpu_data_map_filters(handle :: StarpuDataHandle, filter_1 :: StarpuDataFilter, filter_2 :: StarpuDataFilter)
    starpu_add_destructor!(handle, starpu_data_unpartition_pointer)

    @starpucall(jlstarpu_data_map_filters_2_arg,
            Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
            handle.object,
            Ref{StarpuDataFilter}(filter_1),
            Ref{StarpuDataFilter}(filter_2)
    )
end

"""
    Launches task execution, if "synchronous" task field is set to "false", call
    returns immediately
"""
function starpu_task_submit(task :: StarpuTask)
    
    if (length(task.handles) != length(task.cl.modes))
        error("Invalid number of handles for task : $(length(task.handles)) where given while codelet has $(task.cl.modes) modes")
    end

    starpu_c_task_update(task)

    @starpucall starpu_task_submit Cint (Ptr{Cvoid},) task.c_task
end


function starpu_modes(x :: Symbol)
    if (x == Symbol("STARPU_RW"))
        return STARPU_RW
    elseif (x == Symbol("STARPU_R"))
        return STARPU_R
    else return STARPU_W
    end
end

"""
    Creates and submits an asynchronous task running cl Codelet function.
    Ex : @starpu_async_cl cl(handle1, handle2)
"""
macro starpu_async_cl(expr, modes, cl_arg=(), color ::UInt32=0x00000000)

    if (!isa(expr, Expr) || expr.head != :call)
        error("Invalid task submit syntax")
    end
    if (!isa(expr, Expr)||modes.head != :vect)
        error("Invalid task submit syntax")
    end
    perfmodel = StarpuPerfmodel(
        perf_type = STARPU_HISTORY_BASED,
        symbol = "history_perf"
    )
    println(CPU_CODELETS[string(expr.args[1])])
    cl = StarpuCodelet(
        cpu_func = CPU_CODELETS[string(expr.args[1])],
        # cuda_func = CUDA_CODELETS[string(expr.args[1])],
        #opencl_func="ocl_matrix_mult",
        ### TODO: CORRECT !
        modes = map((x -> starpu_modes(x)),modes.args),
        perfmodel = perfmodel,
        color = color
    )
    handles = Expr(:vect, expr.args[2:end]...)
    #dump(handles)
    quote
        task = StarpuTask(cl = $(esc(cl)), handles = $(esc(handles)), cl_arg=$(esc(cl_arg)))
        starpu_task_submit(task)
    end
end

"""
    Blocks until every submitted task has finished.
"""
function starpu_task_wait_for_all()
    @threadcall(@starpufunc(:starpu_task_wait_for_all),
                          Cint, ())
end

function repl(x::Symbol)
    return x
end
function repl(x::Number)
    return x
end
function repl(x :: Expr)
    if (x.head == :call && x.args[1] == :+)
        if (x.args[2] == :_)
            return x.args[3]
        elseif (x.args[3] == :_)
            return x.args[2]
        else return Expr(:call,:+,repl(x.args[2]),repl(x.args[3]))
        end
    elseif (x.head == :call && x.args[1] == :-)
        if (x.args[2] == :_)
            return Expr(:call,:-,x.args[3])
        elseif (x.args[3] == :_)
            return x.args[2]
        else return Expr(:call,:-,repl(x.args[2]),repl(x.args[3]))
        end
    else return Expr(:call,x.args[1],repl(x.args[2]),repl(x.args[3]))
    end
end
"""
    Declares a subarray.
    Ex : @starpu_filter ha = A[ _:_+1, : ] 
 
"""
macro starpu_filter(expr)
    #dump(expr, maxdepth=20)
    if (expr.head==Symbol("="))
        region = expr.args[2]
        if (region.head == Symbol("ref"))
            farray = expr.args[1]
            println("starpu filter")
            index = 0
            filter2=nothing
            filter3=nothing
            if (region.args[2]==Symbol(":"))
                index = 3
                filter2=:(STARPU_MATRIX_FILTER_BLOCK)
            elseif (region.args[3] == Symbol(":"))
                index = 2
                filter3=:(STARPU_MATRIX_FILTER_VERTICAL_BLOCK)
            else
            end
            ex = repl(region.args[index].args[3])
            if (region.args[index].args[2] != Symbol("_"))
                throw(AssertionError("LHS must be _"))
            end
            ret = quote
                # escape and not global for farray!
                $(esc(farray)) = starpu_data_register($(esc(region.args[1])))
                starpu_data_partition( $(esc(farray)),StarpuDataFilter($(esc(filter)),$(esc(ex))))
            end
            return ret
        else
            ret = quote
                $(esc(farray))= starpu_data_register($(esc(region.args[1])))
            end
            
            dump("coucou"); #dump(region.args[2])
            #                dump(region.args[2])
            #                dump(region.args[3])
            return ret
        end
    end
end
"""
    Blocks until every submitted task has finished.
    Ex : @starpu_sync_tasks begin
                [...]
                starpu_task_submit(task)
                [...]
        end

    TODO : Make the macro only wait for tasks declared inside the following expression.
            (similar mechanism as @starpu_block)
"""
macro starpu_sync_tasks(expr)
    quote
        $(esc(expr))
        starpu_task_wait_for_all()
    end
end

function StarpuDestructible(obj :: T, destructors :: Function...) where T

    if (isempty(starpu_block_list))
        error("Creation of a StarpuDestructible object while not beeing in a @starpu_block")
    end

    l = LinkedList{Function}()

    for destr in destructors
        add_to_tail!(l, destr)
    end

    output = StarpuDestructible{T}(obj, l)
    add_to_head!(starpu_block_list[end], output)

    return output
end

function starpu_enter_new_block()

    push!(starpu_block_list, LinkedList{StarpuDestructible}())
end

function starpu_destruct!(x :: StarpuDestructible)

    @foreach_asc  x.destructors destr begin
        destr.data(x.object)
    end

    empty!(x.destructors)

    return nothing
end


function starpu_exit_block()

    destr_list = pop!(starpu_block_list)

    @foreach_asc destr_list x begin
        starpu_destruct!(x.data)
    end
end

"""
    Adds new destructors to the list of function. They will be executed before
        already stored ones when calling starpu_destruct!
"""
function starpu_add_destructor!(x :: StarpuDestructible, destrs :: Function...)

    for d in destrs
        add_to_head!(x.destructors, d)
    end

    return nothing
end

"""
    Removes detsructor without executing it
"""
function starpu_remove_destructor!(x :: StarpuDestructible, destr :: Function)

    @foreach_asc x.destructors lnk begin

        if (lnk.data == destr)
            remove_link!(lnk)
            break
        end
    end

    return nothing
end

"""
    Executes "destr" function. If it was one of the stored destructors, it
    is removed.
    This function can be used to allow user to execute a specific action manually
        (ex : explicit call to starpu_data_unpartition() without unregistering)
"""
function starpu_execute_destructor!(x :: StarpuDestructible, destr :: Function)

    starpu_remove_destructor!(x, destr)
    return destr(x.object)
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

"""
    Updates fields of the real C structures stored at "c_task" field
"""
function starpu_c_task_update(task :: StarpuTask)

    task_translator = StarpuTaskTranslator(task)

    @starpucall(jlstarpu_task_update,
                Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}),
                Ref{StarpuTaskTranslator}(task_translator),
                task.c_task
                )
end

function starpu_tag_declare_deps(id :: StarpuTag, dep :: StarpuTag, other_deps :: StarpuTag...)

    v = [dep, other_deps...]

    @starpucall(starpu_tag_declare_deps_array,
                Cvoid, (StarpuTag, Cuint, Ptr{StarpuTag}),
                id, length(v), pointer(v)
        )
end

function starpu_iteration_push(iteration)
    @starpucall(starpu_iteration_push,
                Cvoid, (Culong,), iteration
        )
end

function starpu_iteration_pop()
    @starpucall starpu_iteration_pop Cvoid ()
end


function starpu_tag_wait(id :: StarpuTag)
    @starpucall starpu_tag_wait Cint (StarpuTag,) id
end


function starpu_tag_wait(ids :: Vector{StarpuTag})
    @starpucall(starpustarpu_tag_wait_array,
                Cint, (Cuint, Ptr{StarpuTag}),
                length(ids), pointer(ids)
        )
end

function starpu_task_destroy(task :: StarpuTask)
    @starpucall starpu_task_destroy Cvoid (Ptr{Cvoid},) task.c_task
end

"""
    Block until there are n submitted tasks left (to the current context or the global one if there is no current context) to
    be executed. It does not destroy these tasks.
"""
function starpu_task_wait_for_n_submitted(n)
    @starpucall starpu_task_wait_for_n_submitted Cint (Cuint,) n
end

"""
    starpu_task_declare_deps(task :: StarpuTask, dep :: StarpuTask [, other_deps :: StarpuTask...])

    Declare task dependencies between a task and the following provided ones. This function must be called
    prior to the submission of the task, but it may called after the submission or the execution of the tasks in the array,
    provided the tasks are still valid (i.e. they were not automatically destroyed). Calling this function on a task that was
    already submitted or with an entry of task_array that is no longer a valid task results in an undefined behaviour.
"""
function starpu_task_declare_deps(task :: StarpuTask, dep :: StarpuTask, other_deps :: StarpuTask...)

    task_array = [dep.c_task, map((t -> t.c_task), other_deps)...]

    @starpucall(starpu_task_declare_deps_array,
                Cvoid, (Ptr{Cvoid}, Cuint, Ptr{Cvoid}),
                task.c_task,
                length(task_array),
                pointer(task_array)
            )
end

function starpu_c_codelet_update(cl :: StarpuCodelet)

    translating_cl = StarpuCodeletTranslator(cl)

    @starpucall(jlstarpu_codelet_update,
                Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}),
                Ref{StarpuCodeletTranslator}(translating_cl),
                cl.c_codelet
            )
end

export starpu_find_function

function starpu_find_function(name :: String, device :: String ) 
    s=ccall(:starpu_find_function,Cstring, (Cstring,Cstring),Cstring_from_String(name),Cstring_from_String(device))
    if  s == C_NULL
        print("NULL STRING\n")
        error("dead")
    end
    return s
end

function load_starpu_function_pointer(func_name :: String)

    if (isempty(func_name))
        return C_NULL
    end
    #func_pointer = ccall(:dlsym,"libdl",Ptr{Cvoid});
    func_pointer=Libdl.dlsym(starpu_tasks_library_handle, func_name)

    if (func_pointer == C_NULL)
        error("Couldn't find function symbol $func_name into extern library file $starpu_tasks_library")
    end

    return func_pointer
end


mutable struct StarpuCodeletTranslator

    where_to_execute :: UInt32

    color :: UInt32

    cpu_func :: Ptr{Cvoid}
    cpu_func_name :: Cstring

    cuda_func :: Ptr{Cvoid}

    opencl_func :: Ptr{Cvoid}
    
    nbuffers :: Cint
    modes :: Ptr{Cvoid}

    perfmodel :: Ptr{Cvoid}



    function StarpuCodeletTranslator(cl :: StarpuCodelet)

        output = new()

        if (iszero(cl.where_to_execute))
            error("StarpuCodelet field \"where_to_execute\" is empty")
        end

        output.where_to_execute = cl.where_to_execute
        output.color = cl.color

        cpu_func_ptr = load_starpu_function_pointer(cl.cpu_func)
        cuda_func_ptr = load_starpu_function_pointer(cl.cuda_func)
        opencl_func_ptr = load_starpu_function_pointer(cl.opencl_func)

        if (cpu_func_ptr == C_NULL && cuda_func_ptr == C_NULL)
            error("No function specified inside codelet")
        end

        output.cpu_func = cpu_func_ptr
        output.cpu_func_name = Cstring_from_String(cl.cpu_func)

        output.cuda_func = cuda_func_ptr
        output.opencl_func = opencl_func_ptr

        output.nbuffers = Cint(length(cl.modes))
        output.modes = pointer(cl.modes)

        output.perfmodel = cl.perfmodel.c_perfmodel

        return output
    end

end

#StarpuDataFilter
#StarpuCodelet

"""
    Declares a Julia function which is just calling the StarPU function
    having the same name.
"""
macro starpu_noparam_function(func_name, ret_type)

    func = Symbol(func_name)

    quote
        export $func
        global $func() = ccall(($func_name, starpu_task_library_name),
                                $ret_type, ()) :: $ret_type
    end
end


@starpu_noparam_function "starpu_is_initialized" Cint
@starpu_noparam_function "starpu_cublas_init" Cvoid
@starpu_noparam_function "starpu_cublas_set_stream" Cvoid
@starpu_noparam_function "starpu_cublas_shutdown" Cvoid

end
