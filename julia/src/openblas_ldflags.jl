import LinearAlgebra.BLAS
import Libdl


libdir = normpath(joinpath(splitpath(filter(x->occursin(Base.libblas_name,x), Libdl.dllist())[1])[1:end-1]...))
libpath = normpath(filter(x->occursin(Base.libblas_name,x), Libdl.dllist())[1])
libname = Base.libblas_name[4:end]
println("-Wl,-rpath,$libpath -L$libdir -l$libname")

