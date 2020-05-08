import Pkg

try
    using CBinding
    using Clang
    using ThreadPools
catch
    Pkg.activate((@__DIR__)*"/../..")
    Pkg.instantiate()
    using Clang
    using CBinding
    using ThreadPools
end
