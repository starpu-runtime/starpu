import Pkg

try
    using CBinding
    using Clang
catch
    Pkg.activate((@__DIR__)*"/../..")
    Pkg.instantiate()
    using Clang
    using CBinding
end
