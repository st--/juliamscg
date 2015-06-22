module MSCG

using MySpline

import
    Base.length
export
    CGVariable,
    CGInteraction, CGPairInteraction,
    CGPairSpline,
    numcoeff, getcgvars,
    CGDist,
    involvedin, cgvalue, cgderiv,
    cgcalc!

export
    Configuration,
    wrapvec, wrapdiff

export
    make_many_mscgmat, make_mscg_matrix, my_mscg_matrix,
    solve_blockavg

include("config.jl")
include("cg.jl")
include("mscg.jl")

end
