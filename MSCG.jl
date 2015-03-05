module MSCG

import Base.length

include("MySpline.jl")
using MySpline

include("cg.jl")
include("mscg.jl")

end
