import Base.length
export CGVariable,
    CGInteraction, CGPairInteraction,
    CGPairSpline,
    numcoeff, getcgvars,
    CGDist,
    involvedin, cgvalue, cgderiv


abstract CGVariable


abstract CGInteraction

abstract CGPairInteraction <: CGInteraction
# needs to provide .valtype, .site1, .site2

immutable CGPairSpline{CGVar <: CGVariable} <: CGPairInteraction
    valtype::Type{CGVar}
    site1::Symbol
    site2::Symbol
    spl::Spline
end

length(cg::CGPairSpline) = length(cg.spl)


numcoeff(cgs) = mapreduce(length, (+), cgs)

function getcgvars(cg::CGPairInteraction, cgtypes::Vector{Symbol})
    if cg.site1 == cg.site2
        [cg.valtype(cgs...) for cgs in combinations(find(cgtypes .== cg.site1), 2)]
    else
        [cg.valtype(i, j) for i=find(cgtypes .== cg.site1), j=find(cgtypes .== cg.site2)]
    end
end        


immutable CGDist <: CGVariable
    i::Integer
    j::Integer
end

involvedin(cg::CGDist) = [cg.i,cg.j]

cgvalue(cg::CGDist, rs) = vecnorm(rs[cg.i,:] - rs[cg.j,:])
# sqrt((r_i,x - r_j,x)^2 + ...)
# d/dr_k,d = \)

function cgderiv(cg::CGDist, rs, k, d)
    # this is effectively the unit vector between CG sites if k is one of them
    if k == cg.i
        (rs[cg.i,d] - rs[cg.j,d]) / cgvalue(cg, rs)
    elseif k == cg.j
        (rs[cg.j,d] - rs[cg.i,d]) / cgvalue(cg, rs)
    else
        zero(eltype(rs))
    end
end

function cgderiv(cg::CGDist, rs, k)
    if k == cg.i
        (rs[cg.i,:] - rs[cg.j,:]) / cgvalue(cg, rs)
    elseif k == cg.j
        (rs[cg.j,:] - rs[cg.i,:]) / cgvalue(cg, rs)
    else
        zeros(size(rs,2))
    end
end

function cgderiv(cg::CGDist, rs)
    deriv = zeros(rs) #spzeros(size(rs)...)
    d = (rs[cg.i,:] - rs[cg.j,:]) / cgvalue(cg, rs)
    deriv[cg.i,:] = d
    deriv[cg.j,:] = -d
    deriv
end

