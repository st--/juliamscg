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
cgvalue(cg::CGDist, cfg::Configuration, t::Integer) = vecnorm(wrapdiff(cfg, t, cg.i, cg.j))
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

function cgderiv(cg::CGDist, cfg::Configuration, t::Integer)
    deriv = zeros(size(cfg.pos)[2:end])
    rij = wrapdiff(cfg, t, cg.i, cg.j)
    n = rij / vecnorm(rij)
    deriv[cg.i,:] = -n
    deriv[cg.j,:] = n
    deriv
end

function cgcalc!(vals, derivs, i,  cg::CGDist, cfg::Configuration, t::Integer)
    rij = wrapdiff(cfg, t, cg.i, cg.j)
    rijnorm = vecnorm(rij)
    vals[i] = rijnorm
    n = rij / rijnorm
    derivs[cg.i,:,i] = -n
    derivs[cg.j,:,i] = n
end

function myvecnorm(x::Array)
    sum = zero(eltype(x))
    for v in x
        sum += v*v
    end
    sqrt(sum)
end

function cgcalc!!(rij::Vector, vals::Vector, derivs::Array, i::Integer,  cg::CGDist, cfg::Configuration, t::Integer)
    wrapdiff!!(rij, cfg, t, cg.i, cg.j)
    rijnorm = myvecnorm(rij)
    vals[i] = rijnorm
    @inbounds for d=1:3
        n = rij[d] / rijnorm
        derivs[cg.i, d, i] = -n
        derivs[cg.j, d, i] = n
    end
end
