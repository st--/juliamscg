abstract Potential

abstract PairPotential <: Potential

immutable LennardJones{N<:Number} <: PairPotential
    ϵ::N
    σ::N
end

immutable LinearPot <: PairPotential
    k::Number
end
immutable LinearPotCut <: PairPotential
    k::Number
    cutoff::Number
end
feval(p::LinearPot, r::Number) = p.k*r
feval(p::LinearPotCut, r::Number) = r<p.cutoff ? p.k*r : zero(r)

immutable ConstPot <: PairPotential
    f::Number
end
feval(p::ConstPot, r::Number) = p.f

immutable MorsePot <: PairPotential
    a::Number
    r0::Number
end
ueval(p::MorsePot, r::Number) = (1 - exp(-p.a * (r - p.r0)))^2
feval(p::MorsePot, r::Number) = 2p.a * (1 - exp(-p.a * (r - p.r0))) * exp(-p.a * (r - p.r0))

normalised{N<:Number}(v::Vector{N}) = v / vecnorm(v)

ueval{N<:Number}(p::LennardJones{N}, r::N) = 4p.ϵ * ((p.σ/r)^12 - (p.σ/r)^6)
feval{N<:Number}(p::LennardJones{N}, r::N) = - (4p.ϵ/r * (-12(p.σ/r)^12 + 6(p.σ/r)^6))

ueval{N<:Number}(p::PairPotential, x1::Vector{N}, x2::Vector{N}) = eval(p, vecnorm(x1 - x2))
feval{N<:Number}(p::PairPotential, x1::Vector{N}, x2::Vector{N}) = feval(p, vecnorm(x1 - x2)) * normalised(x1 - x2)


function calc_forces(configs, pot)
    nt, Ncg, dim = size(configs)
    @assert dim==3
    
    forces = zeros(configs)
    for t=1:nt, i=1:Ncg, j=i+1:Ncg
        f = feval(pot, vec(configs[t,i,:]), vec(configs[t,j,:]))
        f = reshape(f, (1,1,dim))
        forces[t,i,:] += f
        forces[t,j,:] -= f
    end
    forces
end

