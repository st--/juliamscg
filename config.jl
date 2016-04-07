typealias BoxType Union{Void,Real,Vector}

immutable Configuration
    types::Vector{Symbol}
    box::BoxType
    pos::Array
    force::Array
    function Configuration(types::Vector{Symbol}, box::BoxType, pos::Array, force::Array)
        (length(types) == size(pos, 2) == size(force, 2)) || error("number of sites does not match")
        (size(pos) == size(force)) || error("size of frames does not match")
        (length(pos) == size(pos, 1) * length(types) * 3) || error("dimensions don't multiply up")
        new(types, box, pos, force)
    end
end

Configuration(types::Vector{Symbol}, pos::Array, force::Array) = Configuration(types, nothing, pos, force)

wrapvec(L, v) = (v + L/2) .% L - L/2
wrapvec(L::Vector, v::Matrix) = (v .+ L'/2) .% L' .- L'/2
wrapvec(L::Void, v) = v

wrapdiff(cfg::Configuration, t, i, j) = wrapvec(cfg.box, cfg.pos[t,j,:] - cfg.pos[t,i,:])

wrapbox{T<:Real}(v::T, L::T) = (v + L/2) % L - L/2

wrapvec!(v::Vector, L::Void) = v
function wrapvec!(v::Vector, L::Real)
    @inbounds for d=1:length(v)
        v[d] = wrapbox(v[d], L)
    end
    v
end
function wrapvec!(v::Vector, L::Vector)
    (n = length(v)) == length(L) || throw(DimensionMismatch("v and L must have same length"))
    @inbounds for d=1:n
        v[d] = wrapbox(v[d], L[d])
    end
    v
end

function wrapdiff!!(v::Vector, cfg::Configuration, t::Integer, i::Integer, j::Integer)
    @inbounds for d=1:3
        v[d] = cfg.pos[t,j,d] - cfg.pos[t,i,d]
    end
    wrapvec!(v, cfg.box)
end

