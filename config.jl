export Configuration, wrapvec, wrapdiff

immutable Configuration
    types::Vector{Symbol}
    box
    pos
    force
    function Configuration(types::Vector{Symbol}, box, pos, force)
        (length(types) == size(pos, 2) == size(force, 2)) || error("number of sites does not match")
        (size(pos) == size(force)) || error("size of frames does not match")
        new(types, box, pos, force)
    end
end

Configuration(types::Vector{Symbol}, pos, force) = Configuration(types, nothing, pos, force)

wrapvec(L, v) = (v + L/2) .% L - L/2
wrapvec(L::Vector, v::Matrix) = (v .+ L'/2) .% L' .- L'/2
wrapvec(L::Nothing, v) = v

wrapdiff(cfg::Configuration, t, i, j) = wrapvec(cfg.box, vec(cfg.pos[t,j,:]) - vec(cfg.pos[t,i,:]))

