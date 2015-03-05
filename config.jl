export Configuration, wrapvec, wrapdiff

immutable Configuration
    types::Vector{Symbol}
    box
    pos
    forces
end

Configuration(types::Vector{Symbol}, pos, forces) = Configuration(types, nothing, pos, forces)

wrapvec(L, v) = (v + L/2) .% L - L/2
wrapvec(L::Vector, v::Matrix) = (v .+ L'/2) .% L' .- L'/2
wrapvec(L::Nothing, v) = v

wrapdiff(cfg::Configuration, t, i, j) = wrapvec(cfg.box, vec(cfg.pos[t,j,:]) - vec(cfg.pos[t,i,:]))

