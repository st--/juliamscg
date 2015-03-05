abstract MyGrid
immutable EvenGrid <: MyGrid
    x1::Real
    delta::Real
    n::Integer
end

gridpoint(g::EvenGrid, i) = g.x1 + g.delta * (i-1)
gridpoints(g::EvenGrid) = gridpoint(g, [1:g.n])
ingrid(r, g::EvenGrid) = g.x1 <= r <= (g.x1 + g.delta * (g.n-1))
#gridindex(r, g::EvenGrid) = ifloor((r - g.x1) / g.delta) + 1 # floating point issues...
#function gridindex(r, g::EvenGrid) # a bit messy
#    i = 1
#    while r >= gridpoint(g, i+1) && i < g.n
#        i += 1
#    end
#    i
#end
gridindex(r, g::MyGrid) = gridindex(r, gridpoints(g))
gridindex(r, g) = sum(g .< r)


