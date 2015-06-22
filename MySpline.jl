module MySpline

import Base.length

export
    Spline,
    DeltaSpline, LinearSpline, CubicSpline,
    basisA, basisB, basisC, basisD,
    basisAint, basisBint, basisCint, basisDint,
    basisAd, basisBd, basisCd, basisDd,
    splinebasis, splinebasisint, splinebasisderiv,
    splineeval, splineevalint,
    splinefitmatrix,
    constraint_firstderiv,
    solvecoeffmat,
    findsplinecoeff

abstract Spline

for spltype = (:DeltaSpline, :LinearSpline, :CubicSpline)
    @eval begin
        immutable $spltype <: Spline
            gridx
        end
    end
end

length(s::DeltaSpline) = length(s.gridx)
length(s::LinearSpline) = length(s.gridx)
length(s::CubicSpline) = 2length(s.gridx)

# discrete delta basis set:
function splinebasis(s::DeltaSpline, d::Integer, x::Real)
    gridx = s.gridx
    if d == 1
        gridx[1] - (gridx[2]-gridx[1])/2 < x <= (gridx[d] + gridx[d+1])/2 ? one(x) : zero(x)
    elseif d == length(gridx)
        (gridx[end-1] + gridx[end])/2 < x <= gridx[end] + (gridx[end]-gridx[end-1])/2 ? one(x) : zero(x)
    else
        (gridx[d-1] + gridx[d])/2 < x <= (gridx[d] + gridx[d+1])/2 ? one(x) : zero(x)
    end
end

# linear interpolation:
basisA(gridx, d, x) = !(gridx[d] < x <= gridx[d+1]) ? zero(x) : (gridx[d+1] - x) / (gridx[d+1] - gridx[d])
basisB(gridx, d, x) = !(gridx[d] < x <= gridx[d+1]) ? zero(x) : 1 - basisA(gridx, d, x)

basisAint(gridx, d, x) = - (gridx[d+1] - x)^2 / 2(gridx[d+1] - gridx[d])
basisBint(gridx, d, x) = + (x - gridx[d])^2 / 2(gridx[d+1] - gridx[d])

basisAd(gridx, d, x) = !(gridx[d] < x <= gridx[d+1]) ? zero(x) : - 1 / (gridx[d+1] - gridx[d])
basisBd(gridx, d, x) = !(gridx[d] < x <= gridx[d+1]) ? zero(x) : - basisAd(gridx, d, x)

function gridindex(x, grid)
    N = length(grid)
    k = 1
    while k <= N && grid[k] < x
        k += 1
    end
    k - 1
end

function choosebasis(basisOne::Function, basisTwo::Function, gridx, d::Integer, x::Real)
    # the combination of basisOne and basisTwo stretches over two grid cells
    # this function chooses the correct one to match them up:
    # basisTwo from the previous grid point, or basisOne from the current grid point
    if x <= gridx[1] || x > gridx[end]
        return zero(x)
    end
    i = gridindex(x, gridx)
    if i == d - 1 && d > 1
        basisTwo(gridx, d-1, x)
    elseif i == d && d < length(gridx)
        basisOne(gridx, d, x)
    else
        zero(x)
    end
end

splinebasis(s::LinearSpline, d::Integer, x::Real) = (
    choosebasis(basisA, basisB, s.gridx, d, x)
)
splinebasisint(s::LinearSpline, d::Integer, x::Real) = (
    choosebasis(basisAint, basisBint, s.gridx, d, x)
)
splinebasisderiv(s::LinearSpline, d::Integer, x::Real) = (
    choosebasis(basisAd, basisBd, s.gridx, d, x)
)

# cubic splines:
basisC(gridx, d, x) = (local A = basisA(gridx, d, x);
    1/6 * (A^3 - A)*(gridx[d+1]-gridx[d])^2)
basisD(gridx, d, x) = (local B = basisB(gridx, d, x);
    1/6 * (B^3 - B)*(gridx[d+1]-gridx[d])^2)

basisCint(gridx, d, x) = (local Ai = basisAint(gridx, d, x);
    1/6 * (-Ai^2 - Ai*(gridx[d+1]-gridx[d])) * (gridx[d+1]-gridx[d]))
basisDint(gridx, d, x) = (local Bi = basisBint(gridx, d, x);
    1/6 * (+Bi^2 - Bi*(gridx[d+1]-gridx[d])) * (gridx[d+1]-gridx[d]))

basisCd(gridx, d, x) = (local A = basisA(gridx, d, x); local Ad = basisAd(gridx, d, x);
    1/6 * (3A^2 - 1) * Ad * (gridx[d+1]-gridx[d])^2)
basisDd(gridx, d, x) = (local B = basisB(gridx, d, x); local Bd = basisBd(gridx, d, x);
    1/6 * (3B^2 - 1) * Bd * (gridx[d+1]-gridx[d])^2)

function choosesplitbasis(bases, gridx, d2::Integer, x::Real)
    # takes a 2x2 matrix of basis functions, and calls choosebasis()
    # on the first row if d2 is odd, and on the second if d2 is even
    # so there is the correspondence
    # d2=1,2 -> (basis1,basis2)(1)
    # d2=3,4 -> (basis1,basis2)(2)
    # etc...
    d, which = divrem(d2+1, 2)
    choosebasis(bases[which+1,:]..., gridx, d, x)
end

splinebasis(s::CubicSpline, d2::Integer, x::Real) = (
    choosesplitbasis([basisA basisB; basisC basisD], s.gridx, d2, x)
)
splinebasisint(s::CubicSpline, d2::Integer, x::Real) = (
    choosesplitbasis([basisAint basisBint; basisCint basisDint], s.gridx, d2, x)
)
splinebasisderiv(s::CubicSpline, d2::Integer, x::Real) = (
    choosesplitbasis([basisAd basisBd; basisCd basisDd], s.gridx, d2, x)
)

# splinebasis(s::{LinearSpline,CubicSpline}, 1, gridx[1]) = 0.0,
# even though it's 1.0 infinitesimally to the right...
# -> left-open.
# would need proper boundary conditions to fix this.


splineeval(s::Spline, ϕs, x) = dot(ϕs, [splinebasis(s, d, x) for d=1:length(s)])
# the value of the spline fit at position x, given a spline with grid s and coefficients ϕs
# this is inefficient, as most basis functions / coefficients are actually irrelevant!...

# d2 set d   i=1     i=2     i=3
#          [g1:g2] [g2:g3] [g3:g4]
# 1  A/B 1    A
# 2  C/D 1    C
# 3  A/B 2    B       A
# 4  C/D 2    D       C
# 5  A/B 3            B       A
# 6  C/D 3            D       C
# 7  A/B 4                    B
# 8  C/D 4                    D

# x between gridx[i], gridx[i+1]:
# A <- phi[2i-1]
# C <- phi[2i]
# B <- phi[2i+1]
# D <- phi[2i+2]

function splineevalint(s::CubicSpline, phi, xs)
    # integrate from right, starting at zero
    ys = zeros(xs)

    runningintval = 0.0 # value of integral at right edge of block
    # start with right-most grid interval
    for i = [length(s.gridx)-1 : -1 : 1]
        a, b = s.gridx[i], s.gridx[i+1]
        intval(x) = (phi[2i-1] * basisAint(s.gridx, i, x)
            + phi[2i] * basisCint(s.gridx, i, x)
            + phi[2i+1] * basisBint(s.gridx, i, x)
            + phi[2i+2] * basisDint(s.gridx, i, x))
        valright = intval(b)

        in_block = a .< xs .<= b
        ys[in_block] = [runningintval - (valright - intval(x))
            for x in xs[in_block]]

        runningintval = runningintval - (valright - intval(a))
    end
    ys
end

probabled(s::DeltaSpline, idx) = [idx-1, idx]
probabled(s::LinearSpline, idx) = [idx-1, idx]
probabled(s::CubicSpline, idx) = [2idx-1:2idx+2]

function splinefitrow!(Gr, s::Spline, x)
    @assert length(Gr) == length(s)
    idx = gridindex(x, s.gridx)
    for d=probabled(s, idx) # only do calls which are likely to be non-zero
        (d < 1 || d > length(s)) && continue
        Gr[d] = splinebasis(s, d, x)
    end
end

function splinefitrow!(G, i, s::CubicSpline, x)
    grid = s.gridx
    k = gridindex(x, grid)
    1 <= k < length(grid) || return
    G[i, 2k-1] = basisA(grid, k, x)
    G[i, 2k]   = basisC(grid, k, x)
    G[i, 2k+1] = basisB(grid, k, x)
    G[i, 2k+2] = basisD(grid, k, x)
    G
end

function splinefitmatrix(s::Spline, dataxs)
    Nd = length(s)
    Nt = length(dataxs)
    G = zeros(Nt, Nd)
    @inbounds for i=1:Nt
        splinefitrow!(G, i, s, dataxs[i])
    end
    G
end

function constraint_firstderiv(s::CubicSpline)
    Nd = length(s)
    Np = length(s.gridx)
    Nc = Np - 2 # number of inner points = constraints on first derivative
    Gd = zeros(Nc, Nd)
    for i=2:Np-1
        Δl = s.gridx[i] - s.gridx[i-1]
        Δr = s.gridx[i+1] - s.gridx[i]
        Gd[i-1, 2i-3:2i+2] = [
               -1/Δl,       Δl/6,
            1/Δl + 1/Δr, (Δl + Δr)/3,
               -1/Δr,       Δr/6
        ]
    end
    Gd, zeros(Nc)
end

function solvecoeffmat(s::Spline, G, b; use_normal=true, contderiv=true)
    if use_normal
        b = G' * b
        G = G' * G # smaller, square-matrix... but with larger condition number
    end
    
    if contderiv && isa(s, CubicSpline)
        Cmat, Cvec = constraint_firstderiv(s)
        G = vcat(G, Cmat)
        b = vcat(b, Cvec)
    end
    
    G \ b # =: ϕs
end

function findsplinecoeff(s::Spline, teachx, teachy; use_normal=true, contderiv=true)
    G = splinefitmatrix(s, teachx) # bigger, non-square matrix
    solvecoeffmat(s, G, teachy, use_normal=use_normal, contderiv=contderiv)
end


end #module
