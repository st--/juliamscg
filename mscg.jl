function make_many_mscgmat{I<:CGInteraction}(cgints::Array{I}, cfg::Configuration)
    Nt, Ncg, dim = size(cfg.pos)
    @assert dim == 3

    ND = numcoeff(cgints)

    G = zeros(Nt, Ncg*dim, ND)
    Float = eltype(G)

    # different interactions may have different numbers of coefficients
    iD = 1 # starting index of current interaction's set of coefficients
    for cgint in cgints
        NDi = length(cgint) # current interaction's number of coefficients

        cgvars = filter(allowed, getcgvars(cgint, cfg.types))
        Ncgvars = length(cgvars)

        cgvalues = Array(Float, Ncgvars)
        derivs = Array(Float, Ncg, dim, Ncgvars)
        fmat = Array(Float, Ncgvars, NDi)
        Gsub = Array(Float, Ncg*dim, NDi)

        rij = Array(Float, dim)

        @inbounds for t=1:Nt
            fill!(cgvalues, 0) # not needed?
            fill!(derivs, 0)
            fill!(fmat, 0)
            fill!(Gsub, 0)

            @inbounds for i=1:Ncgvars # much less overhead than enumerate()
                cgcalc!!(rij, cgvalues, derivs, i,  cgvars[i], cfg, t)
            end
            cgderivs = reshape(derivs, (Ncg*dim, Ncgvars))#XXX
            splinefitmatrix!(fmat, cgint.spl, cgvalues)

            BLAS.gemm!('N', 'N', one(Float), cgderivs, fmat, one(Float), Gsub)
            G[t, :, iD:iD+NDi-1] = Gsub
        end

        iD += NDi
    end
    reshape(G, (Nt*Ncg*dim, ND))
end

export add_constraint, mscgcoeff, mscgcalcspline

numconstraints(cgint::CGPairSpline) = length(cgint.spl.gridx) - 2

function add_constraint{I<:CGInteraction}(cgints::Array{I})
    Mrow = mapreduce(numconstraints, (+), cgints)
    Mcol = mapreduce(length, (+), cgints) # = size(mohG, 2)
    conG = zeros(Mrow, Mcol)
    conb = zeros(Mrow)
    irow = icol = 1
    for cgint in cgints
        ncols = length(cgint)
        nrows = numconstraints(cgint)
        conG[irow:irow+nrows-1, icol:icol+ncols-1] = constraint_firstderiv(cgint.spl)[1]
        irow += nrows
        icol += ncols
    end
    conG, conb
end

function mscgcoeff{I<:CGInteraction}(cgints::Array{I}, phis, i)
    ncoeffs = map(length, cgints)
    start = sum(ncoeffs[1:i-1]) + 1
    stop = start + ncoeffs[i] - 1
    phis[start:stop]
end

function mscgcalcspline{I<:CGInteraction}(cgints::Array{I}, phis, i, xs)
    [splineeval(cgints[i].spl, mscgcoeff(cgints, phis, i), x) for x=xs]
end

function make_mscg_matrix(configs, interactions)
    nt, Ncg, dim = size(configs)
    ND = length(interactions)
    
    @assert dim == 3
    Float = eltype(configs)
    
    G = zeros(nt*Ncg*dim, ND)
    
    cgvars = [CGDist(cgs...) for cgs in combinations(1:Ncg, 2)]
    
    #for I=1:length(configs)
    for t=1:nt
        rs = slice(configs, t,:,:) # Nx3 matrix of CG positions in this configuration
        cgvalues = Float[cgvalue(cg,rs) for cg in cgvars]
        fmat = splinefitmatrix(interactions, cgvalues)
        
        for k=1:Ncg, d=1:dim
            I = sub2ind(size(configs), t, k, d)
            
            cgderivs = Float[cgderiv(cg,rs, k,d) for cg in cgvars]
            G[I, :] = cgderivs' * fmat
        end
    end
    G
end


function do_mat_coeff1(rs, grid)
    Ncg = size(rs,1)
    @assert size(rs,2)==3
    Ngrid = length(grid)
    
    G = zeros(Ncg, 3, Ngrid*2)
    for i=1:Ncg
        for j=1:Ncg
            i==j && continue
            
            rvec = rs[j,:] - rs[i,:]
            #wrap_periodic_image!(rvec, boxsize)
            rij = vecnorm(rvec)
            
            (grid[1] < rij < grid[end]) || continue
            
            nvec = rvec / rij
            
            for k=2:length(grid)
                if grid[k] > rij
                    break
                end
            end
            k-=1
            
            G[i,:,2*k-1] -= basisA(grid, k, rij) * nvec
            G[i,:,2*k]   -= basisC(grid, k, rij) * nvec
            G[i,:,2*k+1] -= basisB(grid, k, rij) * nvec
            G[i,:,2*k+2] -= basisD(grid, k, rij) * nvec
        end
    end
    reshape(G, (Ncg*3, Ngrid*2))
end

function my_mscg_matrix(configs, grid)
    Nt, Ncg, dim = size(configs)
    @assert dim==3
    Ngrid = length(grid)
    G = zeros(Nt*Ncg*dim, Ngrid*2)
    for t=1:Nt
        G[t:Nt:end, :] = do_mat_coeff1(reshape(configs[t,:,:], (Ncg,dim)), grid)
    end
    G
end


function solve_blockavg(configs, forces, spl::CubicSpline, blocksize; use_normal=true, contderiv=true)
    @assert size(forces)==size(configs)
    Nt, Ncg, dim = size(configs)
    @assert dim==3
    ϕacc = zeros(length(spl))
    ϕw = zeros(length(spl))
    for t=1:blocksize:Nt
        blockpos = configs[t:t+blocksize-1,:,:]
        blockforce = forces[t:t+blocksize-1,:,:]
        G = my_mscg_matrix(blockpos, spl.gridx)
        b = vec(blockforce)
        ϕ = solvecoeffmat(spl, G, b, use_normal=use_normal, contderiv=contderiv)
        ϕacc += ϕ
        ϕw += (ϕ .!= 0)
    end
    ϕacc ./ ϕw
end

