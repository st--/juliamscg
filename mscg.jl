export Configuration,
    make_many_mscgmat, make_mscg_matrix, my_mscg_matrix,
    solve_blockavg


immutable Configuration
    types::Vector{Symbol}
    pos
    forces
end


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
        
        cgvars = getcgvars(cgint, cfg.types)
        Ncgvars = length(cgvars)
        
        for t=1:Nt
            rs = slice(cfg.pos, t,:,:) # Nx3 matrix of CG positions in this configuration
            
            cgvalues = Float[cgvalue(cg, rs) for cg in cgvars]
            cgderivs = reshape(hcat([cgderiv(cg, rs) for cg in cgvars]...), (Ncg*dim, Ncgvars))
            fmat = splinefitmatrix(cgint.spl, cgvalues)
            
            G[t, :, iD:iD+NDi-1] = cgderivs * fmat
        end
        
        iD += NDi
    end
    reshape(G, (Nt*Ncg*dim, ND))
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

