using LinearAlgebra, Plots, DataFrames, MLJ, Random, JuMP, GLPK, Ipopt, Juniper, StatsBase, Statistics, DelimitedFiles, CSV

function m0J(x,c,C)
    m,g = size(C)
    d=0
        for i = 1 : g
            d += norm(x-C[:,i])^(1/2)
        end
        d = norm(x-c)^(1/2)/d
    return d
end

function baldinho(I,g,n,P)
    J = zeros(Int, n,g)
    U = zeros(g,n)
        for i = 1 : g
            b = I[i]
            J[:,b] = sortperm(P[b,:])
            s = 0
            for j = 1 : n
                s += z[J[j,b]]
                if s <= μ[b] 
                    l = 0
                    if i > 1
                        for h = 1 : i-1
                            if U[I[h], J[j, b]] == 1
                                l += 1
                                s -= z[J[j,b]]
                            end
                        end
                    end
                    if l == 0
                        U[b,J[j,b]] = 1
                    end 
                else 
                    s -= z[J[j,b]]
                end
            end
        end
    return U
end

function CriaX(NumPont, Centroi, NumBlobs, seed, Dim)
    rng = MersenneTwister(seed)
    X, y = make_blobs(NumPont, Dim; centers=NumBlobs, rng)
    dfBlobs = DataFrame(X)
    X1 = zeros(NumPont, Dim)
    for i = 1 : Dim
        X1[:,i] = dfBlobs[:,i]
    end
    X = X1'
    rng = MersenneTwister(seed)
    s = sortperm(X[1,:])
    s = shuffle(rng, s)
    X = X[:,s]
    z = ones(NumPont)
    μ = (NumPont/Centroi)*ones(Centroi)
    g = Centroi
    Floor = floor(minimum(X)) - 1 
    Ceil = ceil(maximum(X)) + 1
    return X, z, μ, g
end

function CriaUC(X,z,μ,g,m,seed)
    d,n= size(X)
    P = zeros(g,n)
    C = reshape([],d,0)

    rng = MersenneTwister(seed)
    r = sample(rng, 1:n, g, replace=false)
    C = X[:,r]
    
    I = collect(1:g)
    for i = 1 : g
        for j = 1 : n
            P[i,j] = m(X[:,j],C[:,i],C)
        end
    end        
    U = baldinho(I,g,n,P)

    return U,C
end

function kmeans_Cap(X,z,μ,g,kmax,m,U,C,seed)
    Nrng = MersenneTwister(seed)
    Floor = floor(minimum(X)) - 1 
    Ceil = ceil(maximum(X)) + 1
    d,n = size(X)
    P = zeros(g,n)

    for k = 1 : kmax
        F = zeros(g)
        for i = 1 : g
            for j = 1 : n
                P[i,j] = m(X[:,j],C[:,i],C)
            end
        end        

        for i = 1 : g
            for j = 1 : n
                F[i] += U[i,j]*norm(X[:,j]-C[:,i])^2
            end
        end
        f1 = sum(F)
        I = sortperm(F,rev=true)
        U = baldinho(I,g,n,P)
        F1 = zeros(g)
        for i = 1 : g
            for j = 1 : n
                F1[i] += U[i,j]*norm(X[:,j]-C[:,i])^2
            end
        end   
        f2 = sum(F1)
        
        contador = 0
        rng = MersenneTwister(seed)
        while f1<f2 && contador < 200
            I = shuffle(rng, I)
            U = baldinho(I,g,n,P)
            F1 = zeros(g)
            for i = 1 : g
                for j = 1 : n
                    F1[i] += U[i,j]*norm(X[:,j]-C[:,i])^2
                end
            end   
            f2 = sum(F1)
            contador += 1
        end
        
        C1 = zeros(d,g)
            for i = 1 : g 
                C1[:,i] = U[i,1]*X[:,1]
                for j = 1 : n
                    C1[:,i] += U[i,j]*X[:,j]
                end
                C1[:,i] = C1[:,i]/sum(U[i,:])
            end
            VC = sum(C1[:,i] .∈ C for i = 1:g)

        if minimum(VC) == 1 || k == kmax
        
            MD = 10*ones(g,g)
                for i = 1 : g
                    for j = 1 : i
                        if i != j
                            MD[i,j] = norm(C[:,i]-C[:,j])                           
                        end
                    end
                end
                
            MMD = minimum(MD)
            IM = argmin(MD)
            IM = [IM[1],IM[2]]

            if MMD > 2.0
                return U, C, k, f2
            
            else
                l = 0
                while MMD < 2.0 && l ≤  g/2
                    NX = Array{Float64}(undef,2,0)
                    NXD = []
                    for j = 1 : n
                        if U[IM[1],j] == 1.0 || U[IM[2],j] == 1.0
                            NX = hcat(NX,X[:,j])
                            push!(NXD,j)
                        end
                    end
                    Nn = n*2/g
                    Nn = floor(Int, Nn)

                    MaxA = maximum(NX[1,:])
                    MaxB = maximum(NX[2,:])
                    MinA = minimum(NX[1,:])
                    MinB = minimum(NX[2,:])

                    NXX = sort(NX[1,:])
                    Q1X = quantile!(NXX, 0.25)
                    MedianaX = median!(NXX)
                    Q3X = quantile!(NXX, 0.75)
                    IQRX = Q3X - Q1X
                    UPX = Q3X + (1.5*IQRX)
                    LOX = Q1X - (1.5*IQRX)

                    NXY = sort(NX[2,:])
                    Q1Y = quantile!(NXY, 0.25)
                    MedianaX = median!(NXY)
                    Q3Y = quantile!(NXY, 0.75)
                    IQRY = Q3Y - Q1Y
                    UPY = Q3Y + (1.5*IQRY)
                    LOY = Q1Y - (1.5*IQRY)

                        for j = 1 : Nn
                            if NX[1,j] ≥ UPX || NX[1,j] ≤ LOX
                                NX[1,j] = C[1,IM[1]]
                                NX[2,j] = C[2,IM[1]]
                            elseif NX[2,j] ≥ UPY || NX[2,j] ≤ LOY
                                    NX[2,j] = C[2,IM[1]]
                                    NX[1,j] = C[1,IM[1]]
                            end
                        end    

                    NMX = zeros(Nn,Nn)
                    for i = 1 : Nn
                        for j = 1 : i
                            if i != j
                                NMX[i,j] = norm(NX[:,i]-NX[:,j])
                            end
                        end
                    end

                    INX = argmax(NMX)
                    INX = [INX[1],INX[2]]
                    NC = NX[:,INX]
                    
                    NX = Array{Float64}(undef,2,0)
                    NXD = []
                    for j = 1 : n
                        if U[IM[1],j] == 1.0 || U[IM[2],j] == 1.0
                            NX = hcat(NX,X[:,j])
                            push!(NXD,j)
                        end
                    end
                    Nn = n*2/g
                    Nn = floor(Int, Nn)

                    NP = zeros(2,Nn)
                    for i = 1 : 2
                        for j = 1 : Nn
                            NP[i,j] = m(NX[:,j],NC[:,i],NC)
                        end
                    end 
                    NU = baldinho([1,2],2,Nn,NP)

                    U[IM[1],NXD[:]] = NU[1,:]
                    U[IM[2],NXD[:]] = NU[2,:]
                    C[:,IM[1]] = NC[:,1]
                    C[:,IM[2]] = NC[:,2]

                    C = zeros(d,g)
                    for i = 1 : g 
                        C[:,i] = U[i,1]*X[:,1]
                        for j = 1 : n
                            C[:,i] += U[i,j]*X[:,j]
                        end
                        C[:,i] = C[:,i]/sum(U[i,:])
                    end

                    F3 = zeros(g)
                    for i = 1 : g
                        for j = 1 : n
                            F3[i] += U[i,j]*norm(X[:,j]-C[:,i])^2
                        end
                    end   
                    f3 = sum(F3)
                    
                    MD = 10*ones(g,g)
                        for i = 1 : g
                            for j = 1 : i
                                if i != j
                                    MD[i,j] = norm(C[:,i]-C[:,j])                           
                                end
                            end
                        end
                        
                    MMD = minimum(MD)
                    IM = argmin(MD)
                    IM = [IM[1],IM[2]]
                    l += 1
                end
                F3 = zeros(g)
                    for i = 1 : g
                        for j = 1 : n
                            F3[i] += U[i,j]*norm(X[:,j]-C[:,i])^2
                        end
                    end   
                    f3 = sum(F3)
                return U, C, k, f3
            end    
        
        else
            C = C1
        end
    end
end

function BeBCVI(U,C,X)
    nl_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0)
    minlp_solver = optimizer_with_attributes(Juniper.Optimizer,  "nl_solver"=>nl_solver)
    m = Model(minlp_solver)
    set_time_limit_sec(m, 180.0)

    x = X
    g, n = size(U)
    a, b = size(C)

    @variable(m, u[i=1:g, j=1:n], binary=true, start=U[i,j])

    @variable(m, c[h=1:a, k=1:b], start=C[h,k])

    @NLobjective(m, Min, sum(sum(u[i,j]*sum((x[t,j] - c[t,i])^2 for t = 1 : a)  for j = 1:n) for i=1:g))

    for j = 1 : n
        @constraint(m, sum(u[i,j] for i = 1 : g) == 1)
    end

    for i = 1 : g
        @constraint(m, sum(u[i,j]*z[j] for j = 1 : n) == μ[i])
    end

    #print(m)
    optimize!(m)
    Time, FOB, c, u = solve_time(m::Model), objective_value(m), JuMP.value.(c), JuMP.value.(u)

    return Time, FOB, c, u
end

function BeBSVI(U,C,X)
    nl_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0)
    minlp_solver = optimizer_with_attributes(Juniper.Optimizer,  "nl_solver"=>nl_solver)
    m = Model(minlp_solver)
    set_time_limit_sec(m, 180.0)

    x = X
    g, n = size(U)
    a, b = size(C)

    @variable(m, u[i=1:g, j=1:n], binary=true)

    @variable(m, c[h=1:a, k=1:b])

    @NLobjective(m, Min, sum(sum(u[i,j]*sum((x[t,j] - c[t,i])^2 for t = 1 : a)  for j = 1:n) for i=1:g))

    for j = 1 : n
        @constraint(m, sum(u[i,j] for i = 1 : g) == 1)
    end

    for i = 1 : g
        @constraint(m, sum(u[i,j]*z[j] for j = 1 : n) == μ[i])
    end

    print(m)
    optimize!(m)
    Time, FOB, c, u = solve_time(m::Model), objective_value(m), JuMP.value.(c), JuMP.value.(u)
    
    return Time, FOB, c, u
end