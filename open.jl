# "Import Tariffs and Global Sourcing" by Antras, Fadeev, Fort, Gutierrez, and Tintelnot

# Replication by Chase Abram and Ignacia Cuevas

# This code is not yet fully functional.

# 

# Load packages
using Parameters
using NLsolve
using Random

using LinearAlgebra

using ForwardDiff

using Plots

##

# Stores exogenous parameters
# In calibrating, we would be updating this object
@with_kw mutable struct OpenModel

    # sigma and theta
    ces = [4.0, 4.0]

    # markups
    mu = ces ./(ces .- 1)
    
    # Entry costs in goods
    f = [1.0 1.0; 1.0 1.0]

    # Expenditure on inputs relative to total sales
    alpha = 1 - 0.4517

    alpha_bar = (1/alpha)^alpha * (1/(1-alpha))^(1-alpha)
    
    # Scaled US population
    # Lus = 0.4531
    # Scaled RoW population
    # Lrow = 9.5469
    # L = [0.4531, 9.5469]
    L = [1.0 1.0]

    # # Estimated Values
    # # Production, final, RoW relative to US
    # Adrow = 0.2821
    # # Productivity, input, RoW relative to US
    # Aurow = 0.1114
    # A = [1.0 1.0; 0.2821 0.1114]
    A = [1.0 1.0; 1.0 1.0]

    # # Icberg cost, final, from US to RoW
    # taud = 3.0301
    # # Iceberg cost, input, from US to RoW
    # tauu = 2.6039
    tau = ones(2,2,2)
    # tau[1,2,1] = 3.0301
    # tau[2,1,1] = 3.0301
    
    # tau[1,2,2] = 2.6039
    # tau[2,1,2] = 2.6039

    # Tariffs
    # t[i,j,s] = tariff imposed by j on import from i in sector s
    t = zeros(2,2,2) #.+ 0.01

    # Subsidies
    v = zeros(2,2,2) #.+ 0.01
end

##


# P_ji^s (Price aggregator by export-import-sector)
function Pjis(m,p,M,j,i,s)
    return M[j,s]^(1/(1-m.ces[s]))*(1+m.t[j,i,s])*p[j,i,s]
    # return safe_power(M[j,s], 1/(1-m.ces[s]))*(1+m.t[j,i,s])*p[j,i,s]
end

#P_i (Price aggregator by import-sector)
function Pis(m,p,M,i,s)
    # tmp_sum = sum([sign(Pjis(m,p,M,j,i,s))*abs(Pjis(m,p,M,j,i,s))^(1-m.ces[s]) for j in 1:2])
    # return sign(tmp_sum)*abs(tmp_sum)^(1/(1-m.ces[s]))
    # return abs(safe_power(sum([safe_power(Pjis(m,p,M,j,i,s),(1-m.ces[s])) for j in 1:2]), 1/(1-m.ces[s])))
    return sum([Pjis(m,p,M,j,i,s)^(1-m.ces[s]) for j in 1:2])^(1/(1-m.ces[s]))
end

#mc_i^S (Marginal cost by import-sector)
function mcis(m,p,w,M,i,s)
    if s == 1
        return m.alpha_bar/m.A[i,s] * w[i]^m.alpha * Pis(m,p,M,i,2)^(1-m.alpha)
        # return m.alpha_bar/m.A[i,s] * safe_power(w[i],m.alpha) * safe_power(Pis(m,p,M,i,s),(1-m.alpha))
    else
        return w[i]/m.A[i,s]
    end
end

# y_i^s (Output by export-sector)
function yis(m,i,s)
    return (m.ces[s]-1)*m.f[i,s]
end

# Goods market eq. residual by import-sector
function goods(m,q,M,i,s)
    if s == 1
        return yis(m,i,s) - sum(m.tau[i,:,s] .* q[i,:,s])
    else
        return yis(m,i,s) - sum(M[:,1] .* m.tau[i,:,s] .* q[i,:,s])
    end
end

# Q_ji^u (Upstream quantity aggregator by export-import)
function Qjiu(m,p,w,q,M,j,i)
    return (1-m.alpha)*mcis(m,p,w,M,i,1)*(yis(m,i,1) + m.f[i,1])/(Pis(m,p,M,i,2)^(1-m.ces[2]))*Pjis(m,p,M,j,i,2)^(-m.ces[2])
end

# T_i (Tax rebates by import)
function Ti(m,p,q,M,i)
    return sum(m.t[:,i,1].*M[:,1].*p[:,i,1].*q[:,i,1] + m.t[:,i,2].*M[:,2].*M[i,1].*p[:,i,2].*q[:,i,2] - m.v[i,:,1].*M[i,1].*p[i,:,1].*q[i,:,1] - m.v[i,:,2].*M[i,2].*M[:,1].*p[i,:,2].*q[i,:,2])
end

# Demand eq. residual by export-import-sector
function demand(m,p,w,q,M,j,i,s)
    if s == 1
        return q[j,i,s] - (w[i]*m.L[i] + Ti(m,p,q,M,i))/(Pis(m,p,M,i,s)^(1-m.ces[s])) * ((1 + m.t[j,i,s])*p[j,i,s])^(-m.ces[s])
    else
        return q[j,i,s] - Qjiu(m,p,w,q,M,j,i)*((1 + m.t[j,i,s])*p[j,i,s]/Pjis(m,p,M,j,i,s))^(-m.ces[s])
    end
end

# Price eq. residual by import-export-sector
function prices(m,p,w,M,i,j,s)
    return p[i,j,s] - m.mu[s]*m.tau[i,j,s]*mcis(m,p,w,M,i,s)/(1+m.v[i,j,s])
end

# Labor levels by import-sector
function labor(m,p,w,q,M,i,s)
    if s == 1
        return m.alpha*mcis(m,p,w,M,i,s)*(yis(m,i,s) + m.f[i,s])/w[i]
    else
        return (yis(m,i,s) + m.f[i,s])/m.A[i,s]
    end
end

# Labor market clearing eq. residual by import
function lmc(m,p,w,q,M,i)
    return m.L[i] - sum([M[i,s]*labor(m,p,w,q,M,i,s) for s in 1:2])
end

# Used to keep variables positive and/or well behaved
# This should not be needed
function transform(z)
    
    # x -> infinity => f(x) -> x
    # x -> -infinity => f(x) -> 0
    # return log(exp(z) + 1)
    
    # Smooth and simple
    return exp.(z)
end

function safe_power(x,p)
    return sign(x)*abs(x)^p
end

function agg_prices(m,p,P,M,i,s)
    return P[i,s] - sum([Pjis(m,p,M,j,i,s)^(1-m.ces[s]) for j in 1:2])^(1/(1-m.ces[s]))
end

# Below this line code will still be quite messy
#######################################################
##
# All equations at once

function eq_all!(F,x,m)
    # Rename variables
    p = reshape(x[1:8], (2,2,2))
    w = x[9:10]
    q = reshape(x[11:18], (2,2,2))
    M = reshape(x[19:22], (2,2))

    # p = reshape(x[1:8], (2,2,2))
    # w = [1.0 x[9]]
    # q = reshape(x[10:17], (2,2,2))
    # M = reshape(x[18:21], (2,2))

    p = transform.(p)
    w = transform.(w)
    M = transform.(M)
    q = transform.(q)

    F_iter = 0

    # Wage normalization
    F_iter += 1
    F[F_iter] = w[1] - 1.0

    for i in 1:2
        for j in 1:2
            for s in 1:2
                F_iter += 1
                F[F_iter] = demand(m,p,w,q,M,j,i,s)
            end
        end
    end

    for i in 1:2
        for j in 1:2
            for s in 1:2
                F_iter += 1
                F[F_iter] = prices(m,p,w,M,i,j,s)
            end
        end
    end

    for i in 1:2
        for s in 1:2
            F_iter += 1
            F[F_iter] = goods(m,q,M,i,s)
        end
    end

    for i in 2:2
        F_iter += 1
        F[F_iter] = lmc(m,p,w,q,M,i)
    end
end


m0 = OpenModel()

F0 = zeros(22)
# x0 = zeros(22) #.+ 0.0im
# x0[1] = 1.0
# x0 = rand(22)
x0 = ones(22)

# F0 = zeros(21)
# x0 = zeros(21)

function solve_open(m, x_init)
    return nlsolve((F,x) -> eq_all!(F,x,m), x_init, #autodiff = :forward,
    show_trace = true, method = :trust_region, iterations = 150, xtol=1e-16)
end


# Solve system
soln = solve_open(m0, x0)
p_soln = reshape(soln.zero[1:8], (2,2,2))
w_soln = soln.zero[9:10]
M_soln = reshape(soln.zero[11:14], (2,2))
q_soln = reshape(soln.zero[15:22], (2,2,2))

# p_soln = reshape(soln.zero[1:8], (2,2,2))
# w_soln = [0.0 soln.zero[9]]
# q_soln = reshape(soln.zero[10:17], (2,2,2))
# M_soln = reshape(soln.zero[18:21], (2,2))

# p_soln = transform.(p_soln)
# w_soln = transform.(w_soln)
# M_soln = transform.(M_soln)
# q_soln = transform.(q_soln)

println("soln")
println("    p: ", p_soln)
println("    w: ", w_soln)
println("    M: ", M_soln)
println("    q: ", q_soln)

println("all residuals")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    demand (",j,",",i,",",s,"): ", demand(m0,p_soln,w_soln,q_soln,M_soln,j,i,s))
        end
    end
end

println("    ---")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    prices (",i,",",j,",",s,"): ", prices(m0,p_soln,w_soln,M_soln,i,j,s))
        end
    end
end

println("    ---")

for i in 1:2
    for s in 1:2
        println("    goods (",i,",",s,"): ", goods(m0,q_soln,M_soln,i,s))
    end
end

println("    ---")

for i in 1:2
    println("    lmc (",i,"): ", lmc(m0,p_soln,w_soln,q_soln,M_soln,i))
end


#######################################################
##
# One piece at a time

# Just Quantities

function eq_q!(F,x,m,p,w,M)
    # Rename variables
    q = reshape(x[1:8], (2,2,2))

    # q = transform.(q)

    F_iter = 0

    for i in 1:2
        for j in 1:2
            for s in 1:2
                F_iter += 1
                F[F_iter] = demand(m,p,w,q,M,j,i,s)
            end
        end
    end
end


function solve_q(m,p,w,M,x_init)
    return nlsolve((F,x) -> eq_q!(F,x,m,p,w,M), x_init, #autodiff = :forward,
    show_trace = false, method = :newton, iterations = 500, ftol=1e-16)
end


F0 = zeros(8)
x0 = randn(8)

m0 = OpenModel()
p0 = ones(2,2,2)
w0 = ones(2)
M0 = ones(2,2)


# Solve system
soln = solve_q(m0,p0,w0,M0,x0)
q_soln = reshape(soln.zero[1:8], (2,2,2))

# q_soln = transform.(q_soln)

println("soln")
println("    p: ", p0)
println("    w: ", w0)
println("    M: ", M0)
println("    q: ", q_soln)

println("all residuals")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    demand (",j,",",i,",",s,"): ", demand(m0,p0,w0,q_soln,M0,j,i,s))
        end
    end
end

println("    ---")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    prices (",i,",",j,",",s,"): ", prices(m0,p0,w0,M0,i,j,s))
        end
    end
end

println("    ---")

for i in 1:2
    for s in 1:2
        println("    goods (",i,",",s,"): ", goods(m0,q_soln,M0,i,s))
    end
end

println("    ---")

for i in 1:2
    println("    lmc (",i,"): ", lmc(m0,p0,w0,q_soln,M0,i))
end

#######################################################
##

# Just Prices (and quantities)

function eq_p!(F,x,m,w,M)
    # Rename variables
    p = reshape(x[1:8], (2,2,2))
    # p = transform.(p)

    # q_guess = reshape(ones(8), (2,2,2))
    # q = solve_q(m,p,w,M,q_guess).zero
    # q = transform.(q)

    F_iter = 0

    for i in 1:2
        for j in 1:2
            for s in 1:2
                F_iter += 1
                F[F_iter] = prices(m,p,w,M,i,j,s)
            end
        end
    end
end


function solve_p(m,w,M,x_init)
    return nlsolve((F,x) -> eq_p!(F,x,m,w,M), x_init, #autodiff = :forward,
    show_trace = false, method = :newton, iterations = 500, ftol=1e-16)
end


F0 = zeros(8)
x0 = randn(8)
x0 = ones(8)

m0 = OpenModel()
w0 = ones(2)
M0 = ones(2,2)


# Solve system
soln = solve_p(m0,w0,M0,x0)
p_soln = abs.(reshape(soln.zero[1:8], (2,2,2)))
# p_soln = transform.(p_soln)

q_guess = reshape(ones(8), (2,2,2))
q_soln = abs.(solve_q(m0,p_soln,w0,M0,q_guess).zero)
# q_soln = transform.(q_soln)

println("soln")
println("    p: ", p_soln)
println("    w: ", w0)
println("    M: ", M0)
println("    q: ", q_soln)

println("all residuals")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    demand (",j,",",i,",",s,"): ", demand(m0,p_soln,w0,q_soln,M0,j,i,s))
        end
    end
end

println("    ---")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    prices (",i,",",j,",",s,"): ", prices(m0,p_soln,w0,M0,i,j,s))
        end
    end
end

println("    ---")

for i in 1:2
    for s in 1:2
        println("    goods (",i,",",s,"): ", goods(m0,q_soln,M0,i,s))
    end
end

println("    ---")

for i in 1:2
    println("    lmc (",i,"): ", lmc(m0,p_soln,w0,q_soln,M0,i))
end

#######################################################
##

# Try M (does not work)

function eq_M!(F,x,m,w)
    # Rename variables
    M = reshape(x,(2,2))
    M = transform.(M)
    # println("M: ", M)

    p_guess = reshape(ones(8),(2,2,2))
    p = abs.(solve_p(m,w,M,p_guess).zero)
    p = reshape(p, (2,2,2))
    # p = transform.(p)
    # println("p: ", p)

    q_guess = reshape(ones(8), (2,2,2))
    q = abs.(solve_q(m,p,w,M,q_guess).zero)
    q = reshape(q, (2,2,2))
    # q = transform.(q)
    # println("q: ", q)

    F_iter = 0

    for i in 1:2
        for s in 1:2
            F_iter += 1
            F[F_iter] = goods(m,q,M,i,s)
        end
    end
end


function solve_M(m,w,x_init)
    return nlsolve((F,x) -> eq_M!(F,x,m,w), x_init, #autodiff = :forward,
    show_trace = false, method = :trust_region, iterations = 50, ftol=1e-16)
end


F0 = zeros(4)
x0 = ones(4) #.+ 0.0im

m0 = OpenModel()
w0 = ones(2)
# w0[2] = 1.0

# Solve system
soln = solve_M(m0,w0,x0)
M_soln = reshape(soln.zero, (2,2))
M_soln = transform.(M_soln)

p_guess = reshape(rand(8), (2,2,2))
p_soln = solve_p(m0,w0,M_soln,p_guess).zero
# p_soln = transform.(p_soln)

q_guess = reshape(rand(8), (2,2,2))
q_soln = solve_q(m0,p_soln,w0,M_soln,q_guess).zero
# q_soln = transform.(q_soln)

println("soln")
println("    p: ", p_soln)
println("    w: ", w0)
println("    M: ", M_soln)
println("    q: ", q_soln)

println("all residuals")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    demand (",j,",",i,",",s,"): ", demand(m0,p_soln,w0,q_soln,M_soln,j,i,s))
        end
    end
end

println("    ---")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    prices (",i,",",j,",",s,"): ", prices(m0,p_soln,w0,M_soln,i,j,s))
        end
    end
end

println("    ---")

for i in 1:2
    for s in 1:2
        println("    goods (",i,",",s,"): ", goods(m0,q_soln,M_soln,i,s))
    end
end

println("    ---")

for i in 1:2
    println("    lmc (",i,"): ", lmc(m0,p_soln,w0,q_soln,M_soln,i))
end

#######################################################
##

# M and w?

# function eq_Mw!(F,x,m)
#     # Rename variables
#     w = x[1:2]
#     M = reshape(x[3:6],(2,2))

#     w = transform.(w)
#     M = transform.(M)
#     # println("M: ", M)

#     p_guess = reshape(ones(8),(2,2,2))
#     p = solve_p(m,w,M,p_guess).zero
#     p = reshape(p, (2,2,2))
#     p = transform.(p)
#     # println("p: ", p)

#     q_guess = reshape(ones(8), (2,2,2))
#     q = solve_q(m,p,w,M,q_guess).zero
#     q = reshape(q, (2,2,2))
#     q = transform.(q)
#     # println("q: ", q)

#     F_iter = 0

#     F_iter += 1
#     F[F_iter] = w[1] - 1.0

#     for i in 1:2
#         for s in 1:2
#             F_iter += 1
#             F[F_iter] = goods(m,q,M,i,s)
#         end
#     end

#     F_iter += 1
#     F[F_iter] = lmc(m,p,w,q,M,2)
    
# end


# function solve_Mw(m,x_init)
#     return nlsolve((F,x) -> eq_Mw!(F,x,m), x_init, #autodiff = :forward,
#     show_trace = true, method = :newton, iterations = 50, xtol=1e-16)
# end


# F0 = zeros(6)
# x0 = zeros(6)

# m0 = OpenModel()

# # Solve system
# soln = solve_Mw(m0,x0)
# println("soln details:", soln)

# w_soln = soln.zero[1:2]
# w_soln = transform.(w_soln)

# M_soln = reshape(soln.zero[3:6], (2,2))
# M_soln = transform.(M_soln)

# p_guess = reshape(ones(8), (2,2,2))
# p_soln = solve_p(m0,w0,M_soln,p_guess).zero
# p_soln = transform.(p_soln)

# q_guess = reshape(ones(8), (2,2,2))
# q_soln = solve_q(m0,p_soln,w0,M_soln,q_guess).zero
# q_soln = transform.(q_soln)

# println("soln")
# println("    p: ", p_soln)
# println("    w: ", w_soln)
# println("    M: ", M_soln)
# println("    q: ", q_soln)

# println("all residuals")

# for i in 1:2
#     for j in 1:2
#         for s in 1:2
#             println("    demand (",j,",",i,",",s,"): ", demand(m0,p_soln,w_soln,q_soln,M_soln,j,i,s))
#         end
#     end
# end

# println("    ---")

# for i in 1:2
#     for j in 1:2
#         for s in 1:2
#             println("    prices (",i,",",j,",",s,"): ", prices(m0,p_soln,w_soln,M_soln,i,j,s))
#         end
#     end
# end

# println("    ---")

# for i in 1:2
#     for s in 1:2
#         println("    goods (",i,",",s,"): ", goods(m0,q_soln,M_soln,i,s))
#     end
# end

# println("    ---")

# for i in 1:2
#     println("    lmc (",i,"): ", lmc(m0,p_soln,w_soln,q_soln,M_soln,i))
# end


#######################################################
##

# w

# Try w (does not work)

function eq_w!(F,x,m)
    # Rename variables
    w = x
    w = transform.(w)
    
    M_guess = reshape(ones(4),(2,2))
    M = solve_M(m,w,M_guess).zero
    M = reshape(M, (2,2))
    # M = transform.(M)

    p_guess = reshape(ones(8),(2,2,2))
    p = solve_p(m,w,M,p_guess).zero
    p = reshape(p, (2,2,2))
    # p = transform.(p)
    # println("p: ", p)

    q_guess = reshape(ones(8), (2,2,2))
    q = solve_q(m,p,w,M,q_guess).zero
    q = reshape(q, (2,2,2))
    # q = transform.(q)
    # println("q: ", q)

    F_iter = 0

    F_iter += 1
    F[F_iter] = w[1] - 1.0

    F_iter += 1
    F[F_iter] = lmc(m,p,w,q,M,2)
end


function solve_w(m,x_init)
    return nlsolve((F,x) -> eq_w!(F,x,m), x_init, #autodiff = :forward,
    show_trace = true, method = :trust_region, iterations = 50, xtol=1e-16)
end


F0 = zeros(2)
x0 = ones(2)

m0 = OpenModel()

# Solve system
soln = solve_w(m0,x0)
w_soln = soln.zero
w_soln = transform.(w_soln)

M_guess = reshape(ones(4), (2,2))
M_soln = solve_M(m0,w_soln,M_guess).zero
# M_soln = transform.(M_soln)

p_guess = reshape(ones(8), (2,2,2))
p_soln = solve_p(m0,w_soln,M_soln,p_guess).zero
# p_soln = transform.(p_soln)

q_guess = reshape(ones(8), (2,2,2))
q_soln = solve_q(m0,p_soln,w_soln,M_soln,q_guess).zero
# q_soln = transform.(q_soln)

println("soln")
println("    p: ", p_soln)
println("    w: ", w_soln)
println("    M: ", M_soln)
println("    q: ", q_soln)

println("all residuals")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    demand (",j,",",i,",",s,"): ", demand(m0,p_soln,w_soln,q_soln,M_soln,j,i,s))
        end
    end
end

println("    ---")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    prices (",i,",",j,",",s,"): ", prices(m0,p_soln,w_soln,M_soln,i,j,s))
        end
    end
end

println("    ---")

for i in 1:2
    for s in 1:2
        println("    goods (",i,",",s,"): ", goods(m0,q_soln,M_soln,i,s))
    end
end

println("    ---")

for i in 1:2
    println("    lmc (",i,"): ", lmc(m0,p_soln,w_soln,q_soln,M_soln,i))
end

##

# Time to code a solver!

# Newton-Raphson for finding zeros of f
function newton(f, x0::AbstractVector{T}; maxit = 1000, ftol = 1e-15, xtol = 1e-15) where T
    it = 0
    # diff = Inf
    fdev = Inf
    xdev = Inf

    xold = x0
    xnew = NaN

    # weight = 1.0

    while it < maxit && (fdev > ftol || xdev > xtol)
        # println("Newton it: ", it)
        # println("    det(AD): ", det(ForwardDiff.jacobian(f,xold)))
        # println("    AD: ", ForwardDiff.jacobian(f,xold))
        # println("    f(x): ", f(xold))
        Jf = ForwardDiff.jacobian(f, xold)
        
        # if det(Jf) == 0
        #     println("det(Jf) = 0")
        #     return xold
        # end

        xnew = xold -  Jf \ f(xold)

        # xnew = (1 - weight) .* xold + weight .* xnew

        weight = 0.5
        weight_it = 0
        max_weight_it = 10000
        while sum(xnew .< 0) > 0 && weight_it < max_weight_it
            # println("weight_it: ", weight_it)
            xnew = weight*xnew + (1-weight)*xold
            weight_it += 1
        end

        # if weight_it == max_weight_it
        #     xnew = 1/2 .* xold
        #     # println("max_weight_it reached, using xnew = ", xnew)
        # end

        # xnew = (xnew .>= 0) .* xnew .+ 0.1 .* (xnew .< 0)
        # println("    xnew: ", xnew)

        fdev = maximum(abs.(f(xnew)))
        xdev = maximum(abs.(xnew - xold))
        xold = xnew
        it += 1
    end

    if it == maxit
        println("Max Newton iterations reached")
    end

    # println("final fdev: ", diff)
    # println("final eval: ", f(xnew))

    return xnew
end

h(x) = [2*x[1] - x[2] - x[3], (x[2] - x[3])^2, (x[3] - 6)^2] 

newton(h, [0,0,1])

#######################################################
##
# One piece at a time

# Just Quantities

function eq_q(x::AbstractVector{T},m,p,w,M) where T
    # println("q call, maximum(abs.(q)): ", maximum(abs.(x)))
    
    # Rename variables
    q = reshape(x[1:8], (2,2,2))

    # q = transform.(q)

    F = ones(T,length(x))

    F_iter = 0

    for i in 1:2
        for j in 1:2
            for s in 1:2
                F_iter += 1
                F[F_iter] = demand(m,p,w,q,M,j,i,s)
            end
        end
    end

    return F
end


x0 = randn(8).* 100

m0 = OpenModel()
p0 = ones(2,2,2)
w0 = ones(2)
M0 = ones(2,2)

eq_q_simp(z) = eq_q(z,m0,p0,w0,M0)

q_soln = reshape(newton(eq_q_simp,x0), (2,2,2))

println("soln")
println("    p: ", p0)
println("    w: ", w0)
println("    M: ", M0)
println("    q: ", q_soln)

println("all residuals")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    demand (",j,",",i,",",s,"): ", demand(m0,p0,w0,q_soln,M0,j,i,s))
        end
    end
end

println("    ---")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    prices (",i,",",j,",",s,"): ", prices(m0,p0,w0,M0,i,j,s))
        end
    end
end

println("    ---")

for i in 1:2
    for s in 1:2
        println("    goods (",i,",",s,"): ", goods(m0,q_soln,M0,i,s))
    end
end

println("    ---")

for i in 1:2
    println("    lmc (",i,"): ", lmc(m0,p0,w0,q_soln,M0,i))
end

#######################################################
##

# Just Prices (and quantities)

function eq_p(x::AbstractVector{T},m,w,M) where T
    # println("p call, maximum(abs.(p)): ", maximum(abs.(x)))
    # Rename variables
    p = reshape(x[1:8], (2,2,2))
    p = transform.(p)

    F = ones(T,length(x))

    F_iter = 0

    for i in 1:2
        for j in 1:2
            for s in 1:2
                F_iter += 1
                F[F_iter] = prices(m,p,w,M,i,j,s)
            end
        end
    end

    return F
end


# x0 = randn(8)
x0 = ones(8)

m0 = OpenModel()
w0 = ones(2)
M0 = ones(2,2)

eq_p_simp(z) = eq_p(z,m0,w0,M0)
p_soln = reshape(newton(eq_p_simp,x0), (2,2,2))
p_soln = transform(p_soln)

eq_q_simp(z) = eq_q(z,m0,p_soln,w0,M0)
q_soln = reshape(newton(eq_q_simp,x0), (2,2,2))


println("soln")
println("    p: ", p_soln)
println("    w: ", w0)
println("    M: ", M0)
println("    q: ", q_soln)

println("all residuals")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    demand (",j,",",i,",",s,"): ", demand(m0,p_soln,w0,q_soln,M0,j,i,s))
        end
    end
end

println("    ---")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    prices (",i,",",j,",",s,"): ", prices(m0,p_soln,w0,M0,i,j,s))
        end
    end
end

println("    ---")

for i in 1:2
    for s in 1:2
        println("    goods (",i,",",s,"): ", goods(m0,q_soln,M0,i,s))
    end
end

println("    ---")

for i in 1:2
    println("    lmc (",i,"): ", lmc(m0,p_soln,w0,q_soln,M0,i))
end

#######################################################
##

# Try M

function eq_M(x::AbstractVector{T},m,w) where T
    # Rename variables
    # println("M call, maximum(abs.(M)): ", maximum(abs.(x)))
    M = reshape(x,(2,2))
    # M = transform.(M)
    # println("M: ", M)

    p_guess = ones(T,8)
    # p = abs.(solve_p(m,w,M,p_guess).zero)
    p = reshape(newton(z -> eq_p(z,m,w,M),p_guess), (2,2,2))
    # p = transform(p)

    # p = ones(T,8)
    # p = reshape(p, (2,2,2))
    # p = transform.(p)
    # println("p: ", p)

    q_guess = ones(T,8)
    q = reshape(newton(z -> eq_q(z,m,p,w,M),q_guess), (2,2,2))
    # q = abs.(solve_q(m,p,w,M,q_guess).zero)
    # q = transform(q)
    
    # q = ones(T,8)
    # q = reshape(q, (2,2,2))
    # q = transform.(q)
    # println("q: ", q)

    F = ones(T, length(x))
    F_iter = 0

    for i in 1:2
        for s in 1:2
            F_iter += 1
            F[F_iter] = goods(m,q,M,i,s)
        end
    end

    # for s in 1:2
    #     F_iter += 1
    #     F[F_iter] = goods(m,q,M,1,s)
    # end

    # for i in 1:2
    #     F_iter += 1
    #     F[F_iter] = lmc(m,p,w,q,M,i)
    # end

    return F
end


# function solve_M(m,w,x_init)
#     return nlsolve((F,x) -> eq_M!(F,x,m,w), x_init, #autodiff = :forward,
#     show_trace = false, method = :trust_region, iterations = 50, ftol=1e-16)
# end


x0 = ones(4) #.+ 0.0im
# x0 = rand(4)

m0 = OpenModel()
m0.tau[1,2,1] = 3.0301
m0.tau[2,1,1] = 3.0301

m0.tau[1,2,2] = 2.6039
m0.tau[2,1,2] = 2.6039

w0 = ones(2)
# w0 = rand(2)
# w0[2] = 1.0

eq_M_simp(z) = eq_M(z,m0,w0)
M_soln = reshape(newton(eq_M_simp,x0), (2,2))

eq_p_simp(z) = eq_p(z,m0,w0,M_soln)
p_guess = rand(8)
p_soln = reshape(newton(eq_p_simp,p_guess), (2,2,2))

q_guess = rand(8)
eq_q_simp(z) = eq_q(z,m0,p_soln,w0,M_soln)
q_soln = reshape(newton(eq_q_simp, q_guess), (2,2,2))


println("soln")
println("    p: ", p_soln)
println("    w: ", w0)
println("    M: ", M_soln)
println("    q: ", q_soln)

println("all residuals")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    demand (",j,",",i,",",s,"): ", demand(m0,p_soln,w0,q_soln,M_soln,j,i,s))
        end
    end
end

println("    ---")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    prices (",i,",",j,",",s,"): ", prices(m0,p_soln,w0,M_soln,i,j,s))
        end
    end
end

println("    ---")

for i in 1:2
    for s in 1:2
        println("    goods (",i,",",s,"): ", goods(m0,q_soln,M_soln,i,s))
    end
end

println("    ---")

for i in 1:2
    println("    lmc (",i,"): ", lmc(m0,p_soln,w0,q_soln,M_soln,i))
end

# Solve for m as a function of prices and Prices?


#######################################################
##
# All equations at once

function eq_all(x::AbstractVector{T},m) where T
    # println("all call")
    # Rename variables
    p = reshape(x[1:8], (2,2,2))
    w = x[9:10]
    q = reshape(x[11:18], (2,2,2))
    M = reshape(x[19:22], (2,2))

    # p = transform.(p)
    # w = transform.(w)
    # M = transform.(M)
    # q = transform.(q)

    F = ones(T, length(x))
    F_iter = 0

    # Wage normalization
    F_iter += 1
    F[F_iter] = w[1] - 1.0

    for i in 1:2
        for j in 1:2
            for s in 1:2
                F_iter += 1
                F[F_iter] = demand(m,p,w,q,M,j,i,s)
            end
        end
    end

    for i in 1:2
        for j in 1:2
            for s in 1:2
                F_iter += 1
                F[F_iter] = prices(m,p,w,M,i,j,s)
            end
        end
    end

    for i in 1:2
        for s in 1:2
            F_iter += 1
            F[F_iter] = goods(m,q,M,i,s)
        end
    end

    for i in 2:2
        F_iter += 1
        F[F_iter] = lmc(m,p,w,q,M,i)
    end

    return F
end


m0 = OpenModel()

x0 = ones(22)

soln = newton(z -> eq_all(z,m0), x0)
println("raw soln: ", soln)
p_soln = reshape(soln[1:8], (2,2,2))
w_soln = soln[9:10]
q_soln = reshape(soln[11:18], (2,2,2))
M_soln = reshape(soln[19:22], (2,2))

# function solve_open(m, x_init)
#     return nlsolve((F,x) -> eq_all!(F,x,m), x_init, #autodiff = :forward,
#     show_trace = true, method = :trust_region, iterations = 150, xtol=1e-16)
# end


# # Solve system
# soln = solve_open(m0, x0)
# p_soln = reshape(soln.zero[1:8], (2,2,2))
# w_soln = soln.zero[9:10]
# M_soln = reshape(soln.zero[11:14], (2,2))
# q_soln = reshape(soln.zero[15:22], (2,2,2))

# p_soln = reshape(soln.zero[1:8], (2,2,2))
# w_soln = [0.0 soln.zero[9]]
# q_soln = reshape(soln.zero[10:17], (2,2,2))
# M_soln = reshape(soln.zero[18:21], (2,2))

# p_soln = transform.(p_soln)
# w_soln = transform.(w_soln)
# M_soln = transform.(M_soln)
# q_soln = transform.(q_soln)

println("soln")
println("    p: ", p_soln)
println("    w: ", w_soln)
println("    M: ", M_soln)
println("    q: ", q_soln)

println("all residuals")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    demand (",j,",",i,",",s,"): ", demand(m0,p_soln,w_soln,q_soln,M_soln,j,i,s))
        end
    end
end

println("    ---")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    prices (",i,",",j,",",s,"): ", prices(m0,p_soln,w_soln,M_soln,i,j,s))
        end
    end
end

println("    ---")

for i in 1:2
    for s in 1:2
        println("    goods (",i,",",s,"): ", goods(m0,q_soln,M_soln,i,s))
    end
end

println("    ---")

for i in 1:2
    println("    lmc (",i,"): ", lmc(m0,p_soln,w_soln,q_soln,M_soln,i))
end

##
m0 = OpenModel()
p0 = ones(2,2,2)
M0 = ones(2,2)
Pjis(m0, p0, M0, 1, 1, 1)
Pis(m0,p0,M0,1,2)

2^(1/(1-m0.ces[2]))

##

# x = (w_F, masses, )?

# Wage gives upstream prices

# Wage + upstream prices + masses gives downstream prices

# Wage + all prices + masses give all quantities

# System: Use Goods Mkt and 1 LMC to pin wage and masses, then rest can be determined?

# Upstream prices
function up_pij(m, w, i, j)
    return m.mu[2]*m.tau[i,j,2]/(1 + m.v[i,j,2]) * w[i]/m.A[i,2]
end

# Upstream export-import
function Pjiu(m,w,M,j,i)
    return M[j,2]^(1/(1-m.ces[2])) * (1 + m.t[j,i,2]) * up_pij(m,w,j,i)
end

# Upstream import
function Piu(m,w,M,i)
    return sum([Pjiu(m,w,M,j,i)^(1-m.ces[2]) for j in 1:2])^(1/(1-m.ces[2]))
end

# Downstream prices
function down_pij(m,w,M,i,j)
    return m.mu[1]*m.tau[i,j,1]/(1 + m.v[i,j,1]) * m.alpha_bar/m.A[i,1] * w[i]^m.alpha_bar * Piu(m,w,M,i)^(1-m.alpha)
end

# Downstream export-import
function Pjid(m,w,M,j,i)
    return M[j,1]^(1/(1-m.ces[1])) * (1 + m.t[j,i,1]) * down_pij(m,w,M,j,i)
end

# Downstream import
function Pid(m,w,M,i)
    return sum([Pjid(m,w,M,j,i)^(1-m.ces[1]) for j in 1:2])^(1/(1-m.ces[1]))
end

# T_i (Tax rebates by import)
function Ti(m,w,q,M,i)
    return sum([m.t[j,i,1].*M[j,1].*down_pij(m,w,M,j,i).*q[j,i,1] + m.t[j,i,2].*M[j,2].*M[i,1].*up_pij(m, w, j, i).*q[j,i,2] - m.v[i,j,1].*M[i,1].*down_pij(m,w,M,i,j).*q[i,j,1] - m.v[i,j,2].*M[i,2].*M[j,1].*up_pij(m, w, i, j).*q[i,j,2] for j in 1:2])
end

function Qjiu(m,w,M,j,i)
    return (1 - m.alpha)*m.alpha_bar/m.A[i,1]*w[i]^m.alpha * Piu(m,w,M,i)^(-m.alpha)*m.ces[1]*m.f[i,1]*(Pjiu(m,w,M,j,i)/Piu(m,w,M,i))^(-m.ces[2])
end

# Solve quantity system (which will now only be function of wages and masses)
function q_residual(m,w,q,M,j,i,s)
    if s == 1
        return q[j,i,s] - (w[i]*m.L[i] + Ti(m,w,q,M,i))/Pid(m,w,M,i) * ((1 + m.t[j,i,s])*down_pij(m,w,M,j,i)/Pid(m,w,M,i))^(-m.ces[s])
    else
        return q[j,i,s] - Qjiu(m,w,M,j,i)*((1 + m.t[j,i,s])*up_pij(m,w,j,i)/Pjiu(m,w,M,j,i))^(-m.ces[s])
    end
end

function q_residuals(m,w,q::AbstractVector{T}, M) where T
    
    F = ones(T, length(q))
    F_iter = 0

    q = reshape(q,(2,2,2))

    # println("test inside")

    for j in 1:2
        for i in 1:2
            for s in 1:2
                F_iter += 1
                F[F_iter] = q_residual(m,w,q,M,j,i,s)
                # F[F_iter] = 0.0
            end
        end
    end

    return F
end

function find_q(m,w,M::AbstractMatrix{T}) where T
    q_init = ones(T,8)

    return reshape(newton(z -> q_residuals(m,w,z,M), q_init), (2,2,2))
end

function goods_residual(m,w,M::AbstractArray{T},i,s) where T
    if s == 1
        return (m.ces[s] - 1)*m.f[i,s] - sum([m.tau[i,j,s]*find_q(m,w,M)[i,j,s] for j in 1:2])
    else
        return (m.ces[s] - 1)*m.f[i,s] - sum([M[j,1]*m.tau[i,j,s]*find_q(m,w,M)[i,j,s] for j in 1:2])
    end
end

function labor(m,w,M,i,s)
    if s == 1
        return m.alpha*m.alpha_bar/m.A[i,s]*w[i]^m.alpha * Piu(m,w,M,i)^(1 - m.alpha) * m.ces[s]*m.f[i,s]/w[i]
    else
        return m.ces[s]*m.f[i,s]/m.A[i,s]
    end
end

function labor_residual(m,w::AbstractVector{T},M,i) where T
    return m.L[i] - sum([M[i,s]*labor(m,w,M,i,s) for s in 1:2])
end

function labor_residual2(m,w2,M)
    return m.L[2] - sum([M[2,s]*labor(m,[1.0 w2],M,2,s) for s in 1:2])
end

function goods_residuals(m,w,M::AbstractVector{T}) where T
    
    F = ones(T, length(M))
    F_iter = 0

    M = reshape(M,(2,2))

    for s in 1:2
        for i in 1:2
            F_iter += 1
            F[F_iter] = goods_residual(m,w,M,i,s)
            # F[F_iter] = goods_residual(m,ones(2,2,2),M,i,s)
        end
    end

    return F
end

function labor_residuals(m,w::AbstractVector{T}) where T
    F = ones(T,length(w))
    F_iter = 0
    
    F_iter = 1
    F[F_iter] = w[1] - 1.0

    F_iter += 1
    F[F_iter] = labor_residual(m,w,find_M(m,w),2)

    return F
end

function lmgm(m,w,M::AbstractVector{T}) where T

    M = reshape(M, (2,2))
    F = ones(T,length(M))
    F_iter = 0

    for i in 1:2
        F_iter += 1
        F[F_iter] = labor_residual(m,w,M,i)
    end

    for i in 1:2
        F_iter += 1
        F[F_iter] = goods_residual(m,w,M,i,2)
    end

    return F
end

function find_M(m,w::AbstractVector{T}) where T
    M_init = ones(T,4)

    return reshape(newton(z -> goods_residuals(m,w,z), M_init), (2,2))
end

function find_w(m)
    w_init = ones(2) .+ .01
    return newton(z -> labor_residuals(m,z), w_init)
end


m0 = OpenModel()
# m0.tau[1,2,1] = 3.0301
# m0.tau[2,1,1] = 3.0301

# m0.tau[1,2,2] = 2.6039
# m0.tau[2,1,2] = 2.6039

m0.tau[1,2,1] = 1.001
m0.tau[2,1,1] = 1.001

m0.tau[1,2,2] = 1.001
m0.tau[2,1,2] = 1.001


# m0.tau[1,2,1] = 100
# m0.tau[2,1,1] = 100

# m0.tau[1,2,2] = 100
# m0.tau[2,1,2] = 100


w0 = ones(2)
# w0[2] = 1.1
M0 = ones(2,2)
# q0 = ones(2,2,2)

# up_pij(m0, w0, 2, 2)
# Pjiu(m0,w0,M0,1,2)
# Piu(m0,w0,M0,2)
# down_pij(m0,w0,M0,2,1)
# Pjid(m0,w0,M0,2,2)
# Pid(m0,w0,M0,2)
# Ti(m0,w0,q0,M0,2)
# Qjiu(m0,w0,M0,2,2)
# q_residual(m0,w0,q0,M0,2,2,2)
# q_residuals(m0,w0,ones(8),M0)
# find_q(m0,w0,M0)
# goods_residual(m0,find_q(m0,w0,M0),M0,1,2)
# labor(m0,w0,M0,1,2)
# labor_residual(m0,w0,M0,2)
# goods_residuals(m0,w0,ones(4))
# find_M(m0,w0)
# det(ForwardDiff.jacobian(z -> goods_residuals(m0,w0,z), ones(4)))
# labor_residuals(m0,w0)



# find_w(m0,M0)
# labor_residual2(m0,1,M0 .* 1/10)
# lmgm(m0,w0,ones(4))
# ForwardDiff.jacobian(z -> lmgm(m0,w0,z), rand(4))

# n = 100
# Ms =[1.0 .* ones(2,2) for k in 1:n]
# for k in 1:n
#     Ms[k][1,2] = 1e-10 * k
# end
# # # plot([Ms[k][1,1] for k in 1:n], [labor_residual(m0,w0,Ms[k],1) for k in 1:n])
# # # plot([Ms[k][1,2] for k in 1:n], [labor(m0,w0,Ms[k],1,1) for k in 1:n])

# # plot([Ms[k][1,2] for k in 1:n], [labor_residual(m0,w0,Ms[k],1) for k in 1:n])
# plot([Ms[k][1,2] for k in 1:n], [goods_residual(m0,w0,Ms[k],1,1) for k in 1:n])

println("Solving for w...")
w_soln = find_w(m0)
println("Solving for M...")
M_soln = find_M(m0, w_soln)
println("Solving for q...")
q_soln = find_q(m0, w_soln, M_soln)
println("Solving for p...")
p_soln = zeros(2,2,2)
for j in 1:2
    for i in 1:2
        for s in 1:2
            if s == 1
                p_soln[i,j,s] = up_pij(m0, w_soln,i,j)
            else
                p_soln[i,j,s] = down_pij(m0,w_soln,M_soln,i,j)
            end
        end
    end
end


println("soln")
println("    p: ", p_soln)
println("    w: ", w_soln)
println("    M: ", M_soln)
println("    q: ", q_soln)

println("all residuals")


for i in 1:2
    println("    labor residual (",i,"): ", labor_residual(m0,w_soln,M_soln,i))
end

println("    ---")

for i in 1:2
    for s in 1:2
        println("    goods residual (",i,",",s,"): ", goods_residual(m0,w_soln,M_soln,i,s))
    end
end