# "Import Tariffs and Global Sourcing" by Antras, Fadeev, Fort, Gutierrez, and Tintelnot

# Replication by Chase Abram and Ignacia Cuevas

# This code is not yet fully functional.

# 

# Load packages
using Parameters
using NLsolve
using Random

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


# P_ji^s (Price aggregator by export-import-sector)
function Pjis(m,p,M,j,i,s)
    return M[j,s]^(1/(1-m.ces[s]))*(1+m.t[j,i,s])*p[j,i,s]
end

#P_i (Price aggregator by import-sector)
function Pis(m,p,M,i,s)
    return sum([Pjis(m,p,M,j,i,s)^(1-m.ces[s]) for j in 1:2])^(1/(1-m.ces[s]))
end

#mc_i^S (Marginal cost by import-sector)
function mcis(m,p,w,M,i,s)
    if s == 1
        return m.alpha_bar/m.A[i,s] * w[i]^m.alpha * Pis(m,p,M,i,s)^(1-m.alpha)
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
    return m.L[i] - M[i,1]*labor(m,p,w,q,M,i,1) - M[i,2]*labor(m,p,w,q,M,i,2)
end

# Used to keep variables positive and/or well behaved
# This should not be needed
function transform(z)
    
    # x -> infinity => f(x) -> x
    # x -> -infinity => f(x) -> 0
    # return log(exp(z) + 1)
    
    # Smooth and simple
    return exp(z)
end

# Below this line code will still be quite messy
#######################################################
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
x0 = rand(22)

# F0 = zeros(21)
# x0 = zeros(21)

function solve_open(m, x_init)
    return nlsolve((F,x) -> eq_all!(F,x,m), x_init, #autodiff = :forward,
    show_trace = true, method = :newton, iterations = 100, xtol=1e-16)
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

p_soln = transform.(p_soln)
w_soln = transform.(w_soln)
M_soln = transform.(M_soln)
q_soln = transform.(q_soln)

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

    q = transform.(q)

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

q_soln = transform.(q_soln)

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
    p = transform.(p)

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
p_soln = reshape(soln.zero[1:8], (2,2,2))
p_soln = transform.(p_soln)

q_guess = reshape(ones(8), (2,2,2))
q_soln = solve_q(m0,p_soln,w0,M0,q_guess).zero
q_soln = transform.(q_soln)

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
    p = solve_p(m,w,M,p_guess).zero
    p = reshape(p, (2,2,2))
    p = transform.(p)
    # println("p: ", p)

    q_guess = reshape(ones(8), (2,2,2))
    q = solve_q(m,p,w,M,q_guess).zero
    q = reshape(q, (2,2,2))
    q = transform.(q)
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
    show_trace = true, method = :newton, linesearch = StrongWolfe(), iterations = 50, ftol=1e-16)
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
p_soln = transform.(p_soln)

q_guess = reshape(rand(8), (2,2,2))
q_soln = solve_q(m0,p_soln,w0,M_soln,q_guess).zero
q_soln = transform.(q_soln)

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
    M = transform.(M)

    p_guess = reshape(ones(8),(2,2,2))
    p = solve_p(m,w,M,p_guess).zero
    p = reshape(p, (2,2,2))
    p = transform.(p)
    # println("p: ", p)

    q_guess = reshape(ones(8), (2,2,2))
    q = solve_q(m,p,w,M,q_guess).zero
    q = reshape(q, (2,2,2))
    q = transform.(q)
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

# Solve system
soln = solve_M(m0,w_soln,x0)
M_soln = reshape(soln.zero, (2,2))
M_soln = transform.(M_soln)

p_guess = reshape(ones(8), (2,2,2))
p_soln = solve_p(m0,w_soln,M_soln,p_guess).zero
p_soln = transform.(p_soln)

q_guess = reshape(ones(8), (2,2,2))
q_soln = solve_q(m0,p_soln,w_soln,M_soln,q_guess).zero
q_soln = transform.(q_soln)

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

