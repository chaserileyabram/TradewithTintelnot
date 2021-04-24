# AFFGT "Import Tariffs and Global Sourcing"

# Sandbox by Chase Abram

# Notes

# Home = 1
# Foreign = 2

# Downstream = 1
# Upstream = 2

# x[i,j,s] means from country i to country j in sector s

# Nonlinear Solver
using NLsolve
using Parameters
# Not currently used
using ForwardDiff

##

# Stores exogenous parameters
# In calibrated, we would be updating this object
@with_kw mutable struct OpenAFFGTModel

    # sigma and theta
    ces = [4.0, 4.0]
    
    # Entry costs in goods
    f = [1.0 1.0; 1.0 1.0]

    # Expenditure on inputs relative to total sales
    alpha = 1 - 0.4517
    
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

    # Iceberg current not implemented
    # # Icberg cost, final, from US to RoW
    # taud = 3.0301
    # # Iceberg cost, input, from US to RoW
    # tauu = 2.6039

    # Tariffs
    # t[i,j,s] = tariff imposed by j on import from i in sector s
    t = zeros(2,2,2)
end

# Now idea is to solve nonlinear system

# p[i,j,s] -> from i to j in sector s
# t[i,j,s] -> tariff on import from i to j in sector s (imposed by j)
# q[i,j,s] -> same
# M[i,s] -> country i, sector s


# # Map index to single array (not used)
# function ijs_index(i,j,s)
#     return 4*(s-1) + 2(j-1) + i
# end

# # Map array back to indices (not used)
# function ijs_index_inv(k)
#     return [1 + ((k % 2) == 0), 1 + ((k % 4) in [0,3]), 1 + ((k % 8) in [0,5,6,7])]
# end

function no_neg(x)
    println("no_neg called")
    return x .* (x .>= 0) #+ 0.01 .* (x .<= 0)
end

# function penalty_neg(x)
#     return Inf*sum(x)
# end

# Aggregate P_j^s
function Pjs(p,M,m,j,s)
    # p = no_neg(p)
    # M = no_neg(M)
    return (sum(M[:,s] .* (p[:,j,s] .* (1 .+ m.t[:,j,s])).^(1-m.ces[s])))^(1/(1-m.ces[s]))
end

# Aggregate Q_j^u (and also Q_j^d == U_j)
function Qjs(q,M,m,j,s)
    # q = no_neg(q)
    # M = no_neg(M)
    return (sum(M[:,s] .* (q[:,j,s]).^((m.ces[s]-1)/m.ces[s])))^(m.ces[s]/(m.ces[s]-1))
end

# Tariff rebate
function Rj(p,q,m,M,j)
    # p = no_neg(p)
    # q = no_neg(q)
    # M = no_neg(M)
    return m.t[3 - j,j,1]*M[3 - j,1]*p[3-j,j,1]*q[3-j,j,1] + m.t[3-j,j,2]*M[3-j,2]*M[j,1]*p[3-j,j,2]*q[3-j,j,2]
end

# kappa objects
# These help with algebra and simplifying the code
function kappa(p,w,q,M,m,j,s)
    # p = no_neg(p)
    # w = no_neg(w)
    # q = no_neg(q)
    # M = no_neg(M)
    if s == 1
        return ((w[j]*m.L[j] + Rj(p,q,m,M,j))/Pjs(p,M,m,j,1))^(1/m.ces[1])*Pjs(p,M,m,j,1)
    elseif s == 2
        return (1/m.alpha - 1)*w[j]*(sum(q[j,:,1]) + m.f[j,1])^(1/m.alpha)*m.A[j,1]^(-1/m.alpha)*Qjs(q,M,m,j,2)^(1/m.ces[2] - 1/m.alpha)
    else
        println("invalid s for kappa")
    end
end

# Residuals from HH demand (4)
function hhd_residual(p,w,M,q,m,i,j)
    # p = no_neg(p)
    # w = no_neg(w)
    # M = no_neg(M)
    # q = no_neg(q)
    return kappa(p,w,q,M,m,j,1)*q[i,j,1]^(-1/m.ces[1]) - (1 + m.t[i,j,1])*p[i,j,1]
end

function demand_residual(p,w,M,q,m,i,j,s)
    if s == 1
        # return q[i,j,s] - ((1 + m.t[i,j,s])*p[i,j,s]/Pjs(p,M,m,j,s))^(-m.ces[s])*(w[j]*m.L[j] + Rj(p,q,m,M,j))/Pjs(p,M,m,j,s)
        return q[i,j,s] - ((1 + m.t[i,j,s])*p[i,j,s]/Pjs(p,M,m,j,s))^(-m.ces[s])*Qjs(q,M,m,j,s)
    elseif s == 2
        return q[i,j,s] - ((1 + m.t[i,j,s])*p[i,j,s]/Pjs(p,M,m,j,s))^(-m.ces[s])*Qjs(q,M,m,j,s)
    else
        println("Invalid sector in demand_residual")
    end
end

# Residuals from d output (4)
function dout_residual(p,w,M,q,m,i,j)
    # p = no_neg(p)
    # w = no_neg(w)
    # M = no_neg(M)
    # q = no_neg(q)
    return kappa(p,w,q,M,m,j,1)*(1 - 1/m.ces[1])*q[i,j,1]^(-1/m.ces[1]) - kappa(p,w,q,M,m,i,2)*Qjs(q,M,m,i,2)^(1-1/m.ces[2])/((1-m.alpha)*(sum(q[i,:,1]) + m.f[i,1]))
end

# Residuals from d input (4) (need to fix and include taxes and iceberg)
function din_residual(p,w,M,q,m,i,j)
    # p = no_neg(p)
    # w = no_neg(w)
    # M = no_neg(M)
    # q = no_neg(q)
    return kappa(p,w,q,M,m,j,2)*q[i,j,2]^(-1/m.ces[2]) - (1 + m.t[i,j,2])*p[i,j,2]
end

# Residuals from u output
function uout_residual(p,w,M,q,m,i,j)
    # p = no_neg(p)
    # w = no_neg(w)
    # M = no_neg(M)
    # q = no_neg(q)
    return kappa(p,w,q,M,m,j,2)*(1 - 1/m.ces[2])*q[i,j,2]^(-1/m.ces[2]) - w[i]/m.A[i,2]
end

# Residuals from zero profits d
function dzp_residual(q,m,j)
    # p = no_neg(p)
    # w = no_neg(w)
    # M = no_neg(M)
    # q = no_neg(q)
    # return sum(p[j,:,1] .* q[j,:,1]) - w[j]*(sum(q[j,:,1]) + m.f[j,1])^(1/m.alpha)*m.A[j,1]^(-1/m.alpha)*Qjs(q,M,m,j,2)^(1-1/m.alpha) - Pjs(p,M,m,j,2)*Qjs(q,M,m,j,2)
    return sum(q[j,:,1]) - (m.ces[1] - 1)*m.f[j,1]
end

# Residuals from zero profits u
function uzp_residual(M,q,m,j)
    # return sum(p[j,:,2] .* q[j,:,2]) - w[j]/m.A[j,2]*(sum(q[j,:,2]) + m.f[j,2])
    return sum(q[j,:,2] .* M[:,1]) - (m.ces[2] - 1)*m.f[j,2]
end

function labor_residual(p,w,M,q,m,j)
    # p = no_neg(p)
    # w = no_neg(w)
    # M = no_neg(M)
    # q = no_neg(q)
    return m.L[j] - M[j,1]*((sum(q[j,:,1]) + m.f[j,1])/(m.A[j,1]*Qjs(q,M,m,j,2)^(1-m.alpha)))^(1/m.alpha) - M[j,2]*(sum(M[:,1] .* q[j,:,2]) + m.f[j,2])/m.A[j,2]
end

##

function all_residuals!(F,x,m)
    
    # Rename variables
    p = reshape(exp.(x[1:8]), (2,2,2))
    w = exp.(x[9:10])
    M = reshape(exp.(x[11:14]), (2,2))
    q = reshape(exp.(x[15:22]), (2,2,2))

    # Residual counter
    F_iter = 0

    # Normalize wage
    for i in 1:1
        F_iter = 1
        F[F_iter] = w[i] - 1.0
    end

    # HH demand
    for i in 1:2
        for j in 1:2
            F_iter += 1
            # F[F_iter] = hhd_residual(p,w,M,q,m,i,j)
            F[F_iter] = demand_residual(p,w,M,q,m,i,j,1)
        end
    end

    # d output
    for i in 1:2
        for j in 1:2
            F_iter += 1
            F[F_iter] = dout_residual(p,w,M,q,m,i,j)
        end
    end

    # d input
    for i in 1:2
        for j in 1:2
            F_iter += 1
            # F[F_iter] = din_residual(p,w,M,q,m,i,j)
            F[F_iter] = demand_residual(p,w,M,q,m,i,j,2)
        end
    end

    # u output
    for i in 1:2
        for j in 1:2
            F_iter += 1
            F[F_iter] = uout_residual(p,w,M,q,m,i,j)
        end
    end

    # zero profits d
    for j in 1:2
        F_iter += 1
        F[F_iter] = dzp_residual(q,m,j)
    end

    # zero profits u
    for j in 1:2
        F_iter += 1
        F[F_iter] = uzp_residual(M,q,m,j)
    end

    # labor market
    for j in 2:2
        F_iter += 1
        F[F_iter] = labor_residual(p,w,M,q,m,j)
    end
end


# p0 = ones(2,2,2)
# # # p0[1,1,1] = 2
# # # p0[2,1,1] = 2
# q0 = ones(2,2,2)
# M0 = ones(2,2)
# # # t0 = zeros(2,2,2)
# # # t0[2,1,1] = 0.5
# # # t0[2,1,2] = 0.05
# w0 = [1, 1]
# # L0 = [1, 1]

# # # forgot the taxes here
# # for i in 1:2
# #     for j in 1:2
# #         println("i = ", i, ", j = ", j)
# #         println("    p0[i,j]: ", p0[i,j])
# #         println("    Pjs(i,j): ", Pjs(p0,M0,t0,i,j))
# #         println("    normalized: ", p0[i,j]/Pjs(p0,M0,t0,i,j))
# #     end
# # end

# # for i in 1:2
# #     for j in 1:2
# #         for s in 1:2
# #             # println("kappa(",j,",",s,"): ", kappa(p0,w0,q0,M0,m0,j,s))
# #             # println("hhd_residual(",i,",",j,",",s,"): ", hhd_residual(p0,w0,M0,q0,m0,i,j))
# #             # println("dout_resid(",i,",",j,",",s,"): ", dout_residual(p0,w0,M0,q0,m0,i,j))
# #             # println("din_resid(",i,",",j,",",s,"): ", din_residual(p0,w0,M0,q0,m0,i,j))
# #             # println("uout_resid(",i,",",j,",",s,"): ", uout_residual(p0,w0,M0,q0,m0,i,j))
# #             # println("dzp_residuals(",j,"): ", dzp_residuals(p0,w0, M0, q0, m0, j))
# #             # println("uzp_residuals(",j,"): ", uzp_residuals(p0,w0, M0, q0, m0, j))
# #             # println("labor_residuals(",j,"): ", labor_residuals(p0,w0, M0, q0, m0, j))
# #         end
# #     end
# # end

# # function f!(F,x)
# #     F[1] = x[1] - 5.0
# #     F[2] = x[2] - 6.0
# #     F[3] = x[3] - 1.0
# #     F[4] = x[4] - 3.0
# # end

# # nlsolve(f!, [-100.0 -100.0 100.0 100.0])

# # function g!(F,x)
# #     F[1,1] = x[1,1] - 5.0
# #     F[2,1] = x[2,1] - 6.0
# #     F[1,2] = x[1,2] - 1.0
# #     F[2,2] = x[2,2] - 3.0
# # end

# # nlsolve(g!, [-100.0 -100.0; 100.0 100.0])

# # Initialize residuals and guess
# m0 = OpenAFFGTModel()
# # p0 = ones(2,2,2)
# # q0 = ones(2,2,2)
# # M0 = ones(2,2)
# # w0 = [1, 1]

m0 = OpenAFFGTModel()
F0 = zeros(22)
# x0 = ones(22) #.+ 0.0im
# # x0[11:14] *= 0.1
x0 = zeros(22) .- 1.0

function solve_open(m, x_init)
    return nlsolve((F,x) -> all_residuals!(F,x,m), x_init, #autodiff = :forward,
    show_trace = true, method = :newton, iterations = 1000)
end

# Solve system
soln = solve_open(m0, x0)

println("soln")
println("    p: ", reshape(exp.(soln.zero[1:8]), (2,2,2)))
println("    w: ", exp.(soln.zero[9:10]))
println("    M: ", reshape(exp.(soln.zero[11:14]), (2,2)))
println("    q: ", reshape(exp.(soln.zero[15:22]), (2,2,2)))



##

# Try to just solve for quantities?
function q_residuals!(F,x,p,w,M,m)

    q = reshape(x[1:8], (2,2,2))
    q = exp.(q)

    # Residual counter
    F_iter = 0

    # Normalize q[1,1,1]
    # F_iter = 1
    # F[F_iter] = q[1,1,1] - 1.0

    # HH demand
    for i in 1:2
        for j in 1:2
            F_iter += 1
            F[F_iter] = demand_residual(p,w,M,q,m,i,j,1)
        end
    end

    # d input
    for i in 1:2
        for j in 1:2
            F_iter += 1
            F[F_iter] = demand_residual(p,w,M,q,m,i,j,2)
        end
    end
end

function solve_q(m, x_init, p, w, M)
    return nlsolve((F,x) -> q_residuals!(F,x,p,w,M,m), x_init,
    show_trace = true, method = :trust_region, iterations = 1000)
end

m0 = OpenAFFGTModel()
# x0 = zeros(2,2,2)
# x0 *= 2
p0 = ones(2,2,2)
# p0[1,1,1] = 1.0
# p0 *= 5.0
q0 = log.(ones(2,2,2)*10)
M0 = ones(2,2)
# M0[1,1,1] = 20.0
w0 = [1.0, 1.0]

q_soln = solve_q(m0,q0,p0,w0,M0)

q1 = reshape(q_soln.zero[1:8], (2,2,2))
q1 = exp.(q1)
println("q soln")
println("    p: ", p0)
# println("    w: ", w0)
println("    M: ", M0)
println("    q: ", q1)

# HH demand
for i in 1:2
    for j in 1:2
        println("hhd(",i,",",j,"): ", demand_residual(p0,w0,M0,q1,m0,i,j,1))
    end
end

# d input
for i in 1:2
    for j in 1:2
        println("din(",i,",",j,"): ", demand_residual(p0,w0,M0,q1,m0,i,j,2))
    end
end

##

# Need to solve small CES system
# Take prices/wages as given (10), solve for quantities and masses (12)
function qM_residuals!(F,x,p,m)
    
    # Rename variables
    q = reshape(x[1:8], (2,2,2))
    M = reshape(x[9:12], (2,2))
    
    # Guarantee positive
    q = exp.(q) ./(1 .+ exp.(q))*5
    M = exp.(M) ./(1 .+ exp.(M))*5

    # Residual counter
    F_iter = 0

    # HH demand
    for i in 1:2
        for j in 1:2
            F_iter += 1
            F[F_iter] = demand_residual(p,M,q,m,i,j,1)
        end
    end

    # d input demand
    for i in 1:2
        for j in 1:2
            F_iter += 1
            F[F_iter] = demand_residual(p,M,q,m,i,j,2)
        end
    end

    # zero profits d
    for j in 1:2
        F_iter += 1
        F[F_iter] = dzp_residual(q,m,j)
    end

    # zero profits u
    for j in 1:2
        F_iter += 1
        F[F_iter] = uzp_residual(M,q,m,j)
    end
end

function solve_qM(m, x_init, p)
    return nlsolve((F,x) -> qM_residuals!(F,x,p,m), x_init,
    show_trace = true, method = :newton, iterations = 100)
end

m0 = OpenAFFGTModel()
println(m0)
p0 = ones(2,2,2)
p0[1,1,1] = 1.1
x0 = zeros(12)
# x0 = ones(12)
# x0 *= 0.001
qM_soln = solve_qM(m0, x0, p0)

q1 = reshape(qM_soln.zero[1:8], (2,2,2))
M1 = reshape(qM_soln.zero[9:12], (2,2))

# q1 = exp.(q1)
# M1 = exp.(M1)

q1 = exp.(q1) ./(1 .+ exp.(q1))*5
M1 = exp.(M1) ./(1 .+ exp.(M1))*5
println("qM soln")
println("    p: ", p0)
println("    M: ", M1)
println("    q: ", q1)

println("qM residuals")
for i in 1:2
    for j in 1:2
        println("    dem 1 (",i,",",j,"): ", demand_residual(p0,M1,q1,m0,i,j,1))
        println("    dem 2 (",i,",",j,"): ", demand_residual(p0,M1,q1,m0,i,j,2))
    end
end

for j in 1:2
    println("    dzp (",j,"): ", dzp_residual(q1,m0,j))
    println("    uzp (",j,"): ", uzp_residual(M1,q1,m0,j))
end


##


function baby!(F,x,p)
    F_iter = 0

    sigma = 2

    P = sum(p.^(1-sigma))^(1/(1-sigma))
    X = sum(x.^((sigma-1)/sigma))^(sigma/(sigma-1))

    for j in 2:2
        F_iter += 1
        F[F_iter] = x[j] - (p[j]/P)^(-sigma)
    end
    F_iter += 1
    F[F_iter] = X - 1
end

p0 = ones(2)
p0[1] = 2
q0 = ones(2)
baby_soln = nlsolve((F,x) -> baby!(F,x,p0), q0,
show_trace = true, method = :trust_region, iterations = 1000,
ftol = 1e-10)

println(baby_soln)


##
p0 = ones(2,2,2)
p0[1,:,1] *= 2
M0 = ones(2,2)
m0 = OpenAFFGTModel()

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("p_{",i,",",j,"}^",s," / P_",j,"^",s,": ", p0[i,j,s]/Pjs(p0, M0, m0,j,s))
        end
    end
end