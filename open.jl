# AFFGT Take 2

# Chase Abram

using Parameters
using NLsolve
using Random

##

# Stores exogenous parameters
# In calibrated, we would be updating this object
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

m0 = OpenModel()

function Pjis(m,p,M,j,i,s)
    return M[j,s]^(1/(1-m.ces[1]))*(1+m.t[j,i,s])*p[j,i,s]
end


function Pis(m,p,M,i,s)
    return sum([Pjis(m,p,M,j,i,s)^(1-m.ces[s]) for j in 1:2])^(1/(1-m.ces[2]))
end


function mcis(m,p,w,M,i,s)
    if s == 1
        return m.alpha_bar/m.A[i,s] * w[i]^m.alpha * Pis(m,p,M,i,s)^(1-m.alpha)
    else
        return w[i]/m.A[i,s]
    end
end

function yis(m,q,M,i,s)
    if s == 1
        return sum(m.tau[i,:,s] .* q[i,:,s])
    else
        return sum(M[:,1] .* m.tau[i,:,s] .* q[i,:,s])
    end
end


function Qjiu(m,p,w,q,M,j,i)
    return (1-m.alpha)*mcis(m,p,w,M,i,1)*(yis(m,q,M,i,1) + m.f[i,1])/Pis(m,p,M,i,2)*(Pjis(m,p,M,j,i,2)/Pis(m,p,M,i,2))^(-m.ces[2])
end

function Ti(m,p,q,M,i)
    return sum(m.t[:,i,1].*M[:,1].*p[:,i,1].*q[:,i,1] + m.t[:,i,2].*M[:,2].*M[i,1].*p[:,i,2].*q[:,i,2] - m.v[i,:,1].*M[i,1].*p[i,:,1].*q[i,:,1] - m.v[i,:,2].*M[i,2].*M[:,1].*p[i,:,2].*q[i,:,2])
end

function demand(m,p,w,q,M,j,i,s)
    if s == 1
        return q[j,i,s] - (w[i]*m.L[i] + Ti(m,p,q,M,i))/Pis(m,p,M,i,s) * ((1 + m.t[j,i,s])*p[j,i,s]/Pis(m,p,M,i,s))^(-m.ces[s])
    else
        return q[j,i,s] - Qjiu(m,p,w,q,M,j,i)*((1 + m.t[j,i,s])*p[j,i,s]/Pjis(m,p,M,j,i,s))^(-m.ces[s])
    end
end

function prices(m,p,w,M,i,j,s)
    return p[i,j,s] - m.mu[s]*m.tau[i,j,s]*mcis(m,p,w,M,i,s)/(1+m.v[i,j,s])
end

function free_entry(m,q,M,i,s)
    return yis(m,q,M,i,s) - (m.ces[s] - 1)*m.f[i,s]
end

function labor(m,p,w,q,M,i,s)
    if s == 1
        return m.alpha*mcis(m,p,w,M,i,s)*(yis(m,q,M,i,s) + m.f[i,s])/w[i]
    else
        return (yis(m,q,M,i,s) + m.f[i,s])/m.A[i,s]
    end
end

function lmc(m,p,w,q,M,i)
    return m.L[i] - M[i,1]*labor(m,p,w,q,M,i,1) - M[i,2]*labor(m,p,w,q,M,i,2)
end

function transform(z)
    # return log(exp(z) + 1)
    return exp(z)
end

##

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
    # F_iter += 1
    # F[F_iter] = w[1] - 1.0

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
            F[F_iter] = free_entry(m,q,M,i,s)
        end
    end

    for i in 1:2
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
    show_trace = true, method = :newton, iterations = 1000, xtol=1e-16)
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
        println("    free_entry (",i,",",s,"): ", free_entry(m0,q_soln,M_soln,i,s))
    end
end

println("    ---")

for i in 1:2
    println("    lmc (",i,"): ", lmc(m0,p_soln,w_soln,q_soln,M_soln,i))
end


##
# One piece at a time

# Quantities

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
    show_trace = false, method = :newton, iterations = 100, xtol=1e-16)
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
        println("    free_entry (",i,",",s,"): ", free_entry(m0,q_soln,M0,i,s))
    end
end

println("    ---")

for i in 1:2
    println("    lmc (",i,"): ", lmc(m0,p0,w0,q_soln,M0,i))
end

##

# Prices 

function eq_p!(F,x,m,w,M)
    # Rename variables
    p = reshape(x[1:8], (2,2,2))
    p = transform.(p)

    q_guess = reshape(ones(8), (2,2,2))
    q = solve_q(m,p,w,M,q_guess).zero

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
    show_trace = true, method = :newton, iterations = 50, xtol=1e-16)
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
q_p_soln = solve_q(m0,p_soln,w0,M0,q_guess).zero
q_p_soln = transform.(q_p_soln)



println("soln")
println("    p: ", p_soln)
println("    w: ", w0)
println("    M: ", M0)
println("    q: ", q_p_soln)

println("all residuals")

for i in 1:2
    for j in 1:2
        for s in 1:2
            println("    demand (",j,",",i,",",s,"): ", demand(m0,p_soln,w0,q_p_soln,M0,j,i,s))
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
        println("    free_entry (",i,",",s,"): ", free_entry(m0,q_p_soln,M0,i,s))
    end
end

println("    ---")

for i in 1:2
    println("    lmc (",i,"): ", lmc(m0,p_soln,w0,q_p_soln,M0,i))
end

