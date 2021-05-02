# Chase Abram

# For solving open economy via JuMP
# Currently only Ipopt is used, not KNITRO


using JuMP #, KNITRO
using Parameters
using ForwardDiff
using Ipopt

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

    # Helpful constant
    alpha_bar = (1/alpha)^alpha * (1/(1-alpha))^(1-alpha)
    
    # Scaled US population
    # Lus = 0.4531
    # Scaled RoW population
    # Lrow = 9.5469
    L = [0.4531 9.5469]
    # L = [1.0 1.0]

    # # Estimated Values
    # # Production, final, RoW relative to US
    # Adrow = 0.2752
    # # Productivity, input, RoW relative to US
    # Aurow = 0.1121
    A = [1.0 1.0; 0.2752 0.1121]
    # A = ones(2,2)

    # # Icberg cost, final, from US to RoW
    # taud = 3.0301
    # # Iceberg cost, input, from US to RoW
    # tauu = 2.6039
    # tau = ones(2,2,2)
    # tau[1,2,1] = 3.0301
    # tau[2,1,1] = 3.0301
    
    # tau[1,2,2] = 2.6039
    # tau[2,1,2] = 2.6039
    tau = cat([1 3.0301; 3.0301 1], [1 2.6039; 2.6039 1], dims = 3)

    # Tariffs
    t = zeros(2,2,2)

    # Subsidies
    v = zeros(2,2,2)
end

##

# Marginal cost country-sector
function mcis(m, w, M, i, s)
    if s == 1
        return m.alpha_bar/m.A[i,s]*w[i]^m.alpha * Pjs(m,w,M,i,2)^(1-m.alpha)
    else
        return w[i]/m.A[i,s]
    end
end

# Price export-import-sector
function pijs(m, w, M, i, j, s)
    return m.mu[s]*m.tau[i,j,s]*mcis(m,w,M,i,s)/(1 + m.v[i,j,s])
end

# Price index export-import-sector
function Pijs(m, w, M, i, j, s)
    return M[i,s]^(1/(1-m.ces[s]))*(1 + m.t[i,j,s])*pijs(m,w,M,i,j,s)
end

# Price index import-sector
function Pjs(m,w,M,j,s)
    return sum([Pijs(m,w,M,i,j,s)^(1-m.ces[s]) for i in 1:2])^(1/(1-m.ces[s]))
end

# Upstream quantity export-import
function qjiu(m,w,M,j,i)
    return (1 - m.alpha)*mcis(m,w,M,i,1)*m.ces[1]*m.f[i,1]/Pjs(m,w,M,i,2) * ((1 + m.t[j,i,2])*pijs(m,w,M,j,i,2)/Pjs(m,w,M,i,2))^(-m.ces[2])
end

# Tax rebate country
function Ti(m,w,q,M,i)
    return sum([m.t[j,i,1]*M[j,1]*pijs(m,w,M,j,i,1)*q[j,i] + m.t[j,i,2]*M[j,2]*M[i,1]*pijs(m,w,M,j,i,2)*qjiu(m,w,M,j,i) - m.v[i,j,1]*M[i,1]*pijs(m,w,M,i,j,1)*q[i,j] - m.v[i,j,2]*M[i,2]*M[j,1]*pijs(m,w,M,i,j,2)*qjiu(m,w,M,i,j) for j in 1:2])
end

# Downstream demand residual export-import
function down_demand_residual_ji(m,w,q,M,j,i)
    return q[j,i] - (w[i]*m.L[i] + Ti(m,w,q,M,i))/Pjs(m,w,M,i,1) * ((1 + m.t[j,i,1])*pijs(m,w,M,j,i,1)/Pjs(m,w,M,i,1))^(-m.ces[1])
end

# Goods mkt residual country-sector
function goods_residual_is(m,w,q,M,i,s)
    if s == 1
        return (m.ces[s] - 1)*m.f[i,s] - sum([m.tau[i,j,s]*q[i,j] for j in 1:2])
        # w[i]*m.L[i] + Ti(m,w,q,M,i) - sum([pijs(m,w,M,j,i,s)*(1 + m.t[j,i,1])*q[j,i]*M[j,s] for j in 1:2])
    else
        return (m.ces[s] - 1)*m.f[i,s] - sum([M[j,1]*m.tau[i,j,s]*qjiu(m,w,M,i,j) for j in 1:2])
    end
end

# Labor market residual country
function labor_residual_i(m,w,M,i)
    return m.L[i] - M[i,1] * m.alpha * m.alpha_bar/m.A[i,1] * m.ces[1]*m.f[i,1]*(Pjs(m,w,M,i,2)/w[i])^(1-m.alpha) - M[i,2]*m.ces[2]*m.f[i,2]/m.A[i,2]
end

# All residuals for wage-quantity-mass system
function wqM_residuals(m,x::AbstractVector{T}) where T
# function wqM_residuals(m,w,M,q)
    
    # Adjust to avoid solver issues
    # y = x
    y = x.^2
    # y = exp.(x)

    # Rename for ease
    w = y[1:2]
    q = reshape(y[3:6], (2,2))
    M = reshape(y[7:10], (2,2))

    # w = [1.0 y[1]]
    # q = reshape(y[2:5], (2,2))
    # M = reshape(y[6:9], (2,2))

    # Initialize errors
    F = ones(T,length(y))
    # F = ones(10)
    F_iter = 0

    # Wage normalization (home)
    F_iter += 1
    F[F_iter] = w[1] - 1.0

    # Downstream demand residuals
    for j in 1:2
        for i in 1:2
            F_iter += 1
            F[F_iter] = down_demand_residual_ji(m,w,q,M,j,i)
        end
    end

    # Goods mkt residuals
    for i in 1:2
        for s in 1:2
            F_iter += 1
            F[F_iter] = goods_residual_is(m,w,q,M,i,s)
        end
    end

    # Labor market clearing (foreign)
    F_iter += 1
    F[F_iter] = labor_residual_i(m,w,M,2)

    # Sum of squared residuals
    return sum(F.^2)
end

# For testing
# m0 = OpenModel()
# wqM_residuals(m0, ones(10))

#################################################################
##
# With simple symmetry

# Initialize OpenModel
m0 = OpenModel(
    L = ones(2),
    A = ones(2,2),
    tau = ones(2,2,2)
)

# Residuals
res(z1, z2, z3, z4, z5, z6, z7, z8, z9, z10) = wqM_residuals(m0,[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10])

# For KNITRO
# m = Model(with_optimizer(KNITRO.Optimizer)) # (1)

# For Ipopt
m = Model(with_optimizer(Ipopt.Optimizer))

# Setup variables
init_value = ones(10)
@variable(m, x[i=1:10] >= 0, start = init_value[i])

# Register residuals with model
register(m, :res, 10, res, autodiff = true) # (4)

# Set objective
@NLobjective(m, Min, res(x[1], x[2], x[3], x[4], x[5], x[6], x[7],x[8],x[9], x[10]))

# Solve
optimize!(m)
# (value.(x), objective_value(m), termination_status(m)) # (5)

# Get solution
soln = value.(x).^2
w_soln = soln[1:2]
q_soln = reshape(soln[3:6], (2,2))
M_soln = reshape(soln[7:10], (2,2))

obj_val = objective_value(m)

# See results
println("----------Simple symmetry----------")
println("Found:")
println("    p soln: ", [pijs(m0,w_soln,M_soln,i,j,s) for i in 1:2, j in 1:2, s in 1:2])
println("    w_soln: ", w_soln)
println("    q down: ", q_soln)
println("    q up: ", [qjiu(m0,w_soln,M_soln,j,i) for j in 1:2, i in 1:2])
println("    M_soln: ", M_soln)
println("    obj_val: ", obj_val)
println()
println("Equilibrium conditions:")
println("    w_H (1)")
println("    Downstream demand (4)")
println("    Goods Mkt (4)")
println("    Labor Mkt F (1)")
println()
println("w_H: ", w_soln[1] - 1.0)

for j in 1:2
    for i in 1:2
        println("down demand (",j,",",i,"): ", down_demand_residual_ji(m0,w_soln,q_soln,M_soln,j,i))
    end
end

for i in 1:2
    for s in 1:2
        println("goods (",i,",",s,"): ", goods_residual_is(m0,w_soln, q_soln, M_soln, i, s))
    end
end

println("LMC_F: ", labor_residual_i(m0, w_soln, M_soln,2))
println()
println("Verification via LMC_H: ", labor_residual_i(m0,w_soln,M_soln,1))

#################################################################
##
# Symmetric Autarky

# Initialize OpenModel
m0 = OpenModel(
    L = ones(2) .* 0.4531,
    A = ones(2,2),
    tau = cat([1 5; 5 1], [1 5; 5 1], dims = 3)
)

# Residuals
res(z1, z2, z3, z4, z5, z6, z7, z8, z9, z10) = wqM_residuals(m0,[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10])

# For KNITRO
# m = Model(with_optimizer(KNITRO.Optimizer)) # (1)

# For Ipopt
m = Model(with_optimizer(Ipopt.Optimizer))

# Setup variables
init_value = ones(10)
@variable(m, x[i=1:10] >= 0, start = init_value[i])

# Register residuals with model
register(m, :res, 10, res, autodiff = true) # (4)

# Set objective
@NLobjective(m, Min, res(x[1], x[2], x[3], x[4], x[5], x[6], x[7],x[8],x[9], x[10]))

# Solve
optimize!(m)
# (value.(x), objective_value(m), termination_status(m)) # (5)

# Get solution
soln = value.(x).^2
w_soln = soln[1:2]
q_soln = reshape(soln[3:6], (2,2))
M_soln = reshape(soln[7:10], (2,2))

obj_val = objective_value(m)

# See results
println("----------Symmetric Autarky----------")
println("Found:")
println("    p soln: ", [pijs(m0,w_soln,M_soln,i,j,s) for i in 1:2, j in 1:2, s in 1:2])
println("    w_soln: ", w_soln)
println("    q down: ", q_soln)
println("    q up: ", [qjiu(m0,w_soln,M_soln,j,i) for j in 1:2, i in 1:2])
println("    M_soln: ", M_soln)
println("    obj_val: ", obj_val)
println()
println("Equilibrium conditions:")
println("    w_H (1)")
println("    Downstream demand (4)")
println("    Goods Mkt (4)")
println("    Labor Mkt F (1)")
println()
println("w_H: ", w_soln[1] - 1.0)

for j in 1:2
    for i in 1:2
        println("down demand (",j,",",i,"): ", down_demand_residual_ji(m0,w_soln,q_soln,M_soln,j,i))
    end
end

for i in 1:2
    for s in 1:2
        println("goods (",i,",",s,"): ", goods_residual_is(m0,w_soln, q_soln, M_soln, i, s))
    end
end

println("LMC_F: ", labor_residual_i(m0, w_soln, M_soln,2))
println()
println("Verification via LMC_H: ", labor_residual_i(m0,w_soln,M_soln,1))

#################################################################
##
# With calibrated parameters

# Initialize OpenModel
m0 = OpenModel()

# Residuals
res(z1, z2, z3, z4, z5, z6, z7, z8, z9, z10) = wqM_residuals(m0,[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10])

# For KNITRO
# m = Model(with_optimizer(KNITRO.Optimizer)) # (1)

# For Ipopt
m = Model(with_optimizer(Ipopt.Optimizer))

# Setup variables
init_value = ones(10)
@variable(m, x[i=1:10] >= 0, start = init_value[i])

# Register residuals with model
register(m, :res, 10, res, autodiff = true) # (4)

# Set objective
@NLobjective(m, Min, res(x[1], x[2], x[3], x[4], x[5], x[6], x[7],x[8],x[9], x[10]))

# Solve
optimize!(m)
# (value.(x), objective_value(m), termination_status(m)) # (5)

# Get solution
soln = value.(x).^2
w_soln = soln[1:2]
q_soln = reshape(soln[3:6], (2,2))
M_soln = reshape(soln[7:10], (2,2))

obj_val = objective_value(m)

# See results
println("----------Calibration----------")
println("Found:")
println("    p soln: ", [pijs(m0,w_soln,M_soln,i,j,s) for i in 1:2, j in 1:2, s in 1:2])
println("    w_soln: ", w_soln)
println("    q down: ", q_soln)
println("    q up: ", [qjiu(m0,w_soln,M_soln,j,i) for j in 1:2, i in 1:2])
println("    M_soln: ", M_soln)
println("    obj_val: ", obj_val)
println()
println("Equilibrium conditions:")
println("    w_H (1)")
println("    Downstream demand (4)")
println("    Goods Mkt (4)")
println("    Labor Mkt F (1)")
println()
println("w_H: ", w_soln[1] - 1.0)

for j in 1:2
    for i in 1:2
        println("down demand (",j,",",i,"): ", down_demand_residual_ji(m0,w_soln,q_soln,M_soln,j,i))
    end
end

for i in 1:2
    for s in 1:2
        println("goods (",i,",",s,"): ", goods_residual_is(m0,w_soln, q_soln, M_soln, i, s))
    end
end

println("LMC_F: ", labor_residual_i(m0, w_soln, M_soln,2))
println()
println("Verification via LMC_H: ", labor_residual_i(m0,w_soln,M_soln,1))

