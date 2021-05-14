# Chase Abram

# For solving open economy via JuMP
# Currently only Ipopt is used, not KNITRO


using JuMP #, KNITRO
using Parameters
using ForwardDiff
using Ipopt
using Optim

##
include("open_knitro.jl")
##

function opt_tariff(m)

    # welfare implied by tariffs
    function fitness(ts::AbstractVector{T}) where T

        # Set tariffs
        m.t[2,1,1] = ts[1]
        m.t[2,1,2] = ts[2]

        # Residuals
        res(z1, z2, z3, z4, z5, z6, z7, z8, z9, z10) = wqM_residuals(m,[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10])

        # For Ipopt
        mi = Model(with_optimizer(Ipopt.Optimizer))

        # Setup variables
        init_value = ones(T,10)
        @variable(mi, x[i=1:10] >= 0, start = init_value[i])

        # Register residuals with model
        register(mi, :res, 10, res, autodiff = true) # (4)

        # Set objective
        @NLobjective(mi, Min, res(x[1], x[2], x[3], x[4], x[5], x[6], x[7],x[8],x[9], x[10]))

        # Solve (silently)
        set_optimizer_attribute(mi, "print_level", 0)
        optimize!(mi)
        # (value.(x), objective_value(m), termination_status(m)) # (5)

        # Get solution
        soln = value.(x).^2
        w_soln = soln[1:2]
        q_soln = reshape(soln[3:6], (2,2))
        M_soln = reshape(soln[7:10], (2,2))

        # println("fitness: ", Ui(m, w_soln, q_soln, M_soln, 1))
        println("    ts: ", ts)
        # Home welfare (negative)
        return -1.0 * Ui(m, w_soln, q_soln, M_soln, 1)
    end

    # fit(t1, t2) = fitness([t1, t2])
    # # For Ipopt
    # mo = Model(with_optimizer(Ipopt.Optimizer))

    # # Setup variables
    # init_value = zeros(2)
    # @variable(mo, x[i=1:2] >= 0, start = init_value[i])

    # # Register residuals with model
    # register(mo, :fit, 2, fit, autodiff = true) # (4)
    # # register(mo, :fitness, 2, fitness, autodiff = true) # (4)

    # # println("x: ", [x[1], x[2]])
    # # println("fitness(x): ", fitness(x[1:2]))
    # # Set objective
    # @NLobjective(mo, Max, fit(x[1], x[2]))

    # # Solve (silently)
    # # set_optimizer_attribute(mo, "print_level", 0)
    # optimize!(mo)

    op = optimize(fitness, [0.0, 0.0], Optim.Options(show_trace = true))


    return Optim.minimizer(op) # value.(x)
end

m0 = OpenModel()
opt_tariff(m0)

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
set_optimizer_attribute(m, "print_level", 0)
optimize!(m)
# (value.(x), objective_value(m), termination_status(m)) # (5)

# Get solution
soln = value.(x).^2
w_soln = soln[1:2]
q_soln = reshape(soln[3:6], (2,2))
M_soln = reshape(soln[7:10], (2,2))

obj_val = objective_value(m)

# See results
# open("open_out.txt", "a") do io
#     println(io,"----------Paper parameters----------")
#     println(io, "model: ", m0)
#     println(io,"Found:")
#     println(io,"    p soln: ", [pijs(m0,w_soln,M_soln,i,j,s) for i in 1:2, j in 1:2, s in 1:2])
#     println(io,"    w_soln: ", w_soln)
#     println(io,"    q down: ", q_soln)
#     println(io,"    q up: ", [qjiu(m0,w_soln,M_soln,j,i) for j in 1:2, i in 1:2])
#     println(io,"    M_soln: ", M_soln)
#     println(io,"    obj_val: ", obj_val)
#     println(io,"")
#     println(io,"Equilibrium conditions:")
#     println(io,"    w_H (1)")
#     println(io,"    Downstream demand (4)")
#     println(io,"    Goods Mkt (4)")
#     println(io,"    Labor Mkt F (1)")
#     println(io,"")

#     println(io, "Eq. Residuals:")
#     println(io,"w_H: ", w_soln[1] - 1.0)

#     for j in 1:2
#         for i in 1:2
#             println(io,"down demand (",j,",",i,"): ", down_demand_residual_ji(m0,w_soln,q_soln,M_soln,j,i))
#         end
#     end

#     for i in 1:2
#         for s in 1:2
#             println(io,"goods (",i,",",s,"): ", goods_residual_is(m0,w_soln, q_soln, M_soln, i, s))
#         end
#     end

#     println(io,"LMC_F: ", labor_residual_i(m0, w_soln, M_soln,2))
#     println(io,"")
#     println(io,"Verification via LMC_H: ", labor_residual_i(m0,w_soln,M_soln,1))
# end



