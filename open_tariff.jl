# Chase Abram

# Replication of "Import Tariffs and Global Sourcing"
# by Antras, Fadeev, Fort, Gutierrez, and Tintelnot

# This file addresses optimal tariff policy questions

##

# For solving open economy via JuMP
# Currently only Ipopt is used, not KNITRO


using JuMP #, KNITRO
using Parameters
using ForwardDiff
using Ipopt
using Optim
using .Threads

using Plots
##
include("open_knitro.jl")
##

function opt_tariff(m)

    # welfare implied by tariffs
    function fitness(ts::AbstractVector{T}) where T

        # Set tariffs
        m.t[2,1,1] = ts[1]
        m.t[2,1,2] = ts[2]

        # fit_U = zeros(T,1)
        # fit_U[1] = -sum((ts .- 1).^2)

        # return fit_U[1]

        # Residuals
        res(z1, z2, z3, z4, z5, z6, z7, z8, z9, z10) = wqM_residuals(m,[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10])

        # For Ipopt
        mi = Model(with_optimizer(Ipopt.Optimizer, tol = 1e-20))

        # Setup variables
        iinit_value = ones(T,10)
        # @variable(mi, x[i=1:10] >= 0, start = iinit_value[i])
        @variable(mi, x[i=1:10] >= 0, start = iinit_value[i])

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

        # See tariff while searching
        println("    ts: ", ts)
        println("    U1: ", Ui(m, w_soln, q_soln, M_soln, 1))
        println("    U2: ", Ui(m, w_soln, q_soln, M_soln, 2))

        # Find (negative) utility
        # home_U = zeros(Real,1)
        # home_U[1] = Ui(m, w_soln, q_soln, M_soln, 1)

        # Home welfare (negative for minimization)
        return -Ui(m, w_soln, q_soln, M_soln, 1)
    end

    # # Residuals
    # fit(z1, z2) = fitness([z1, z2])

    # # fit(z1, z2) = (1 - z1)^2 + (1 - z2)^2

    # # For Ipopt
    # mo = Model(with_optimizer(Ipopt.Optimizer))

    # # Setup variables
    # oinit_value = zeros(Real,2)
    # @variable(mo, x[i=1:2], start = oinit_value[i])

    # # Register residuals with model
    # register(mo, :fit, 2, fit, autodiff = true) # (4)

    # # Set objective
    # @NLobjective(mo, Max, fit(x[1], x[2]))

    # # Solve (silently)
    # # set_optimizer_attribute(mo, "print_level", 0)
    # optimize!(mo)

    # return value.(x)

    # n = 10

    # ws = zeros(n,n)
    # ts = LinRange(0,1,n)

    # Threads.@threads for i in 1:n
    #     for j in 1:n
    #         ws[i,j] = fitness([ts[i], ts[j]])
    #     end
    # end

    # p = plot(ts,ts,ws)
    # display(p)

    # Find max (min of negative)
    op = optimize(fitness, [0.0, 0.0], Optim.Options(show_trace = true), autodiff = :forward)

    # Implied argmin
    return Optim.minimizer(op) # value.(x)
end

# optimal tariffs by country i. full set of policies available?
function opt_tariff(m, i; full = false, out_maxit = 50)

    # welfare implied by tariffs
    function fitness(ts::AbstractVector{T}) where T

        # i's instruments
        if i == 1
            # Set tariffs
            m.t[2,1,1] = ts[1]
            m.t[2,1,2] = ts[2]

            if full
                # subsidy
                m.t[1,1,2] = ts[3]
                m.v[1,2,1] = ts[4]
            end
        else
            m.t[1,2,1] = ts[1]
            m.t[1,2,2] = ts[2]

            if full
                # subsidy
                m.t[2,2,2] = ts[3]
                m.v[2,1,1] = ts[4]
            end
        end

        # Residuals
        res(z1, z2, z3, z4, z5, z6, z7, z8, z9, z10) = wqM_residuals(m,[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10])

        # For Ipopt
        mi = Model(with_optimizer(Ipopt.Optimizer, tol = 1e-20))

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

        # Get solution
        soln = value.(x).^2
        w_soln = soln[1:2]
        q_soln = reshape(soln[3:6], (2,2))
        M_soln = reshape(soln[7:10], (2,2))

        # See tariff and utilities while searching
        println("    ts(",i,"): ", ts)
        println("    U1: ", Ui(m, w_soln, q_soln, M_soln, 1))
        println("    U2: ", Ui(m, w_soln, q_soln, M_soln, 2))

        # Welfare (negative for minimization)
        return -Ui(m, w_soln, q_soln, M_soln, i)
    end

    # Find max (min of negative)
    if i == 1
        if full
            op = optimize(fitness, [m.t[2,1,1], m.t[2,1,2], m.t[1,1,2], m.t[1,2,1]], Optim.Options(show_trace = true, iterations = out_maxit), autodiff = :forward)
        else
            op = optimize(fitness, [m.t[2,1,1], m.t[2,1,2]], Optim.Options(show_trace = true, iterations = out_maxit), autodiff = :forward)
        end
    else
        if full
            op = optimize(fitness, [m.t[1,2,1], m.t[1,2,2], m.t[2,2,2], m.t[1,2,1]], Optim.Options(show_trace = true, iterations = out_maxit), autodiff = :forward)
        else
            op = optimize(fitness, [m.t[1,2,1], m.t[1,2,2]], Optim.Options(show_trace = true, iterations = out_maxit), autodiff = :forward)
        end
    end

    # Implied argmin
    return Optim.minimizer(op)
end

# Solve Nash equilibrium of tariff war
function tariff_war(m; full = false, maxit = 20, tol = 1e-6, br_maxit = 50)
    it = 0
    diff = Inf
    old_t = zeros(size(m.t))
    old_v = zeros(size(m.v))

    while it < maxit && diff > tol
        println()
        println()
        println("tariff BR iteration: ", it, ", diff: ", diff)
        println("    t: ", m.t)
        println("    v: ", m.v)
        println()
        println()

        old_t .= m.t
        old_v .= m.v
        opt_tariff(m,1 + (it % 2); full = full, out_maxit = br_maxit)

        if full
            diff = max(maximum(abs.(m.t - old_t)), maximum(abs.(m.v - old_v)))
        else
            diff = maximum(abs.(m.t - old_t))
        end

        it += 1
    end
end


##
# Optimal Tariffs
m0 = OpenModel()
opt_tariff(m0, 2; full=true)

##
m1 = OpenModel()
tariff_war(m1; br_maxit = 20)




