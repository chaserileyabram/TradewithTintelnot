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
include("open_solve.jl")
##

# optimal tariffs by country i 
# full = true if full policy set is available, false if just tariffs
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

        # Solve model
        solve(m)

        # See tariff and utilities while searching
        println("    ts(",i,"): ", ts)
        println("    U1: ", Ui(m, m.w, m.q, m.M, 1))
        println("    U2: ", Ui(m, m.w, m.q, m.M, 2))

        # Welfare (negative for minimization)
        return -Ui(m, m.w, m.q, m.M, i)
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

# optimal tit-for-tat by country i
function opt_tft(m, i; full = false, out_maxit = 50)

    # welfare implied by tariffs
    function fitness(ts::AbstractVector{T}) where T

        # i's instruments
        if i == 1
            # Set tariffs
            m.t[2,1,1] = ts[1]
            m.t[2,1,2] = ts[2]

            # Implied response
            m.t[1,2,1] = ts[1]
            m.t[1,2,2] = ts[2]

            if full
                # subsidy
                m.t[1,1,2] = ts[3]
                m.v[1,2,1] = ts[4]

                # Implied response
                m.t[2,2,2] = ts[3]
                m.v[2,1,1] = ts[4]
            end
        else
            m.t[1,2,1] = ts[1]
            m.t[1,2,2] = ts[2]

            # Implied response
            m.t[2,1,1] = ts[1]
            m.t[2,1,2] = ts[2]

            if full
                # subsidy
                m.t[2,2,2] = ts[3]
                m.v[2,1,1] = ts[4]

                # Implied response
                m.t[1,1,2] = ts[3]
                m.v[1,2,1] = ts[4]
            end
        end

        # Solve model
        solve(m)

        # See tariff and utilities while searching
        println("    ts(",i,"): ", ts)
        println("    U1: ", Ui(m, m.w, m.q, m.M, 1))
        println("    U2: ", Ui(m, m.w, m.q, m.M, 2))

        # Welfare (negative for minimization)
        return -Ui(m, m.w, m.q, m.M, i)
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
function tariff_war(m; full = false, maxit = 50, tol = 1e-6, br_maxit = 20)
    
    # Initialize
    it = 0
    diff = Inf
    old_t = zeros(size(m.t))
    old_v = zeros(size(m.v))

    # Continue until found or max iterations
    while it < maxit && diff > tol
        
        # Progress report
        println()
        println()
        println("tariff BR iteration: ", it, ", diff: ", diff)
        println("    t: ", m.t)
        println("    v: ", m.v)
        println()
        println()

        # Save old
        old_t .= m.t
        old_v .= m.v

        # Find optimal tariff
        # Alternate on who gets to respond
        opt_tariff(m,1 + (it % 2); full = full, out_maxit = br_maxit)

        # Find difference
        if full
            diff = max(maximum(abs.(m.t - old_t)), maximum(abs.(m.v - old_v)))
        else
            diff = maximum(abs.(m.t - old_t))
        end

        it += 1
    end
end

###################
# Test Kitchen

##
# Optimal Tariffs
m0 = OpenModel(ces = [7.0, 4.0])
opt_tariff(m0, 1; full = true, out_maxit = 100)

##
m1 = OpenModel()
m1.t = cat([0.0 0.39631484140672424; 0.41149732928496124 0.0],
[0.0 0.2160215159373675; 0.2230487883892319 0.0], dims = 3)
tariff_war(m1; maxit = 50, br_maxit = 20)

##
m2 = OpenModel()
opt_tft(m2,1)





