include("nbody_display.jl")
function mod2(v ::Vector{Float64})
    return sqrt(v[1]^2 + v[2]^2)
end

function compute_accelerations(positions ::Vector{Vector{Float64}}, accelerations ::Vector{Vector{Float64}}, masses ::Vector{Float64}, eps ::Float64)
    G ::Float64 = 6.67E-11
    n ::Int64 = length(accelerations)
    for i = 1:n
        sumacc ::Vector{Float64} = [0,0]
        for j = 1:n
            if i != j
                dv ::Vector{Float64} = positions[j] - positions[i]
                
                sumacc = sumacc + G * masses[j] * dv / (mod2(dv) + eps)^3
            end
        end
        accelerations[i] = sumacc
    end    
end

function update_pos_vel(positions ::Vector{Vector{Float64}}, velocities ::Vector{Vector{Float64}}, accelerations ::Vector{Vector{Float64}}, dt ::Float64)
    n ::Int64 = length(positions)
    for i = 1:n
        velocities[i] = velocities[i] + accelerations[i] * dt
        positions[i] = positions[i] + velocities[i] * dt
    end
end




function nbody_jl(positions ::Vector{Vector{Float64}}, velocities ::Vector{Vector{Float64}}, accelerations ::Vector{Vector{Float64}}, masses ::Vector{Float64}, nbr_simulations ::Int64, eps ::Float64, dt ::Float64)
    for i = 1:nbr_simulations
        compute_accelerations(positions, accelerations, masses, eps)
        update_pos_vel(positions, velocities, accelerations,dt)
        # pixels ::Array{Array{Int64}} = [[0,0,0] for i = 1:1000*1000]
        # graph_planets(pixels, positions, -4E8, 4E8, 1000, 1000, "nbody$i.ppm")
    end
end