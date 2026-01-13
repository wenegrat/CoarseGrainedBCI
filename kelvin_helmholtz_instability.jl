# Kelvin-Helmholtz instability simulation

using Oceananigans
using CairoMakie
using Printf

# Setup grid
Nx = Nz = 128
Lx = Ly = Lz = 10
grid = RectilinearGrid(size=(Nx, 1, Nz), x=(-Lx/2, Lx/2), y=(-Ly/2, Ly/2), z=(-Lz/2, Lz/2),
                       topology=(Periodic, Periodic, Bounded))

# Create model without background fields
ν = 2e-3
model = NonhydrostaticModel(grid;
                            advection = UpwindBiased(order=5),
                            closure = ScalarDiffusivity(ν=ν, κ=ν),
                            buoyancy = BuoyancyTracer(),
                            tracers = :b)

# Define initial conditions: shear flow with stratification and perturbation
Ri = 0.1
h = 1/4

# Base shear flow
shear_flow(x, z) = tanh(z)

# Base stratification
stratification(x, z) = h * Ri * tanh(z / h)

# Small perturbation to trigger instability
perturbation(x, z) = 0 * sin(2π * x / 10) * exp(-z^2 / 2)

# Set initial conditions
uᵢ(x, y, z) = shear_flow(x, z)
bᵢ(x, y, z) = stratification(x, z) + perturbation(x, z)
wᵢ(x, y, z) = 0.01 * cos(2π * x / 10) * exp(-z^2 / 2)

set!(model, u=uᵢ, b=bᵢ, w=wᵢ)

# Setup simulation
simulation = Simulation(model, Δt=0.01, stop_time=200.0)

# Add output writer
u, v, w = model.velocities
b = model.tracers.b

u_center = @at (Center, Center, Center) u
v_center = @at (Center, Center, Center) v
w_center = @at (Center, Center, Center) w

using Oceanostics: PotentialEnergyEquation
ρ₀ = 1025 # kg/ m^3
pe = ρ₀ * PotentialEnergyEquation.PotentialEnergy(model)

PE = Integral(pe)

vorticity = Field(∂z(u) - ∂x(w))

using NCDatasets
simulation.output_writers[:fields] =
    NetCDFWriter(model, (; ω=vorticity, b, pe, PE, u=u_center, v=v_center, w=w_center),
                 schedule = TimeInterval(2),
                 filename = "kelvin_helmholtz_instability",
                 overwrite_existing = true)

# Run simulation
@info "Running Kelvin-Helmholtz instability simulation..."
run!(simulation)

# Load and plot results
@info "Creating animation..."

filepath = simulation.output_writers[:fields].filepath

ω_timeseries = FieldTimeSeries(filepath, "ω")
b_timeseries = FieldTimeSeries(filepath, "b")

times = ω_timeseries.times

n = Observable(1)

ωₙ = @lift ω_timeseries[$n]
bₙ = @lift b_timeseries[$n]

fig = Figure(size=(900, 500))

title = @lift @sprintf("Kelvin-Helmholtz Instability: t = %.1f", times[$n])
fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

kwargs = (xlabel="x", ylabel="z", limits=((-5, 5), (-5, 5)), aspect=1)

ax_ω = Axis(fig[2, 1]; title="Vorticity", kwargs...)
ax_b = Axis(fig[2, 3]; title="Buoyancy", kwargs...)

hm_ω = heatmap!(ax_ω, ωₙ; colormap=:balance, colorrange=(-1, 1))
Colorbar(fig[2, 2], hm_ω)

hm_b = heatmap!(ax_b, bₙ; colormap=:balance, colorrange=(-0.05, 0.05))
Colorbar(fig[2, 4], hm_b)

frames = 1:length(times)

record(fig, "kelvin_helmholtz_instability.mp4", frames, framerate=8) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

@info "Animation saved as kelvin_helmholtz_instability.mp4"
