# Kelvin-Helmholtz instability simulation

using Oceananigans
using CairoMakie
using Printf
using CUDA: has_cuda_gpu
using Oceananigans.Architectures: on_architecture

#+++ Create grid
Lx = Lz = 10
Ly = 5
if has_cuda_gpu()
    arch = GPU()
    Nx = Nz = 256
    Ny = Nx÷2
    @info "CUDA GPU detected! Running 3D simulation with $(Nx)×$(Ny)×$(Nz) grid on GPU"
else
    arch = CPU()
    Nx = Nz = 128
    Ny = 1
    @info "No CUDA GPU detected. Running 2D simulation with $(Nx)×$(Ny)×$(Nz) grid on CPU"
end

grid = RectilinearGrid(arch; size=(Nx, Ny, Nz),
                       x=(-Lx/2, Lx/2), y=(-Ly/2, Ly/2), z=(-Lz/2, Lz/2),
                       topology=(Periodic, Periodic, Bounded))
#---

#+++ Create model
ν = 2e-3
model = NonhydrostaticModel(grid;
                            advection = UpwindBiased(order=5),
                            closure = ScalarDiffusivity(ν=ν, κ=ν),
                            buoyancy = BuoyancyTracer(),
                            tracers = :b)

u, v, w = model.velocities
b = model.tracers.b
#---

#+++ Define initial conditions: shear flow with stratification and perturbation
Ri = 0.1
h = 1/4
perturbation_amplitude = 0.01

shear_flow(x, z) = tanh(z) # Base shear flow
stratification(x, z) = h * Ri * tanh(z / h) # Base stratification
perturbation(x, z) = perturbation_amplitude * sin(2π * x / 10) * exp(-z^2 / 2) # Small perturbation to trigger instability

# Set initial conditions
uᵢ(x, y, z) = shear_flow(x, z)
bᵢ(x, y, z) = stratification(x, z)
wᵢ(x, y, z) = perturbation_amplitude * cos(2π * x / 10) * exp(-z^2 / 2)

set!(model, u=uᵢ, b=bᵢ, w=wᵢ)
#---

#+++ Setup simulation
simulation = Simulation(model, Δt=0.01, stop_time=200.0)

#+++ Add progress messenger
using Oceanostics.ProgressMessengers
walltime_per_timestep = StepDuration(with_prefix=false)
walltime = Walltime()

progress(simulation) = @info (PercentageProgress(with_prefix=false, with_units=false)
                              + walltime
                              + TimeStep()
                              + "CFL = " * AdvectiveCFLNumber(with_prefix=false)
                              + "Diffusive CFL = " * DiffusiveCFLNumber(with_prefix=false)
                              + MaxUVelocity()
                              + "step dur = " * walltime_per_timestep
                              )(simulation)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))
#---

#+++ Add TimeStepWizard for adaptive timestepping
N²_max = ∂z(b) |> Field |> maximum
max_Δt = 0.2 / √N²_max # Max timestep is 0.2 times the buoyancy period
conjure_time_step_wizard!(simulation, IterationInterval(1);
                          max_change=1.05,
                          cfl=0.8,
                          min_Δt=1e-4,
                          max_Δt)
#---
#---

#+++ Add output writer
u_center = @at (Center, Center, Center) u
v_center = @at (Center, Center, Center) v
w_center = @at (Center, Center, Center) w

using Oceanostics: PotentialEnergyEquation
ρ₀ = 1025 # kg/ m^3
pe = ρ₀ * PotentialEnergyEquation.PotentialEnergy(model)

PE = Integral(pe)

vorticity = Field(∂z(u) - ∂x(w))

using NCDatasets
output_filename = "output/kelvin_helmholtz_instability_$(Nx)x$(Ny)x$(Nz)"
simulation.output_writers[:fields] =
    NetCDFWriter(model, (; ω=vorticity, b, pe, PE, u=u_center, v=v_center, w=w_center),
                 schedule = TimeInterval(4),
                 filename = output_filename,
                 array_type = Array{Float64},
                 overwrite_existing = true)

output_filename_2d = "output/kelvin_helmholtz_instability_$(Nx)x$(Ny)x$(Nz)_2d.nc"
simulation.output_writers[:twod_fields] =
NetCDFWriter(model, (; ω=vorticity, b),
            schedule = TimeInterval(2),
            filename = output_filename_2d,
            array_type = Array{Float32},
            indices = (:, 1, :),
            overwrite_existing = true)


@info "Output will be saved to: $(output_filename).nc"
#---

# Run simulation
@info "Running Kelvin-Helmholtz instability simulation..."
run!(simulation)

#+++ Load and plot results
@info "Creating animation..."

filepath = simulation.output_writers[:twod_fields].filepath

ω_timeseries = FieldTimeSeries(filepath, "ω", architecture=CPU())
b_timeseries = FieldTimeSeries(filepath, "b", architecture=CPU())

# The architecture of the FieldTimeSeries isn't working as expected, so we need to load the data on the CPU manually:
ω_timeseries = on_architecture(CPU(), ω_timeseries)
b_timeseries = on_architecture(CPU(), b_timeseries)

times = ω_timeseries.times

n = Observable(1)

ωₙ = @lift view(ω_timeseries[$n], :, 1, :)
bₙ = @lift view(b_timeseries[$n], :, 1, :)

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

animation_filename = "animations/$(basename(output_filename)).mp4"
record(fig, animation_filename, frames, framerate=12) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

@info "Animation saved as $(animation_filename)"
#---