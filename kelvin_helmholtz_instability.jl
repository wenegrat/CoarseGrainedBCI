# Kelvin-Helmholtz instability simulation
using Oceananigans
using CairoMakie
using Printf
using ArgParse
using CUDA: has_cuda_gpu
using Oceananigans.Architectures: on_architecture
using Oceanostics: PotentialEnergyEquation, KineticEnergyEquation, FlowDiagnostics, GaussianFilter, StrainRateTensor
using Oceanostics.ProgressMessengers
@info "Finished loading packages"

include("utils.jl")

#+++ Parse command-line arguments
let s = ArgParseSettings()
    @add_arg_table! s begin
        "--Nz"
            help = "Number of vertical grid points (default: 512 on CPU, 4096 on GPU)"
            arg_type = Int
            required = false
            default = has_cuda_gpu() ? 4096 : 256

        "--U"
            help = "Velocity profile amplitude U₀ (default: 1.0)"
            arg_type = Float64
            required = false
            default = 1

        "--stop_time"
            help = "Simulation stop time (default: 200.0)"
            arg_type = Float64
            required = false
            default = 200.0

        "--Re0"
            help = "Base Reynolds number (default: 5e-4)"
            arg_type = Float64
            required = false
            default = 5e-4

        "--Ri"
            help = "Base Richardson number (default: 0.1)"
            arg_type = Float64
            required = false
            default = 0.1

        "--Pr"
            help = "Prandtl number (default: 1.0)"
            arg_type = Float64
            required = false
            default = 1

        "--h"
            help = "Buoyancy layer half-width relative to velocity half-width (default: 1.0, i.e. same scale for both)"
            arg_type = Float64
            required = false
            default = 1

        "--perturbation_amplitude"
            help = "Perturbation amplitude (default: 0.05)"
            arg_type = Float64
            required = false
            default = 0.05
    end
    global parsed_args = parse_args(s, as_symbols=true)
end
params = (; parsed_args...)
#---

#+++ Define simulation parameters
# Theoretical most unstable wavenumber for the KH instability taken from
# Kaminski and Smyth (2019): https://doi.org/10.1016/j.ocemod.2019.04.005
# which in turn refers to Miles (1961).
# We refer to Michalke (1964)'s resuts: k_max · δ_u = 0.4446 which seems to also match.
let
    k_max = 0.4446 / params.h
    λ_max = 2π / k_max

    Lx = λ_max
    Ly = λ_max / 3
    Lz = 25 * params.h
    Re₀ = params.Re0
    B₀ = params.U^2 * params.Ri / params.h
    global params = (; params..., k_max, λ_max, Lx, Ly, Lz, B₀, Re₀)
end
@info @sprintf("Most unstable KH wavenumber: k_max = %.4f  (λ_max = %.2f, Lx = %.1f)",
               params.k_max, params.λ_max, params.Lx)
#---

#+++ Create grid
if has_cuda_gpu()
    arch = GPU()
    x_aspect_ratio = 1   # Δx / Δz ratio
    y_aspect_ratio = Inf # Δy / Δz ratio
else
    @warn "No CUDA GPU detected. Running on CPU with a coarse grid and high aspect ratio."

    arch = CPU()
    x_aspect_ratio = 2   # Δx / Δz ratio
    y_aspect_ratio = Inf # Δy / Δz ratio
end

@info "Cell aspect ratio: Δx/Δz = $(x_aspect_ratio), Δy/Δz = $(y_aspect_ratio)"

# Calculate horizontal resolutions based on aspect ratios
Nx = round(Int, params.Nz * (params.Lx / params.Lz) / x_aspect_ratio)
Ny = isinf(y_aspect_ratio) ? 1 : round(Int, params.Nz * (params.Ly / params.Lz) / y_aspect_ratio)

# Adjust grid sizes to be factorizable by 2, 3, and 5 (for FFT performance)
Nx = closest_factor_number((2, 3, 5), Nx)
Ny = closest_factor_number((2, 3, 5), Ny)

params = (; params..., Nx, Ny)

grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz),
                       x=(-params.Lx/2, params.Lx/2),
                       y=(-params.Ly/2, params.Ly/2),
                       z=(-params.Lz/2, params.Lz/2),
                       topology=(Periodic, Periodic, Bounded))
#---

#+++ Define Reynolds number, viscosity and diffusivity
let
    if grid.Ny == 1
        Re = params.Re₀ * params.Nz^2
    else
        Re = params.Re₀ * params.Nz^(4/3) # Double check this
    end
    ν = params.U * params.h / Re
    κ = ν / params.Pr
    global params = merge(params, (; ν, κ, Re))
end
#---

#+++ Create model
model = NonhydrostaticModel(grid;
                            advection = Centered(order=4),
                            closure = ScalarDiffusivity(ν=params.ν, κ=params.κ),
                            buoyancy = BuoyancyTracer(),
                            tracers = :b)
u, v, w = model.velocities
b = model.tracers.b
#---

#+++ Define initial conditions: shear flow with stratification and perturbation
shear_flow(x, z) = params.U * tanh(z / params.h) # Base shear flow
stratification(x, z) = params.B₀ * tanh(z / params.h) # Base stratification
perturbation(x, z) = params.perturbation_amplitude * abs(randn()) * exp(-z^2) * sin(x * params.k_max - π) # Small perturbation to trigger instability

# Set initial conditions
uᵢ(x, y, z) = shear_flow(x, z)
bᵢ(x, y, z) = stratification(x, z)
wᵢ(x, y, z) = perturbation(x, z)
set!(model, u=uᵢ, b=bᵢ, w=wᵢ)
#---

#+++ Setup simulation
#+++ Set initial Δt to 10% of the CFL condition using params.U
Δx = minimum_xspacing(grid)
initial_Δt = 0.1 * Δx / params.U
simulation = Simulation(model, Δt=initial_Δt, stop_time=params.stop_time)
#---

#+++ Add progress messenger
walltime_per_timestep = StepDuration(with_prefix=false)
walltime = Walltime()

Δx = minimum_xspacing(grid)

ε = KineticEnergyEquation.DissipationRate(model)
ε̄ = Average(ε, dims=(1, 2)) |> Field
η = (params.ν^3 / ε̄) ^ (1/4)


progress(simulation) = @info (PercentageProgress(with_prefix=false, with_units=false)
                              + walltime
                              + TimeStep()
                              + "CFL = " * AdvectiveCFLNumber(with_prefix=false)
                              + "Diffusive CFL = " * DiffusiveCFLNumber(with_prefix=false)
                              + MaxWVelocity()
                              + "step dur = " * walltime_per_timestep
                              + (sim -> @sprintf("Kolmogorov length/Δx = %.2f", minimum(η) / Δx))
                              )(simulation)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))
#---

#+++ Add TimeStepWizard for adaptive timestepping
N²_max = ∂z(b) |> Field |> maximum
max_Δt = 0.2 / √N²_max # Max timestep is 0.2 times the buoyancy period
conjure_time_step_wizard!(simulation, IterationInterval(1);
                          max_change=1.05,
                          cfl=0.8,
                          diffusive_cfl=0.3,
                          min_Δt=1e-4,
                          max_Δt)
#---
#---

#+++ Add output writer
u_center = @at (Center, Center, Center) u
v_center = @at (Center, Center, Center) v
w_center = @at (Center, Center, Center) w

Ri_field = FlowDiagnostics.RichardsonNumber(model)
S_field  = FlowDiagnostics.StrainRateTensorModulus(model)

ρ₀ = 1025 # kg/ m^3
pe = ρ₀ * PotentialEnergyEquation.PotentialEnergy(model)

PE = Integral(pe)

vorticity = Field(∂z(u) - ∂x(w))

#+++ Gaussian-filtered u, v, w, b at multiple filter scales for subfilter-scale analysis
# ℓ is the FWHM of the Gaussian kernel; σ = ℓ / (2√(2 ln 2)) is the std dev passed to GaussianFilter
filter_ℓs = (1, 7)
_FWHM_to_σ(ℓ) = ℓ / (2 * sqrt(2 * log(2)))
_fields = (u=u_center, v=v_center, w=w_center, b=b)
_filt_pairs = [Symbol("$(n)_ℓ$(ℓ)") => GaussianFilter(f; dims=(1, 3), σ=_FWHM_to_σ(ℓ)) for ℓ in filter_ℓs for (n, f) in pairs(_fields)]
filtered_fields = (; _filt_pairs...)
#---

#+++ Online cross-scale KE transfer Πₖ = -τⁱʲ S̄ⁱʲ  (mirrors offline postprocessing/03_energy_transfer.py)
# Cross-scale (subfilter → resolved) kinetic-energy flux of Aluie et al. (2018, JPO), Eq. (7):
#
#     Πₖ = -τⁱʲ S̄ⁱʲ ,   τⁱʲ = filter(uⁱuʲ) - ūⁱ ūʲ ,   S̄ⁱʲ = ½(∂ūⁱ/∂xʲ + ∂ūʲ/∂xⁱ)
#
# τⁱʲ is the sub-filter stress tensor (built here from filtered velocity products) and S̄ⁱʲ the
# strain rate of the filtered velocity (from Oceanostics' StrainRateTensor). The Gaussian filter
# (FWHM = ℓ, in x and z) is configured to reproduce the offline post-processing filter: periodic in
# x, edge-extended in the bounded z (offline `mode="nearest"`), and truncated at 4σ (see below). The
# runs are 2D in x–z (v ≡ 0), so only the i,j ∈ {1,3} components survive — matching the offline
# calculation, which omits ρ₀ (Πₖ is per unit mass, units m² s⁻³). Πₖ > 0 is forward (downscale)
# transfer.
function cross_scale_ke_flux(model, ℓ; boundary=:edge, truncate=4)
    grid = model.grid
    u, v, w = model.velocities
    σ = _FWHM_to_σ(ℓ)
    # Match scipy.ndimage.gaussian_filter1d's stencil radius (= ⌊truncate·σ/Δ + ½⌋ cells) in each
    # direction. The offline filter (scipy, default truncate=4) keeps the Gaussian out to 4σ, whereas
    # Oceanostics' GaussianFilter truncates at only 2σ by default; with the radius matched the two
    # kernels are identical (same exp(-Δi²/2σ²) weights, same normalization), so the online and
    # offline Πₖ agree up to the strain-operator discretization, which vanishes with resolution.
    radius(Δ) = max(1, floor(Int, truncate * σ / Δ + 0.5))
    N = (2radius(minimum_xspacing(grid)) + 1, 2radius(minimum_zspacing(grid)) + 1)
    filt(ψ) = GaussianFilter(ψ; dims=(1, 3), σ, boundary, N)

    # Filtered velocities ūⁱ at their native staggered locations → valid input for StrainRateTensor
    ū = Field(filt(u)); v̄ = Field(filt(v)); w̄ = Field(filt(w))
    S̄ = StrainRateTensor(grid, ū, v̄, w̄)  # S̄ⁱʲ of the filtered velocity (only S₁₁, S₃₃, S₁₃ are used)

    # Sub-filter stress τⁱʲ = filter(uⁱuʲ) - ūⁱūʲ, built at cell centers (cf. the spatial_filtering
    # example in Oceanostics). Products of the full velocity are filtered; the ūⁱūʲ term reuses the
    # filtered velocities above, interpolated to centers.
    uᶜ = @at (Center, Center, Center) u
    wᶜ = @at (Center, Center, Center) w
    filter_uu = Field(filt(Field(uᶜ * uᶜ)))
    filter_uw = Field(filt(Field(uᶜ * wᶜ)))
    filter_ww = Field(filt(Field(wᶜ * wᶜ)))

    ūᶜ = @at (Center, Center, Center) ū
    w̄ᶜ = @at (Center, Center, Center) w̄
    τ₁₁ = filter_uu - ūᶜ * ūᶜ
    τ₃₃ = filter_ww - w̄ᶜ * w̄ᶜ
    τ₁₃ = filter_uw - ūᶜ * w̄ᶜ

    # Double contraction τⁱʲ S̄ⁱʲ at cell centers (off-diagonal counted twice). S̄₁₃ lives at
    # (Face, Center, Face); interpolate it to centers to co-locate with τ₁₃.
    S̄₁₃ᶜ = @at (Center, Center, Center) S̄.S₁₃
    return -(τ₁₁ * S̄.S₁₁ + τ₃₃ * S̄.S₃₃ + 2 * τ₁₃ * S̄₁₃ᶜ)
end

_Πₖ_ops   = [ℓ => cross_scale_ke_flux(model, ℓ) for ℓ in filter_ℓs]
_Πₖ_pairs = vcat([Symbol("Π_K_ℓ$(ℓ)")     => Πₖ           for (ℓ, Πₖ) in _Πₖ_ops],
                 [Symbol("Π_K_ℓ$(ℓ)_int") => Integral(Πₖ) for (ℓ, Πₖ) in _Πₖ_ops])
ke_transfer_fields = (; _Πₖ_pairs...)
#---

outputs = (; ω=vorticity, b, pe, PE, u=u_center, v=v_center, w=w_center, filtered_fields..., ke_transfer_fields..., ε̄, ε, Ri=Ri_field, S=S_field)

using NCDatasets
simulation_name = "khi_Nz$(params.Nz)_Ri$(@sprintf("%.2f", params.Ri))"
output_filename = "output/$(simulation_name).nc"

if !(model.closure isa ScalarDiffusivity)
    ν = viscosity(model)
    κ = diffusivity(model, Val(:b))
    outputs = (; outputs..., ν, κ)
end

simulation.output_writers[:fields] =
    NetCDFWriter(model, outputs,
                 schedule = ConsecutiveIterations(TimeInterval(2)), # Consecutive iterations every 4 periods to calculate time derivatives
                 filename = output_filename,
                 array_type = Array{Float64},
                 global_attributes = params,
                 overwrite_existing = true)

output_filename_2d = "output/$(simulation_name)_2d.nc"
simulation.output_writers[:twod_fields] =
NetCDFWriter(model, outputs,
             schedule = TimeInterval(2),
             filename = output_filename_2d,
             array_type = Array{Float32},
             indices = (:, 1, :),
             global_attributes = params,
             overwrite_existing = true)


@info "Output will be saved to: $(output_filename).nc"
#---

#+++ Run simulation
show_gpu_status()
@info @sprintf("""
================================================================================
  Kelvin-Helmholtz instability simulation
================================================================================
  Grid:          Nx=%d, Ny=%d, Nz=%d
  Domain:        Lx=%.1f, Ly=%.1f, Lz=%.1f
  Stop time:     %.1f
  Richardson:    Ri = %.4f
  Reynolds:      Re = %.1f  (Re₀ = %.2e)
  Prandtl:       Pr = %.1f
  Viscosity:     ν  = %.2e
  Diffusivity:   κ  = %.2e
  KH wavenumber: k_max = %.4f  (λ_max = %.2f)
================================================================================
""",
    params.Nx, params.Ny, params.Nz,
    params.Lx, params.Ly, params.Lz,
    params.stop_time,
    params.Ri,
    params.Re, params.Re₀,
    params.Pr,
    params.ν,
    params.κ,
    params.k_max, params.λ_max)
@info "Running Kelvin-Helmholtz instability simulation..."
run!(simulation)
#---

#+++ Plot results
@info "Creating animation..."
plot_filepath = output_filename_2d
include("plot_kelvin_helmholtz_instability.jl")
#---
