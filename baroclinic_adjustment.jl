# Baroclinic adjustment (double-front, doubly-periodic-horizontal channel)
using Oceananigans
using Oceananigans.Units
using Printf
using Random
using ArgParse
using Oceanostics: PotentialEnergyEquation, GaussianFilter

@info "Finished loading packages"
Random.seed!(8675309)

include("utils.jl")

#+++ Workaround for a bug in Oceanostics v0.17.2's multi-direction `GaussianFilter` staging
# `GaussianFilter(; dims=(1,2), σ)` on a grid with real Ny>1 and periodic y crashes with a
# heap-corruption SIGILL (isolated via bisection: base model alone runs clean; adding filtered fields
# with dims=(1,2) crashes after the first timestep; dims=(1,) alone runs clean). Per the package's own
# docstring, a multi-direction filter is normally evaluated as "a sequence of 1D passes through
# intermediate fields" via an internal `_compute_staged_filter!` path -- that staged path is what's
# broken for a periodic y-direction; the single-direction path each 1D filter falls through to
# ("1D filters ... use the unrolled single-direction kernel") is not. So instead of asking
# Oceanostics to combine both directions in one call, we do the (mathematically equivalent, since
# Gaussian filtering is separable) two 1D passes ourselves, each going through the working
# single-direction path. Reported upstream: https://github.com/tomchor/Oceanostics.jl/issues/262
#
# NOTE: this wrapper is only applied to the plain filtered-field outputs below, not threaded through
# Oceanostics' composite cross-scale functions (KineticEnergyCrossScaleFlux,
# CoarseGrainedKineticEnergyDissipationRate) -- doing so caused a separate, severe compile-time blowup
# (the wrapper doubles the nesting depth of an already deeply nested KernelFunctionOperation tree,
# and LLVM choked on it). Those composite diagnostics (Πₖ, ε_Kˢ) are deferred to offline Python
# post-processing instead (see Phase 2 of the project plan) until the upstream bug is fixed.
struct SequentialGaussianFilter{D, S}
    dims::D
    σ::S
end
SequentialGaussianFilter(; dims, σ) = SequentialGaussianFilter(dims, σ)

function (F::SequentialGaussianFilter)(ψ)
    result = ψ
    # Materialize every intermediate stage (required so the next 1D pass reads concrete data, not a
    # lazy expression tree), but leave the *final* stage as a lazy operation -- matching the native
    # `GaussianFilter`'s own calling convention, since callers (e.g. `filtered_velocities` in
    # Oceanostics) do `Field(filter(ψ))` themselves and double-wrapping `Field(Field(...))` is unnecessary.
    for (i, d) in enumerate(F.dims)
        op = GaussianFilter(result; dims=(d,), σ=F.σ)
        result = i < length(F.dims) ? Field(op) : op
    end
    return result
end
#---

#+++ Parse command-line arguments
let s = ArgParseSettings()
    @add_arg_table! s begin
        "--Nx"
            help = "Number of x grid points (default: 48, matching the Oceananigans baroclinic_adjustment example)"
            arg_type = Int
            required = false
            default = 48

        "--Ny"
            help = "Number of y grid points (default: 48)"
            arg_type = Int
            required = false
            default = 48

        "--Nz"
            help = "Number of z grid points (default: 8)"
            arg_type = Int
            required = false
            default = 8

        "--N2"
            help = "Background stratification N² (default: 1e-5 s⁻²)"
            arg_type = Float64
            required = false
            default = 1e-5

        "--M2"
            help = "Horizontal buoyancy gradient M² across each front (default: 1e-7 s⁻²)"
            arg_type = Float64
            required = false
            default = 1e-7

        "--front_width"
            help = "Width of each front, in km (default: 100)"
            arg_type = Float64
            required = false
            default = 100.0

        "--perturbation_amplitude"
            help = "Noise amplitude as a fraction of the front's buoyancy jump Δb (default: 0.01)"
            arg_type = Float64
            required = false
            default = 0.01

        "--latitude"
            help = "Latitude for the beta-plane Coriolis approximation (default: -45)"
            arg_type = Float64
            required = false
            default = -45.0

        "--nu"
            help = "Scalar viscosity ν for the explicit closure (default: 1.0 m² s⁻¹; kept simple/isotropic \
                    so the online SFS dissipation diagnostic stays well-defined, as in the KH setup)"
            arg_type = Float64
            required = false
            default = 1.0

        "--Pr"
            help = "Prandtl number; sets κ = ν / Pr (default: 1.0)"
            arg_type = Float64
            required = false
            default = 1.0

        "--stop_time"
            help = "Simulation stop time in days (default: 20)"
            arg_type = Float64
            required = false
            default = 20.0

        "--filter_scales"
            help = "Two horizontal filter scales (FWHM, in km) for the online coarse-graining diagnostics (default: 50 100; \
                    kept modest relative to the example's ~20.8km grid spacing so the Gaussian filter's stencil radius stays \
                    small relative to the domain -- see the halo-sizing note where the grid is built)"
            arg_type = Float64
            nargs = 2
            required = false
            default = [50.0, 100.0]

        "--progress_interval"
            help = "Print a progress message every this many iterations (default: 100; use a small value \
                    like 1 for short debug/smoke-test runs where 100 iterations may never be reached)"
            arg_type = Int
            required = false
            default = 100
    end
    global parsed_args = parse_args(s, as_symbols=true)
end
filter_scales_km = pop!(parsed_args, :filter_scales)
params = (; parsed_args...)
#---

#+++ Define simulation parameters
# Domain matches the Oceananigans baroclinic_adjustment literated example exactly (Lx=Ly=1000km, Lz=1km),
# but topology is (Periodic, Periodic, Bounded) instead of (Periodic, Bounded, Bounded): rather than a
# single front against two channel walls, we place *two* opposite-signed fronts so the buoyancy field
# closes periodically in y (see the double-ramp `bᵢ` below). This avoids side-wall boundary layers, which
# would otherwise complicate the coarse-grained KE/APE budget (extra wall dissipation term) and the
# horizontal Gaussian filter (which would need edge-extension at the bounded y walls, as in the original
# KH setup's bounded z). With both horizontal directions periodic, the filter is a pure periodic wrap.
Lx = 1000kilometers
Ly = 1000kilometers
Lz = 1kilometers
params = (; params..., Lx, Ly, Lz)
#---

#+++ Create grid
Nx = closest_factor_number((2, 3, 5), params.Nx)
Ny = closest_factor_number((2, 3, 5), params.Ny)
Nz = params.Nz
params = (; params..., Nx, Ny)

# The horizontal Gaussian filter's stencil radius (truncated at 4σ, matching scipy's default) must fit
# within the grid's halo, or the filter's @inbounds accesses run past the allocated halo region and
# silently corrupt memory (observed as a segfault at an unrelated later point, e.g. inside `show` while
# printing an unrelated error/warning -- NOT a clean bounds-check failure). The default halo Oceananigans
# picks is sized for the advection scheme (a few points for Centered(4)), not for whatever filter scale is
# requested, so it must be sized explicitly here from the *largest* requested filter scale.
_FWHM_to_σ(ℓ) = ℓ / (2 * sqrt(2 * log(2)))
_filter_radius(σ, Δ) = max(1, floor(Int, 4σ / Δ + 0.5))

Δx = params.Lx / Nx
Δy = params.Ly / Ny
σmax = _FWHM_to_σ(maximum(filter_scales_km) * kilometers)
Hx = max(3, _filter_radius(σmax, Δx))
Hy = max(3, _filter_radius(σmax, Δy))

grid = RectilinearGrid(size=(Nx, Ny, Nz),
                       x=(0, params.Lx),
                       y=(-params.Ly/2, params.Ly/2),
                       z=(-params.Lz, 0),
                       halo=(Hx, Hy, 3),
                       topology=(Periodic, Periodic, Bounded))
@info "Grid created: Nx=$Nx, Ny=$Ny, Nz=$Nz, halo=($Hx, $Hy, 3)"
#---

#+++ Create model
model = HydrostaticFreeSurfaceModel(grid;
                                    free_surface = ImplicitFreeSurface(),
                                    coriolis = BetaPlane(latitude=params.latitude),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    momentum_advection = Centered(order=4),
                                    tracer_advection = Centered(order=4),
                                    closure = ScalarDiffusivity(ν=params.nu, κ=params.nu/params.Pr))
u, v, w = model.velocities
b = model.tracers.b
@info "Model created"
#---

#+++ Define initial conditions: double front + background stratification + noise
# `ramp(y, Δy)` is the same linear ramp (0 → 1 over a width Δy) used in the Oceananigans example. A single
# ramp centered at y₀ would leave b(-Ly/2) ≠ b(Ly/2), which isn't compatible with a periodic y topology.
# Instead we place two ramps of opposite sign at y₁ = -Ly/4 and y₂ = +Ly/4: the buoyancy is low outside
# the two fronts (wrapping through the periodic boundary) and high in the plateau between them, so the
# profile is periodic by construction and the domain hosts two opposite-signed fronts (a standard
# "double-front"/periodic-strip configuration used to avoid channel side walls).
ramp(y, Δy) = min(max(0, y/Δy + 1/2), 1)

Δy = params.front_width * kilometers
Δb = Δy * params.M2
ϵb = params.perturbation_amplitude * Δb

y₁ = -params.Ly/4
y₂ = +params.Ly/4

double_ramp(y, Δy) = ramp(y - y₁, Δy) - ramp(y - y₂, Δy)

bᵢ(x, y, z) = params.N2 * z + Δb * double_ramp(y, Δy) + ϵb * randn()

set!(model, b=bᵢ)
@info "Initial conditions set"
#---

#+++ Setup simulation
simulation = Simulation(model, Δt=20minutes, stop_time=params.stop_time * days)
conjure_time_step_wizard!(simulation, IterationInterval(20), cfl=0.2, max_Δt=20minutes)
@info "Simulation object created"
#---

#+++ Add progress messenger
wall_clock = Ref(time_ns())

function print_progress(sim)
    u, v, w = model.velocities
    progress = 100 * (time(sim) / sim.stop_time)
    elapsed = (time_ns() - wall_clock[]) / 1e9

    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max|u,v,w|: (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            progress, iteration(sim), prettytime(sim), prettytime(elapsed),
            maximum(abs, u), maximum(abs, v), maximum(abs, w), prettytime(sim.Δt))

    wall_clock[] = time_ns()
    return nothing
end
add_callback!(simulation, print_progress, IterationInterval(params.progress_interval))
#---

#+++ Add output writer
# Interpolate velocities to (Center, Center, Center) before writing/filtering: leaving them on their
# native staggered (Face) locations makes the offline Python post-processing (which multiplies fields
# together, e.g. uⁱuʲ for the stress tensor) broadcast across mismatched staggered coordinate dims
# (x_faa, y_afa, z_aaf) instead of a pointwise product -- silently producing a hugely-wrong-shaped,
# nonsensical result. Matches the KH setup's convention (u_center/v_center/w_center).
u_center = @at (Center, Center, Center) u
v_center = @at (Center, Center, Center) v
w_center = @at (Center, Center, Center) w

ζ = Field(∂x(v) - ∂y(u))
ρ₀ = 1025 # kg/m^3
pe = ρ₀ * PotentialEnergyEquation.PotentialEnergy(model)
PE = Integral(pe)

#+++ Gaussian-filtered u, v, w, b at multiple filter scales — HORIZONTAL ONLY (dims=(1,2)).
# Coarse-graining is applied over (x, y) at each depth level, not in z: horizontal scales span the
# mesoscale/submesoscale range this budget targets, while the vertical direction has its own distinct
# structure (stratification, surface/bottom boundary layers) that shouldn't be smoothed over. Both
# horizontal directions are periodic here, so (unlike the KH setup's bounded-z filter) no edge-extension
# boundary handling is needed — it's a pure periodic wrap.
filter_ℓs = Tuple(filter_scales_km .* kilometers)
_fields = (u=u_center, v=v_center, w=w_center, b=b)
_filt_pairs = [Symbol("$(n)_ℓ$(round(Int, ℓ/1000))km") => SequentialGaussianFilter(dims=(1, 2), σ=_FWHM_to_σ(ℓ))(f) for ℓ in filter_ℓs for (n, f) in pairs(_fields)]
filtered_fields = (; _filt_pairs...)
#---

@info "Online diagnostics (filtered fields) built"
# NOTE: cross-scale KE transfer Πₖ and SFS dissipation ε_Kˢ (KineticEnergyCrossScaleFlux /
# CoarseGrainedKineticEnergyDissipationRate) are deferred to offline Python post-processing rather
# than computed online here -- see the note on SequentialGaussianFilter above for why.
#---

outputs = (; ζ, b, pe, PE, u=u_center, v=v_center, w=w_center, filtered_fields...)

using NCDatasets
simulation_name = "bci_Nx$(params.Nx)_Ny$(params.Ny)_Nz$(params.Nz)"
output_filename = "output/$(simulation_name).nc"

simulation.output_writers[:fields] =
    NetCDFWriter(model, outputs,
                 schedule = ConsecutiveIterations(TimeInterval(12hours)),
                 filename = output_filename,
                 array_type = Array{Float64},
                 global_attributes = params,
                 overwrite_existing = true)

output_filename_2d = "output/$(simulation_name)_surface.nc"
simulation.output_writers[:surface] =
    NetCDFWriter(model, outputs,
                 schedule = TimeInterval(12hours),
                 filename = output_filename_2d,
                 array_type = Array{Float32},
                 indices = (:, :, grid.Nz),
                 global_attributes = params,
                 overwrite_existing = true)

@info "Output writers configured. Output will be saved to: $(output_filename)"
#---

#+++ Run simulation
@info @sprintf("""
================================================================================
  Baroclinic adjustment (double-front, doubly-periodic-horizontal channel)
================================================================================
  Grid:          Nx=%d, Ny=%d, Nz=%d
  Domain:        Lx=%.1f km, Ly=%.1f km, Lz=%.1f km
  Stop time:     %.1f days
  N²=%.2e s⁻², M²=%.2e s⁻², front width=%.1f km
  Latitude:      %.1f
  ν=%.2e, κ=%.2e
================================================================================
""",
    params.Nx, params.Ny, params.Nz,
    params.Lx/1000, params.Ly/1000, params.Lz/1000,
    params.stop_time,
    params.N2, params.M2, params.front_width,
    params.latitude,
    params.nu, params.nu/params.Pr)
@info "Running baroclinic adjustment simulation..."
run!(simulation)
#---
