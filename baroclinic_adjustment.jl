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

        "--closure"
            help = "Turbulence closure: 'constant' (ScalarDiffusivity/anisotropic-Laplacian with fixed \
                    --nu_h/--nu_v, matches the KH setup's explicit-dissipation design, but doesn't converge \
                    as resolution improves), 'scale_aware' (default; anisotropic Laplacian with ν set from a \
                    grid-Péclet-number criterion, ν = (U/Pe_target)·Δ per direction -- shrinks automatically \
                    as the grid is refined, and is automatically much smaller vertically than horizontally \
                    since Δz ≪ Δx,Δy; see --Pe_cell), or 'smagorinsky' (SmagorinskyLilly LES closure -- NOT \
                    recommended at this resolution: its Richardson-number stability correction (Cb) zeroes \
                    out the eddy viscosity almost everywhere at mesoscale-permitting resolution, since \
                    resolved horizontal straining is tiny compared to the background stratification; \
                    disabling Cb instead makes it wildly too large, since its filter width assumes an \
                    isotropic grid cell -- kept only for reference/future work, see --C_smag/--Cb_smag)"
            arg_type = String
            required = false
            default = "scale_aware"
            range_tester = (s -> s in ("constant", "scale_aware", "smagorinsky"))

        "--nu_h"
            help = "Horizontal viscosity ν_h for the 'constant' closure (default: 1.0 m² s⁻¹). Ignored \
                    unless --closure=constant; see --Pe_cell for the 'scale_aware' closure instead."
            arg_type = Float64
            required = false
            default = 1.0

        "--nu_v"
            help = "Vertical viscosity ν_v for the 'constant' closure (default: 1.0 m² s⁻¹; see --nu_h). \
                    Ignored unless --closure=constant."
            arg_type = Float64
            required = false
            default = 1.0

        "--Pe_cell"
            help = "Target cell Péclet number Pe = UΔ/ν for the 'scale_aware' closure (default: 2.0, the \
                    classical threshold above which Centered-scheme advection-diffusion produces spurious \
                    2Δx grid-scale oscillations). Sets ν_h = (U/Pe_cell)·Δx (or Δy) and ν_v = (U/Pe_cell)·Δz, \
                    where U = M²·Lz/f is the thermal-wind velocity scale intrinsic to this problem's own \
                    parameters (not an arbitrary choice). Lower Pe_cell -> more dissipation. Ignored unless \
                    --closure=scale_aware."
            arg_type = Float64
            required = false
            default = 2.0

        "--C_smag"
            help = "Smagorinsky constant C for the 'smagorinsky' closure (default: 0.16, Lilly 1966). \
                    νₑ = (C·Δᶠ)²·√(2Σ²)·√(1 - Cb·N²/Σ²). Ignored unless --closure=smagorinsky."
            arg_type = Float64
            required = false
            default = 0.16

        "--Cb_smag"
            help = "Stratification-correction multiplier Cb for the 'smagorinsky' closure (default: 1.0; \
                    set to 0 to disable -- see --closure's help for why neither setting works well at this \
                    resolution). Ignored unless --closure=smagorinsky."
            arg_type = Float64
            required = false
            default = 1.0

        "--Pr"
            help = "Prandtl number: sets κ_h = ν_h/Pr, κ_v = ν_v/Pr for the 'constant'/'scale_aware' \
                    closures, or the turbulent Prandtl number κₑ = νₑ/Pr for 'smagorinsky' (default: 1.0)"
            arg_type = Float64
            required = false
            default = 1.0

        "--advection_scheme"
            help = "Advection scheme: 'centered' (default; Centered(order=4), matches KH's setup) or 'weno' \
                    (WENO(order=5), matches the Oceananigans baroclinic_adjustment example; has its own \
                    implicit, scale-selective numerical dissipation on top of whichever explicit closure is \
                    also active -- recommended pairing is --advection_scheme=weno --closure=smagorinsky)"
            arg_type = String
            required = false
            default = "centered"
            range_tester = (s -> s in ("centered", "weno"))

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
Δz = params.Lz / Nz
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
# Three closure options (--closure):
#
# 'constant' -- anisotropic Laplacian ScalarDiffusivity with fixed --nu_h/--nu_v. The grid is highly
# anisotropic (Δx,Δy ~ 20km vs Δz ~ 125m), so a single isotropic ν/κ gives a vertical diffusive damping
# time orders of magnitude shorter than horizontal at the same value -- e.g. ν=κ=1 m²/s gives a ~264 day
# horizontal damping time at the deformation radius (harmless over a 20-day run) but a ~4.3 hour vertical
# damping time (much faster than the ~1.2 day Eady growth time), smearing out the vertical
# shear/stratification structure baroclinic instability depends on before it can develop. Its downside:
# a fixed ν doesn't converge as the grid is refined, so it isn't scale-selective enough to control
# Centered's lack of implicit dissipation without either over-damping real structure (too large) or
# under-damping grid-scale noise (too small) -- see 'scale_aware' below.
#
# 'scale_aware' (default) -- anisotropic Laplacian ScalarDiffusivity, but with ν set automatically from a
# grid-Péclet-number criterion instead of a fixed value: ν = (U/Pe_cell)·Δ per direction, where U is a
# velocity scale intrinsic to this problem (the thermal-wind scale U = M²·Lz/f) and Pe_cell (--Pe_cell,
# default 2) is the classical grid-Péclet-number threshold above which Centered-scheme
# advection-diffusion produces spurious 2Δx grid-scale oscillations. This makes ν shrink automatically
# as the grid is refined (Δx,Δy,Δz → 0), and automatically anisotropic (νv ≪ νh follows directly from
# Δz ≪ Δx,Δy, with no hand-tuning) -- while remaining an explicit, deterministic Laplacian closure (not a
# residual, not high-order/biharmonic), so the SFS dissipation diagnostic stays as well-defined as the
# 'constant' closure's.
#
# 'smagorinsky' -- SmagorinskyLilly LES closure: a *diagnostic* eddy viscosity νₑ computed from the
# locally resolved strain rate. NOT recommended at this resolution: testing showed its Richardson-number
# stability correction (Cb term) clamps νₑ to exactly zero almost everywhere, since the resolved
# horizontal straining at mesoscale-permitting resolution is orders of magnitude smaller than the
# background stratification N² -- and disabling Cb (--Cb_smag 0) instead makes νₑ wildly too large
# (~1000s of m²/s), since SmagorinskyLilly's filter width assumes an isotropic grid cell, which this grid
# is not. Kept only for reference/future work at finer resolution.
νh_scale_aware = νv_scale_aware = nothing  # populated below only for --closure=scale_aware
closure = if params.closure == "smagorinsky"
    SmagorinskyLilly(C=params.C_smag, Cb=params.Cb_smag, Pr=params.Pr)
elseif params.closure == "scale_aware"
    Ω_earth = 7.2921159e-5  # rad/s
    f = 2 * Ω_earth * sind(params.latitude)
    U_scale = params.M2 * params.Lz / abs(f)
    global νh_scale_aware = (U_scale / params.Pe_cell) * sqrt(Δx * Δy)
    global νv_scale_aware = (U_scale / params.Pe_cell) * Δz
    κh, κv = νh_scale_aware / params.Pr, νv_scale_aware / params.Pr
    (HorizontalScalarDiffusivity(ν=νh_scale_aware, κ=κh),
     VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=νv_scale_aware, κ=κv))
else
    νh, νv = params.nu_h, params.nu_v
    κh, κv = params.nu_h / params.Pr, params.nu_v / params.Pr
    (νh == 0 && κh == 0 && νv == 0 && κv == 0) ? nothing :
        (HorizontalScalarDiffusivity(ν=νh, κ=κh),
         VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=νv, κ=κv))
end

advection_scheme = params.advection_scheme == "weno" ? WENO(order=5) : Centered(order=4)

# Overwrite nu_h/nu_v with the actual computed values so they're recorded correctly in the NetCDF
# global attributes below -- the Python postprocessing pipeline falls back to these attributes for any
# closure whose ν/κ isn't written out as a spatial field (i.e. everything except 'smagorinsky').
if params.closure == "scale_aware"
    global params = (; params..., nu_h=νh_scale_aware, nu_v=νv_scale_aware)
end

model = HydrostaticFreeSurfaceModel(grid;
                                    free_surface = ImplicitFreeSurface(),
                                    coriolis = BetaPlane(latitude=params.latitude),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    momentum_advection = advection_scheme,
                                    tracer_advection = advection_scheme,
                                    closure = closure)
u, v, w = model.velocities
b = model.tracers.b
if params.closure == "scale_aware"
    @info "Model created (advection=$(params.advection_scheme), closure=scale_aware, νh=$νh_scale_aware, νv=$νv_scale_aware)"
else
    @info "Model created (advection=$(params.advection_scheme), closure=$(params.closure))"
end
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
# diffusive_cfl matters once the 'scale_aware' closure is in play: ν grows at coarser resolution (it's
# tied to Δ, not fixed), so the explicit horizontal-diffusion stability limit can become the binding
# constraint (tighter than plain advective CFL) -- without this, a run can blow up (NaN) well before the
# advective cfl=0.2 limit would ever ask for a smaller Δt.
conjure_time_step_wizard!(simulation, IterationInterval(20), cfl=0.2, diffusive_cfl=0.2, max_Δt=20minutes)
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

# Smagorinsky's eddy viscosity/diffusivity are spatially/temporally varying fields (unlike the
# 'constant' closure's fixed ν/κ, which are recorded as scalar global attributes below), so they must be
# written out as actual output fields for the offline SFS dissipation diagnostics to use -- matching the
# KH setup's equivalent conditional block for non-constant closures.
if params.closure == "smagorinsky"
    νₑ = viscosity(model)
    κₑ = diffusivity(model, Val(:b))
    outputs = (; outputs..., νₑ, κₑ)
end

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
  Advection:     %s
  Closure:       %s%s
================================================================================
""",
    params.Nx, params.Ny, params.Nz,
    params.Lx/1000, params.Ly/1000, params.Lz/1000,
    params.stop_time,
    params.N2, params.M2, params.front_width,
    params.latitude,
    params.advection_scheme,
    params.closure,
    params.closure == "scale_aware" ? @sprintf(" (νh=%.4g, νv=%.4g m² s⁻¹)", params.nu_h, params.nu_v) : "")
@info "Running baroclinic adjustment simulation..."
run!(simulation)
#---
