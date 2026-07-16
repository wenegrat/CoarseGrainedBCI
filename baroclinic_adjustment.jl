# Baroclinic adjustment (double-front, doubly-periodic-horizontal channel)
using Oceananigans
using Oceananigans.Units
using Printf
using Random
using ArgParse
using Oceanostics: PotentialEnergyEquation, GaussianFilter, KineticEnergyCrossScaleFlux, SubFilterKineticEnergyDissipationRate, FilteredKineticEnergyDissipationRate

@info "Finished loading packages"
Random.seed!(8675309)

include("utils.jl")

# Πₖ and ε_Kˢ used to be deferred to offline Python post-processing because Oceanostics v0.17.2's
# multi-direction `GaussianFilter(; dims=(1,2), σ)` crashed (heap-corruption SIGILL) on a grid with
# real Ny>1 and periodic y -- see https://github.com/tomchor/Oceanostics.jl/issues/262. Fixed upstream
# in v0.17.3 (PR #263). `KineticEnergyCrossScaleFlux`/`SubFilterKineticEnergyDissipationRate` (the
# latter added in the still-unmerged `tc/sfs-ke` branch, pinned in Project.toml/Manifest.toml) are now
# computed online directly below -- both validated against the previous offline Python implementation
# on a smoke-test run (0.99 spatial correlation, rms agreement within ~1%) before switching over.

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
                    grid-Péclet-number criterion, ν = (U/Pe_cell)·Δ per direction -- shrinks automatically \
                    as the grid is refined, and is automatically much smaller vertically than horizontally \
                    since Δz ≪ Δx,Δy; see --Pe_cell_h/--Pe_cell_v), or 'smagorinsky' (SmagorinskyLilly LES closure -- NOT \
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
                    unless --closure=constant; see --Pe_cell_h/--Pe_cell_v for the 'scale_aware' closure instead."
            arg_type = Float64
            required = false
            default = 1.0

        "--nu_v"
            help = "Vertical viscosity ν_v for the 'constant' closure (default: 1.0 m² s⁻¹; see --nu_h). \
                    Ignored unless --closure=constant."
            arg_type = Float64
            required = false
            default = 1.0

        "--Pe_cell_h"
            help = "Target horizontal cell Péclet number Pe = UΔx/ν_h for the 'scale_aware' closure \
                    (default: 100.0 -- empirically tuned against horizontal surface-buoyancy smoothness \
                    across a resolution sweep; well above the classical linear-stability threshold of ~2, \
                    which was found to over-damp the actual baroclinic instability). Sets \
                    ν_h = (U/Pe_cell_h)·√(Δx·Δy), where U = M²·Lz/f is the thermal-wind velocity scale \
                    intrinsic to this problem's own parameters. Lower Pe_cell_h -> more horizontal \
                    dissipation. Independent of --Pe_cell_v (see its help for why they're not tied \
                    together). Ignored unless --closure=scale_aware."
            arg_type = Float64
            required = false
            default = 100.0

        "--Pe_cell_v"
            help = "Target vertical cell Péclet number Pe = UΔz/ν_v for the 'scale_aware' closure \
                    (default: 50.0 -- empirically re-tuned by a Pe_cell_v sweep at 96x96x16, see below; NOT \
                    the same as --Pe_cell_h's 100.0). Sets ν_v = (U/Pe_cell_v)·Δz. Kept separate from \
                    --Pe_cell_h because there's no reason the same empirically-tuned Péclet target should \
                    transfer between directions: the two grid spacings differ by ~2 orders of magnitude and \
                    the physics forcing each direction's grid-scale noise (horizontal straining vs. vertical \
                    shear) is different. Centered advection has no implicit dissipation, so nonlinear terms \
                    continuously alias energy toward the grid scale in both directions as soon as any \
                    resolved eddying motion exists -- sharing a single Pe_cell=100 under-damped this in the \
                    vertical once Δz was refined (visibly noisier fields, persistently-elevated SFS budget \
                    terms that didn't shrink with resolution the way a one-off IC transient would). A sweep \
                    over Pe_cell_v ∈ {100,50,20,10,5,2} at 96x96x16 found: below ~20 the flow becomes visibly \
                    over-damped (max|u| dropping from ~2 m/s at Pe_cell_v=100 to ~0.1 m/s at Pe_cell_v=2 --  \
                    baroclinic instability itself gets suppressed, not just grid noise), which spuriously \
                    *improves* the KE-budget residual ratio simply because all terms shrink together, not \
                    because the numerics genuinely improved -- watch for this artifact rather than chasing \
                    the residual ratio down in isolation. Pe_cell_v=50 was the best point that still matched \
                    the baseline's growth strength while measurably improving KE closure (residual ratio \
                    ~30-42% -> ~22-28%). Lower Pe_cell_v -> more vertical dissipation. Ignored unless \
                    --closure=scale_aware."
            arg_type = Float64
            required = false
            default = 50.0

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

        "--output_interval_hours"
            help = "Interval between saved 3D/surface output snapshots, in hours (default: 12.0). Every \
                    offline coarse-graining diagnostic (∂ₜ tendency, Πₖ/Π_A cross-scale flux, εₖˢ/ε_Aˢ \
                    dissipation) is computed only from these saved snapshots -- a coarser interval means a \
                    coarser centered time-difference for tendencies, and nonlinear flux/dissipation \
                    products (uᵢuⱼ, uᵢρ) evaluated at more widely-spaced-in-time snapshots rather than the \
                    continuously-evolving field. Smaller values test whether the budget-closure residual is \
                    partly an aliasing artifact of insufficient temporal sampling."
            arg_type = Float64
            required = false
            default = 12.0

        "--filter_scales"
            help = "Two horizontal filter scales (FWHM, in km) for the online coarse-graining diagnostics \
                    (default: 50 100). No halo-size penalty for large scales -- the periodic (x,y) filter \
                    stencil reads via wrapped interior indexing, not halo cells (see the grid-construction note)."
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

# The horizontal Gaussian filter's periodic (x, y) stencil does NOT need a halo sized to its truncation
# radius: Oceanostics' GaussianFilter reads periodic-direction neighbors via a wrapped *interior* array
# index (`wrap_periodic_index`), never through halo/ghost cells, however wide the stencil is (confirmed
# directly in Oceanostics' SpatialFilters source, and by tomchor). A small, fixed halo -- matching the z
# halo below, and Oceananigans' own advection-scheme default -- is sufficient regardless of filter scale.
# (An earlier version of this file sized Hx/Hy from the filter's 4σ radius, defending against a heap-
# corruption bug that was actually a kernel-launch-sizing issue in Oceanostics <v0.17.3, unrelated to halo
# width -- see the note above on issue #262/PR #263, which this repo is already pinned past.)
_FWHM_to_σ(ℓ) = ℓ / (2 * sqrt(2 * log(2)))

Δx = params.Lx / Nx
Δy = params.Ly / Ny
Δz = params.Lz / Nz

grid = RectilinearGrid(size=(Nx, Ny, Nz),
                       x=(0, params.Lx),
                       y=(-params.Ly/2, params.Ly/2),
                       z=(-params.Lz, 0),
                       halo=(3, 3, 3),
                       topology=(Periodic, Periodic, Bounded))
@info "Grid created: Nx=$Nx, Ny=$Ny, Nz=$Nz, halo=(3, 3, 3)"
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
# grid-Péclet-number criterion instead of a fixed value: ν_h = (U/Pe_cell_h)·√(Δx·Δy), ν_v =
# (U/Pe_cell_v)·Δz, where U is a velocity scale intrinsic to this problem (the thermal-wind scale
# U = M²·Lz/f). This makes ν shrink automatically as the grid is refined, and is automatically
# anisotropic (νv ≪ νh follows directly from Δz ≪ Δx,Δy) -- while remaining an explicit, deterministic
# Laplacian closure (not a residual, not high-order/biharmonic), so the SFS dissipation diagnostic stays
# as well-defined as the 'constant' closure's. Pe_cell_h and Pe_cell_v are independent CLI parameters
# (--Pe_cell_h defaults to 100, --Pe_cell_v to 50 -- see --Pe_cell_v's help for the sweep that picked 50):
# a resolution sweep with a single shared Pe_cell=100 (tuned only by eye against horizontal
# surface-buoyancy smoothness) showed visibly noisier fields and
# persistently-elevated (not resolution-decaying) SFS budget terms as Δz was refined at fixed Pe_cell --
# i.e. the horizontally-tuned Pe_cell under-damps continuously-regenerated vertical grid-scale noise
# (Centered advection has no implicit dissipation, so nonlinear terms keep aliasing energy toward the
# grid scale in both directions once any resolved eddying motion exists). Splitting the two lets each
# direction's dissipation be tuned to its own noise-generation/damping balance.
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
    global νh_scale_aware = (U_scale / params.Pe_cell_h) * sqrt(Δx * Δy)
    global νv_scale_aware = (U_scale / params.Pe_cell_v) * Δz
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

# NonhydrostaticModel instead of HydrostaticFreeSurfaceModel: no free surface at all (no η, no
# barotropic pressure correction -- see conversation for the NetCDFWriter dimension-inference limitation
# that made patching the free-surface term into the old hydrostatic setup impractical), and w becomes a
# genuine prognostic variable with its own momentum equation and dissipative dynamics rather than
# diagnostic. This matches tomchor's own Eady baroclinic-instability example (Oceanostics PR #260),
# which closes its coarse-grained KE budget cleanly (~11-15% residual/dominant, vs our ~40-60% under the
# old hydrostatic setup) -- see conversation for the ongoing investigation into how much of that gap is
# closure/numerics vs. the buoyancy-production-term convention (w̄b̄ vs w̄b̄ᵣ).
model = NonhydrostaticModel(grid;
                            coriolis = BetaPlane(latitude=params.latitude),
                            buoyancy = BuoyancyTracer(),
                            tracers = :b,
                            advection = advection_scheme,
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
# Start conservatively (Δt=1minute) rather than jumping straight to max_Δt: the random per-cell noise in
# the initial buoyancy perturbation (ϵb·randn()) gets proportionally sharper as the grid is refined (same
# noise amplitude spread over a smaller Δx means a steeper initial gradient), and at fine resolution this
# can blow up Centered advection within the first handful of iterations -- faster than a wizard that only
# re-checks every IterationInterval(20) steps can react. IterationInterval(5) lets it adapt quickly once
# the flow is actually resolved smoothly (usually within the first ~10-20 iterations).
simulation = Simulation(model, Δt=1minute, stop_time=params.stop_time * days)
# diffusive_cfl matters once the 'scale_aware' closure is in play: ν grows at coarser resolution (it's
# tied to Δ, not fixed), so the explicit horizontal-diffusion stability limit can become the binding
# constraint (tighter than plain advective CFL) -- without this, a run can blow up (NaN) well before the
# advective cfl=0.2 limit would ever ask for a smaller Δt.
conjure_time_step_wizard!(simulation, IterationInterval(5), cfl=0.2, diffusive_cfl=0.2, max_Δt=20minutes)
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
gaussian_filters = [GaussianFilter(; dims=(1, 2), σ=_FWHM_to_σ(ℓ)) for ℓ in filter_ℓs]  # one reusable filter object per scale

_fields = (u=u_center, v=v_center, w=w_center, b=b)
_filt_pairs = [Symbol("$(n)_ℓ$(round(Int, ℓ/1000))km") => gf(f) for (ℓ, gf) in zip(filter_ℓs, gaussian_filters) for (n, f) in pairs(_fields)]
filtered_fields = (; _filt_pairs...)
#---

@info "Online diagnostics (filtered fields) built"

#+++ Cross-scale KE flux Πₖ and SFS KE dissipation ε_Kˢ, per filter scale
# w is a genuine prognostic variable in this NonhydrostaticModel, with its own momentum equation and
# dissipative dynamics -- unlike the earlier HydrostaticFreeSurfaceModel setup, where w was diagnostic
# and had to be excluded from the KE budget for it to close even in principle. Πₖ is therefore the full
# 3D contraction (dims=(1,2,3)), not horizontal-only: restricting to horizontal-only here would
# reintroduce a different missing term (horizontal-vertical pressure redistribution) in place of the
# free-surface one this model switch removes.
#
# ε_Kˢ (SubFilterKineticEnergyDissipationRate) and εˡ (FilteredKineticEnergyDissipationRate) have no
# `dims` restriction in their public API at all -- both always include w's full contribution via the
# model's actual per-direction viscous fluxes, which is the physically correct behavior now (it was a
# small "phantom" w-diffusion term under the old hydrostatic setup, verified negligible there via a
# smoke test: ~1e-8 relative magnitude, 0.99 spatial correlation with the w-excluded offline formula).
#
# εˡ is included alongside Πₖ/ε_Kˢ for an ongoing investigation into whether the *filtered* (large-scale)
# KE budget (∂ₜK̄ = w̄b̄ᵣ - Πₖ - εˡ, K̄ = ½(ū²+v̄²+w̄²)) is a more robust diagnostic than the SFS budget this
# repo has focused on so far -- unlike ε_Kˢ = filter(ε) - εˡ (a difference of two large, closely-related
# quantities, fragile by construction), εˡ is a single directly-computed term with no cancellation,
# matching how tomchor's own Eady baroclinic-instability example (Oceanostics PR #260) closes its
# coarse-grained KE budget. That example's own setup (NonhydrostaticModel, no free surface, full 3D Πₖ)
# is exactly what this switch adopts; see conversation history for the ongoing investigation into how
# much of the closure gap is numerics vs. the buoyancy-production-term convention (w̄b̄ vs w̄b̄ᵣ).
_ke_pairs = vcat(
    [Symbol("Π_K_ℓ$(round(Int, ℓ/1000))km")  => KineticEnergyCrossScaleFlux(model, gf; dims=(1, 2, 3)) for (ℓ, gf) in zip(filter_ℓs, gaussian_filters)],
    [Symbol("ε_Kˢ_ℓ$(round(Int, ℓ/1000))km") => SubFilterKineticEnergyDissipationRate(model, gf)        for (ℓ, gf) in zip(filter_ℓs, gaussian_filters)],
    [Symbol("ε_l_ℓ$(round(Int, ℓ/1000))km")  => FilteredKineticEnergyDissipationRate(model, gf)         for (ℓ, gf) in zip(filter_ℓs, gaussian_filters)],
)
ke_budget_fields = (; _ke_pairs...)
#---

outputs = (; ζ, b, pe, PE, u=u_center, v=v_center, w=w_center, filtered_fields..., ke_budget_fields...)

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

output_interval = params.output_interval_hours * hours

simulation.output_writers[:fields] =
    NetCDFWriter(model, outputs,
                 schedule = ConsecutiveIterations(TimeInterval(output_interval)),
                 filename = output_filename,
                 array_type = Array{Float64},
                 global_attributes = params,
                 overwrite_existing = true)

output_filename_2d = "output/$(simulation_name)_surface.nc"
simulation.output_writers[:surface] =
    NetCDFWriter(model, outputs,
                 schedule = TimeInterval(output_interval),
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
