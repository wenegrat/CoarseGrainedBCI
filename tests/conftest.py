import os
import xarray as xr
from pathlib import Path

PP_OUTPUT = Path(__file__).parent.parent / "postprocessing" / "output"
STEM = os.environ.get("BCI_STEM", "bci_Nx48_Ny48_Nz8")


def pytest_addoption(parser):
    parser.addoption(
        "--ref-suffix",
        default="",
        help="Suffix appended to postprocessing output filenames (e.g. '_fixed_ref')",
    )
    parser.addoption(
        "--min-time-days",
        default=1.0,
        type=float,
        help="Exclude budget time samples before this many days from the residual calculation (default: "
             "1.0). The first ~1-2 saved time samples are dominated by a large initial transient (the "
             "density-sorting reference profile adjusting away from the perturbed initial condition, plus "
             "any initial-condition noise still decaying) that isn't part of the actual budget balance -- "
             "including it inflates rms(term) for whichever term the transient happens to dominate (often "
             "∫Rˢ dV or ∫-∂ₜ SFS APE dV), which either masks or distorts the real relative residual.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "ref_suffix: parameterise tests by reference-profile suffix",
    )


def pytest_generate_tests(metafunc):
    if "l_idx" in metafunc.fixturenames:
        ref_suffix = metafunc.config.getoption("--ref-suffix")
        path = PP_OUTPUT / f"{STEM}_sfs_ke_budget_integrated{ref_suffix}.nc"
        ds = xr.open_dataset(path, decode_timedelta=False)
        metafunc.parametrize("l_idx", range(len(ds.filter_scale)))
