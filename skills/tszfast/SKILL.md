---
description: Compute and plot the tSZ C_ell^yy quickly using the classy_szfast JAX pipeline (cl_yy_from_params). For sanity checks, parameter sweeps, gradient probes, and benchmarks.
argument-hint: "[profile=arnaud10|battaglia12] [param=value ...]"
allowed-tools: Read Write Bash(~/pyvenvs/py312-class_sz/bin/python *)
---

# Quick tSZ C_ell^yy with the JAX pipeline

Run a fast forward / sweep / gradient computation of C_ell^yy using `classy_szfast.differentiable.cl_yy_from_params`. Use this for sanity checks before committing to a long cobaya run.

User-supplied arguments in `$ARGUMENTS`: optional `profile=arnaud10|battaglia12`, and any `name=value` overrides for cosmology or profile fields. If none given, use Planck-18 ΛCDM + Arnaud10 defaults.

## Steps

1. Write a self-contained Python script to a temp file (e.g. `/tmp/clyy_quick.py`) that:
   - Imports `jax`, `jnp`, sets `JAX_PLATFORM_NAME=cpu`, enables `jax_enable_x64`
   - Builds `CosmoParams` and `ProfileParamsA10` (or B12) with the user overrides
   - Calls `cl_yy_from_params(ell=jnp.geomspace(2, 5000, 80), cosmo, profile_params=profile, profile=<chosen>, delta_crit=<500 or 200>)`
   - Prints `D_ell × 1e12` at a handful of multipoles
   - Times one call (block until ready) and reports ms/eval

2. Run with `~/pyvenvs/py312-class_sz/bin/python /tmp/clyy_quick.py`.

3. Report back: timing, the D_ell values, anything anomalous (NaNs, monotonicity breaks, negative 1h, …). If the user asked for a plot, save it to `/tmp/clyy_quick.png` and tell them.

4. If the user asked for **gradients**, wrap the call in `jax.grad(lambda c, p: jnp.sum(cl_yy_from_params(...)[0]))(cosmo, profile)` and print the gradient w.r.t. each field.

5. If the user asked for a **parameter sweep**, loop over the swept variable, store D_ell arrays, save a multi-line plot to `/tmp/clyy_sweep.png`.

## Defaults / conventions

- Cosmology: Planck 18 (`omega_b=0.02242, omega_cdm=0.11933, H0=67.66, tau_reio=0.0561, ln10_10_As=3.047, n_s=0.9665`)
- Arnaud 10 defaults via `ProfileParamsA10()`; Battaglia 12 defaults via `ProfileParamsB12()`
- `delta_crit`: `500` for arnaud10, `200` for battaglia12 (always pair correctly)
- `ell`: `jnp.geomspace(2, 5000, 80)` unless user requests otherwise
- `n_z=100, n_m=200` (sane defaults; lower for faster, higher for converged precision tests)

## Pitfalls

- Don't forget `jax.config.update("jax_enable_x64", True)` — single-precision JAX gives wrong gradients on the halo integrals
- `block_until_ready()` matters for honest timing; JAX is async by default
- The JAX pipeline does NOT support `P0GNFW`-style overrides directly; you must use the `ProfileParamsA10(P0=..., gamma=..., alpha=..., beta=...)` field names
