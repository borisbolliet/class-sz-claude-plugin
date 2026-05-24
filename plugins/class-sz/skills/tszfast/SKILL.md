---
description: Compute and plot the tSZ C_ell^yy quickly using classy_szlite (cl_yy_factory). For sanity checks, parameter sweeps, gradient probes, and benchmarks before committing to a long MCMC run.
argument-hint: "[param=value ...]   e.g. P0=10  beta=5.0  ell_max=6000"
allowed-tools: Read Write Bash(~/pyvenvs/py312-class_sz/bin/python *)
---

# Quick tSZ C_ell^yy with classy_szlite

Run a fast forward / sweep / gradient computation of C_ell^yy using
`classy_szlite.cl_yy_factory`. Use this for sanity checks before
committing to a long cobaya / NumPyro run.

User-supplied arguments in `$ARGUMENTS`: optional `name=value`
overrides for cosmology or profile fields (e.g. `P0=10`,
`beta=5.0`, `omega_cdm=0.12`, `ell_max=6000`). If none given, use
Planck-18 ΛCDM defaults and the Arnaud-10 profile at its A10
best-fit values.

## Steps

1. Write a self-contained Python script to `/tmp/clyy_quick.py` that:
   - Sets `JAX_PLATFORM_NAME=cpu`, enables `jax_enable_x64`
   - Builds `CosmoParams` and `ProfileParamsA10` with the user
     overrides
   - Builds the factory once: `ev = csl.cl_yy_factory(cosmo, ell)`
   - Calls `ev(profile)`, prints `D_ell × 1e12` at a handful of
     multipoles
   - Times one warm call (`block_until_ready`) and reports ms / eval

2. Run with `~/pyvenvs/py312-class_sz/bin/python /tmp/clyy_quick.py`.

3. Report back: timing, the D_ell values, anything anomalous (NaNs,
   monotonicity breaks, negative 1h, …). If the user asked for a
   plot, save it to `/tmp/clyy_quick.png` and tell them.

4. If the user asked for **gradients**, wrap in `jax.grad`:
   ```python
   def loss(P0, beta):
       cl1, cl2 = ev(csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25))
       return jnp.sum(cl1 + cl2)
   g_P0, g_beta = jax.grad(loss, argnums=(0, 1))(8.13, 5.48)
   ```
   and print the gradient w.r.t. each field. (Note that for
   cosmology gradients you need the full `csl.cl_yy(...)` pipeline,
   not the factory — see the SKILL note below.)

5. If the user asked for a **parameter sweep**, loop over the swept
   variable (reusing the same `ev` if cosmology is fixed), store
   D_ell arrays, save a multi-line plot to `/tmp/clyy_sweep.png`.

## Reference template

```python
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np, time
import classy_szlite as csl

cosmo   = csl.CosmoParams()                                    # override fields here
profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)     # override fields here
ell     = jnp.geomspace(2, 5000, 80)

ev = csl.cl_yy_factory(cosmo, ell)                             # one-shot
cl_1h, cl_2h = ev(profile)
cl_1h.block_until_ready()

# Timing one warm call
t = []
for _ in range(20):
    t0 = time.perf_counter()
    a, b = ev(profile)
    a.block_until_ready(); b.block_until_ready()
    t.append(time.perf_counter() - t0)
print(f"warm call: {1e3*np.mean(t[3:]):.2f} ms ± {1e3*np.std(t[3:]):.2f}")

# D_ell at a few representative multipoles
for L in (100, 500, 1500, 3000):
    i = int(np.argmin(np.abs(np.asarray(ell) - L)))
    dl = (ell[i] * (ell[i] + 1) / (2 * np.pi)) * (cl_1h[i] + cl_2h[i]) * 1e12
    print(f"  ell={int(ell[i]):4d}   D_ell^yy × 1e12 = {float(dl):.4e}")
```

## Defaults / conventions

- Cosmology: Planck-18 ΛCDM via `CosmoParams()` (the defaults already
  represent the LCDM-equivalent point of the v2 emulators).
- Profile: `ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)` (the
  may26 ACT-DR6 baseline). All other fields default to A10 best fit
  (`c500=1.156, gamma=0.3292, alpha=1.062`).
- `ell`: `jnp.geomspace(2, 5000, 80)` unless user requests otherwise.
- `n_z=100, n_m=200` (sane defaults; lower for faster smoke tests,
  higher for converged precision work).

## Pitfalls

- Don't forget `jax.config.update("jax_enable_x64", True)` —
  single-precision JAX gives wrong gradients on the halo integrals.
- `block_until_ready()` matters for honest timing; JAX is async by
  default.
- `cl_yy_factory` precomputes CosmoGrids + HaloGrids ONCE — if you
  want to **sweep cosmology**, build a fresh `ev` per cosmology, OR
  use the full `csl.cl_yy(cosmo, profile, ell)` pipeline (slower,
  ~18 ms/eval warm).
- **Don't `jax.jit` the closure** — internals call `mcfit.TophatVar`,
  not jit-safe. The closure is already fast (~5 ms / call); `jax.grad`
  works directly on it.
- `ProfileParamsA10` field names are `P0, c500, gamma, alpha, beta, B`
  (no `GNFW` suffix) — the suffix is a cobaya-YAML convention only.
