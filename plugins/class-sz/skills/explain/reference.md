# classy_szlite — detailed reference

Load this when SKILL.md's surface isn't enough.

## Public API (everything top-level on `classy_szlite`)

### Parameter containers (JAX pytrees)

| Name | Module | Fields |
| --- | --- | --- |
| `CosmoParams` | `classy_szlite.params` | `omega_b, omega_cdm, H0, tau_reio, ln10_10_As, n_s, m_ncdm, N_ur, fEDE, log10z_c, thetai_scf, r` |
| `ProfileParamsA10` | `classy_szlite.params` | `P0, c500, gamma, alpha, beta, B` |

Both are `typing.NamedTuple` subclasses → JAX pytrees → you can
`jax.grad` w.r.t. them directly and get back the same container type
with `jax.Array` fields.

### Functions

| Function | Returns | One-line description |
| --- | --- | --- |
| `derived(cosmo)` | dict | σ8, Ω_m, S8 + 17-elt `der_full` array |
| `cl_TTTEEE(cosmo, spectra=(…), ell_factor=True)` | dict | CMB D_ℓ ^TT/TE/EE (dimensionless × T_CMB²) |
| `Pk(cosmo, z_arr)` | `(k, P_lin)` | linear matter power, k in h/Mpc, P in (Mpc/h)³ |
| `Pnl(cosmo, z_arr)` | `(k, P_nl)` | non-linear matter power (HMcode) |
| `distances(cosmo, z_arr)` | `(Hz/c, chi, Da)` | Hz/c in 1/Mpc, chi/Da in Mpc |
| `cl_yy(cosmo, profile, ell, …)` | `(cl_1h, cl_2h)` | full halo-model tSZ pipeline (~18 ms warm) |
| `cl_yy_factory(cosmo, ell, …)` | closure `ev(profile)` | precompute cosmology + halo grids → ~5 ms / call |
| `cosmo_to_dict(cosmo)` | dict | emulator-style kwargs for low-level use |

`cl_yy` and `cl_yy_factory` kwargs:
`n_z=100, n_m=200, m_min=1e10, m_max=3.5e15, delta_crit=500.0,
z_grid=None` (default `jnp.geomspace(0.005, 3.0, n_z)`).

## Cosmology parameter conventions

| Field | Default | Notes |
| --- | --- | --- |
| `omega_b`    | 0.02242  | physical baryon density |
| `omega_cdm`  | 0.11933  | physical CDM density |
| `H0`         | 67.66    | Hubble constant (km/s/Mpc) |
| `tau_reio`   | 0.054    | reionisation optical depth |
| `ln10_10_As` | 3.047    | log primordial amplitude (`ln 10^{10} A_s`) |
| `n_s`        | 0.9665   | scalar tilt |
| `m_ncdm`     | 0.02 eV  | per-species ν mass (3 deg ν → Σmν = 0.06 eV) |
| `N_ur`       | 0.00441  | ultra-relativistic species (Planck-18) |
| `fEDE`       | 0.001    | EDE fraction (silently LCDM-equivalent here) |
| `log10z_c`   | 3.562    | EDE critical redshift |
| `thetai_scf` | 2.83     | initial scalar-field angle |
| `r`          | 0.0      | tensor-to-scalar ratio |

**ΛCDM**: override only the standard 6 (`omega_b, omega_cdm, H0,
tau_reio, ln10_10_As, n_s`); everything else stays at the
LCDM-equivalent default.

**EDE / w-CDM / m_ν / N_eff exploration**: set the relevant extra
fields explicitly. The `v2` emulator covers all of these.

**ν convention**: `m_ncdm` is the per-species ν mass; `derived()`
computes `Σmν = 3 × m_ncdm` for `Ω_m`. The default `0.02 eV × 3 =
0.06 eV` matches Planck-18.

## ProfileParamsA10 (Arnaud 2010 GNFW)

| Field | Default | cobaya YAML name (conventional) |
| --- | --- | --- |
| `P0`    | 8.130  | `P0GNFW` |
| `c500`  | 1.156  | `c500` |
| `gamma` | 0.3292 | `gammaGNFW` |
| `alpha` | 1.062  | `alphaGNFW` |
| `beta`  | 5.4807 | `betaGNFW` |
| `B`     | 1.25   | `B` (hydrostatic mass bias; M_true = M_obs / B) |

Note the YAML side typically uses the `…GNFW` suffix convention; the
`/class-sz:build-likelihood` scaffold maps them onto the JAX field
names in the Theory's `calculate`.

## Halo-model integration controls

Defaults give max |ΔC/C| < 10⁻³ across ℓ ∈ [100, 8000] versus a
high-resolution reference:

| Kwarg | Default | Notes |
| --- | --- | --- |
| `n_z` | 100 | redshift grid (geomspace 0.005 → 3.0) |
| `n_m` | 200 | halo-mass grid (loglinear m_min → m_max) |
| `m_min` | 1e10 | lower mass cutoff (M_⊙/h); below ~1e10 contributes negligibly |
| `m_max` | 3.5e15 | upper mass cutoff (M_⊙/h); don't go below 3e15 |
| `delta_crit` | 500 | matches Arnaud 2010 / A10 |
| `z_grid` | `None` | override → `jnp.geomspace(0.005, 3.0, n_z)` |

Bumping `n_z` to 300 buys you ~10⁻⁴ accuracy at ~3× cost. `n_m` is
saturated by ~100 — past that, no measurable change.

## CosmoPower emulator data layout

`classy_szlite` does **not** bundle emulator data. Place the files
at one of (in priority order):

1. `$CLASSY_SZLITE_DATA_DIR`
2. `~/class_sz_data/`

Expected layout — the pickle-free `_v2_plain.npz` files:

```
<root>/ede/
├── TTTEEE/{TT,TE,EE}_v2_plain.npz
├── PP/PP_v2_plain.npz
├── PK/{PKL,PKNL}_v2_plain.npz
├── growth-and-distances/{HZ,DAZ,S8Z}_v2_plain.npz
└── derived-parameters/DER_v2_plain.npz
```

Get them from
[cosmopower-organization/ede](https://github.com/cosmopower-organization/ede).
They load with `allow_pickle=False` — no TensorFlow / cosmopower
needed.

## What the `v2` emulators cover

| Parameter direction | Covered? | Notes |
| --- | --- | --- |
| ΛCDM (6 std params) | ✓ | LCDM-equivalent point at `fEDE = 0.001` (default) |
| m_ν-ΛCDM            | ✓ | sample `m_ncdm` (per-species, Σmν = 3 m_ncdm) |
| w-CDM               | ✓ | sample `w` direction in DA + Hz emulators |
| N_eff-ΛCDM          | ✓ | sample `N_ur` |
| EDE (Smith+19)      | ✓ | sample `fEDE`, `log10z_c`, `thetai_scf` |
| r (tensor-to-scalar)| ✓ | sample `r`, BB outputs available |

Match to the CAMB-based [Jense et al. 2024
emulators](https://github.com/cosmopower-organization/jense_2024_emulators)
is **< 0.1 σ in ΛCDM** for ACT DR6 parameter constraints.

## Performance / throughput (Apple M-series, single-thread JAX)

| Function | mean ± std (ms) | calls/s |
| --- | --- | --- |
| `derived` | 0.54 ± 0.04 | 1850 |
| `cl_TTTEEE` | 2.52 ± 0.14 | 400 |
| `Pk` | 1.49 ± 0.12 | 670 |
| `distances` | 1.29 ± 0.09 | 770 |
| `cl_yy` (full) | 17.84 ± 0.58 | 56 |
| `cl_yy_factory` (fixed-cosmo) | 5.38 ± 0.42 | 185 |
| `cl_yy_factory + jax.grad` | 17.12 ± 1.01 | 58 |

Numbers are n=100 warm calls per benchmark with freshly randomised
inputs. JAX cold compile is ~1 s for the factory + a couple of
seconds for the full pipeline; both costs are paid once.

## Gradient examples

```python
import jax, jax.numpy as jnp
import classy_szlite as csl

cosmo = csl.CosmoParams()
ell = jnp.geomspace(2, 5000, 30)
ev = csl.cl_yy_factory(cosmo, ell)

# Profile-only (recommended path for inference)
def loss(P0, beta):
    cl1, cl2 = ev(csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25))
    return jnp.sum(cl1 + cl2)
g_P0, g_beta = jax.grad(loss, argnums=(0, 1))(8.13, 5.48)

# Cosmology + profile (full pipeline)
def full_loss(omega_cdm, P0, beta):
    c = csl.CosmoParams(omega_cdm=omega_cdm)
    p = csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25)
    cl1, cl2 = csl.cl_yy(c, p, ell)
    return jnp.sum(cl1 + cl2)
g = jax.grad(full_loss, argnums=(0, 1, 2))(0.118, 8.13, 5.48)

# Whole CosmoParams pytree at once
def cl_loss(cosmo):
    cl1, cl2 = csl.cl_yy(cosmo, csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25), ell)
    return jnp.sum(cl1 + cl2)
grads = jax.grad(cl_loss)(csl.CosmoParams())
# grads.omega_b, grads.omega_cdm, grads.fEDE, ...   all jax.Array
```

`jax.grad` matches central-difference finite differences to ~10⁻¹² —
exact to double-precision round-off.

## cobaya integration

`classy_szlite` does **not** ship a cobaya Theory wrapper out of the
box — you write a thin one per-likelihood, as in the SKILL.md
canonical pattern. The standard shape:

1. `initialize()` — build a `CosmoParams` from this Theory's class
   attributes (the standard 6, fixed), call
   `csl.cl_yy_factory(cosmo, ell)` once, stash the closure.
2. `get_can_provide()` returns `["Cl_sz"]` (or whatever your
   likelihood requires).
3. `calculate(state, **p)` builds a `ProfileParamsA10` from the
   sampled cobaya params, calls the closure, multiplies by the
   `ell(ell+1)/2π × 1e12` prefactor, stuffs the dict into `state`.
4. `get_Cl_sz(self)` returns `self._current_state["Cl_sz"]`.

For a NumPyro-based inference path (NUTS / HMC) instead of cobaya
RW-MH, skip the cobaya wrapper entirely and use the factory closure
inside a `numpyro.factor("loglike", ...)` call — see the SKILL.md
"Gradient-based sampling" recipe.

## File / directory conventions

- Local venv: `~/pyvenvs/py312-class_sz/bin/python` (+ `pip`,
  `cobaya-run`, `cobaya-install`, `getdist`)
- classy_szlite source (editable install): `~/GitHub/classy_szlite/`
- Emulator data: `~/class_sz_data/ede/` (or `$CLASSY_SZLITE_DATA_DIR`)
- Canonical test workdir for the may26 / ACT-DR6 fit:
  `~/Desktop/class-sz-plugin-tests/`
  - `data/` — bandpowers, cov, multipoles, foreground template
  - `clyy_v2.py` — the Likelihood + Theory + Foreground module
  - `clyy_v2.yaml` — cobaya input
  - `may26.proposal.covmat` — proposal covmat (2×2 over P0GNFW, betaGNFW)
  - `chains/` — cobaya output

## Common pitfalls

- **CWD shadowing**: don't run `cobaya-run` from `~/GitHub` — there's
  a `cobaya/` subfolder there that PEP 420 picks up before the
  editable install. Always `cd <workdir>` first.
- **Multipoles must match data**: the `multipoles_file` ell centres
  must equal the bandpower data file's ell centres exactly.
  Silent bug otherwise.
- **JAX precision**: `jax_enable_x64` is set on import. Don't disable
  it — likelihood posteriors will be biased at the bandpower
  covariance level otherwise.
- **`m_ncdm` is per-species** (3 degenerate ν by convention) so
  `Σmν = 3 × m_ncdm`. `derived()` uses this.
- **Don't `jax.jit(ev)`** — internals call `mcfit.TophatVar` which
  is not jit-safe.
