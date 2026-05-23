---
description: class_sz / classy_szfast — CLASS extension for halo-model SZ observables. Two calculation pipelines (classic Class_sz, JAX cl_yy_from_params), pressure profiles (GNFW, Arnaud, Battaglia), mass functions, cobaya integration via classy_szfast.classy_sz.classy_sz, and the SOLikeT SZLikelihood pattern. Use when writing or debugging tSZ power spectrum calculations, scaffolding a y-map likelihood, or running cobaya MCMCs that fit C_ell^yy.
when_to_use: User mentions class_sz, classy_sz, classy_szfast, tSZ power spectrum, Cl_sz, y-map likelihood, SZLikelihood, GNFW / Arnaud / Battaglia pressure profile, P0GNFW, betaGNFW, cl_yy_from_params, CosmoParams, ProfileParamsA10/B12, ACT / Planck tSZ bandpowers.
allowed-tools: Read Grep Glob Bash(~/pyvenvs/py312-class_sz/bin/python *) Bash(~/pyvenvs/py312-class_sz/bin/cobaya-run *) Bash(~/pyvenvs/py312-class_sz/bin/cobaya-install *) Bash(~/pyvenvs/py312-class_sz/bin/getdist *)
---

# class_sz / classy_szfast assistant

Help with [class_sz](https://github.com/CLASS-SZ/class_sz) (CLASS extension for halo-model SZ observables) and [classy_szfast](https://github.com/CLASS-SZ/classy_szfast) (Python wrapper + emulators + JAX pipeline).

**Local venv:** `~/pyvenvs/py312-class_sz/bin/python` has classy_sz, classy_szfast, cobaya, soliket all installed. Always invoke that python when running examples.

## Two calculation pipelines — pick by use case

| Pipeline | Module | When to use |
| --- | --- | --- |
| **Classic** | `from classy_sz import Class as Class_sz` | Full halo-model observables (cluster counts, kSZ², trispectrum, CIB, etc.); production cobaya runs via `classy_szfast.classy_sz.classy_sz` |
| **JAX ultrafast** | `from classy_szfast.differentiable import cl_yy_from_params` | Fast/differentiable C_ell^yy only; gradient-based inference, emulator training, parameter sweeps; ~200 evals/s |

### Pipeline 1 — Classic `Class_sz()`

```python
from classy_sz import Class as Class_sz
c = Class_sz()
c.set(cosmo_params)        # omega_b, omega_cdm, H0, tau_reio, ln10^{10}A_s, n_s, cosmo_model
c.set(precision_params)    # n_z_pressure_profile, n_m_pressure_profile, FFT controls
c.set({
    'output': 'tSZ_1h',                   # or 'tSZ_tSZ_1h,tSZ_Trispectrum', 'kSZ_kSZ_1h', ...
    'mass_function': 'T08M500c',
    'pressure_profile': 'GNFW',           # or 'B12' (Battaglia), 'A10' (Arnaud)
    'multipoles': '/path/to/ls.txt',      # OR set ell_min/ell_max/dlogell
    'z_min': 0.005, 'z_max': 3.0,
    'M_min': 1e10, 'M_max': 3.5e15,
    'c500': 1.156, 'gammaGNFW': 0.3292, 'alphaGNFW': 1.062,
    'P0GNFW': 8.13, 'betaGNFW': 5.4807,
})
c.compute_class_szfast()                  # NOT compute() — fast mode uses emulators for cosmology
out = c.cl_sz()                           # {'ell': [...], '1h': [...], '2h': [...]}
T_llp = c.tllprime_sz()                   # trispectrum (when 'tSZ_Trispectrum' in output)
```

### Pipeline 2 — JAX `cl_yy_from_params`

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from classy_szfast.differentiable import CosmoParams, ProfileParamsA10, ProfileParamsB12, cl_yy_from_params

cosmo = CosmoParams(omega_b=0.02242, omega_cdm=0.11933, H0=67.66,
                    tau_reio=0.0561, ln10_10_As=3.047, n_s=0.9665)
profile = ProfileParamsA10()                                  # Arnaud 10 defaults; override P0, c500, gamma, alpha, beta

ell = jnp.geomspace(2, 5000, 80)
cl_1h, cl_2h = cl_yy_from_params(
    ell, cosmo, profile_params=profile,
    profile='arnaud10', delta_crit=500.0,                     # use 'battaglia12' + 200.0 for B12
    n_z=100, n_m=200,                                          # halo-model grids
)
# gradient via jax.grad — fully differentiable
g_cosmo, g_profile = jax.grad(lambda c, p: jnp.sum(cl_yy_from_params(ell, c, profile_params=p,
    profile='arnaud10', delta_crit=500.0)[0]), argnums=(0,1))(cosmo, profile)
```

## Pressure profiles

| Name | `pressure_profile` (classic) | `profile=` (JAX) | `delta_crit` | Sampled params |
| --- | --- | --- | --- | --- |
| GNFW / Arnaud 2010 | `GNFW` | `arnaud10` | 500 | `P0GNFW`, `c500`, `gammaGNFW`, `alphaGNFW`, `betaGNFW` |
| Battaglia 2012 | `B12` | `battaglia12` | 200 | `P0_A`, `P0_am`, `P0_az`, `xc_A`, `xc_am`, `xc_az`, `beta_A`, `beta_am`, `beta_az` |

In the JAX pipeline these are fields of the `ProfileParamsA10` / `ProfileParamsB12` NamedTuples — pass partial overrides as kwargs, defaults fill the rest.

## Mass functions

`mass_function` in the classic pipeline — common values: `T08M500c` (Tinker 2008 with M500c), `T08M200c`, `T08M200m`, `T10M200m`, `B16M200m` (Bocquet 2016), etc.

## cobaya integration

Theory wrapper: **`classy_szfast.classy_sz.classy_sz`** — subclasses `cobaya.theories.classy.classy`. It registers a `Collector` that calls `classy_sz.cl_sz()` and exposes the result via `provider.get_Cl_sz()`.

YAML:
```yaml
theory:
  classy_szfast.classy_sz.classy_sz:
    use_class_sz_fast_mode: 1            # use compute_class_szfast() (emulators)
    use_class_sz_no_cosmo_mode: 1        # SKIP cosmology — fixed via extra_args; sample only profile/HOD/...
    extra_args:
      output: tSZ_1h
      mass_function: T08M500c
      pressure_profile: GNFW
      multipoles: /path/to/ls.txt
      c500: 1.156
      gammaGNFW: 0.3292
      alphaGNFW: 1.062
      z_min: 0.005
      z_max: 3.0
      M_min: 1.0e10
      M_max: 3.5e15
      omega_b: 0.0226           # only when use_class_sz_no_cosmo_mode: 1
      omega_cdm: 0.118
      H0: 68.22
      logA: 3.06
      n_s: 0.9743
likelihood:
  soliket.ymap.ymap_ps.SZLikelihood:
    sz_data_directory: /path/to/data/
    ymap_ps_file: data_ps-ell-y2-erry2_...txt
    ymap_cov_file: cov_..._test.txt
params:
  P0GNFW:  { prior: {min: 0, max: 20}, ref: {dist: norm, loc: 8.13,    scale: 0.1}, proposal: 0.1 }
  betaGNFW:{ prior: {min: 0, max: 10}, ref: {dist: norm, loc: 5.4807,  scale: 0.1}, proposal: 0.1 }
```

The classy_sz cobaya wrapper supports extra observables beyond `Cl_sz`: `sz_binned_cluster_counts`, `sz_unbinned_cluster_counts`. Request them by adding to the likelihood's `get_requirements`.

## SOLikeT SZLikelihood pattern

`soliket.ymap.ymap_ps.SZLikelihood` (subclass of `soliket.gaussian.GaussianLikelihood`) is the canonical y-map bandpower likelihood. Skeleton:

```python
from soliket.gaussian import GaussianLikelihood
import numpy as np, os
from typing import Optional

class SZLikelihood(GaussianLikelihood):
    sz_data_directory: Optional[str] = None
    ymap_ps_file: Optional[str] = None        # ell  y^2  sigma  (3 cols)
    ymap_cov_file: Optional[str] = None       # N×N covariance (N = number of bandpowers)

    def initialize(self):
        D = np.loadtxt(os.path.join(self.sz_data_directory, self.ymap_ps_file))
        self.ell_plc, self.y2AndFg_plc, self.sigma_tot_plc = D[:,0], D[:,1], D[:,2]
        try:
            self.covmat = np.loadtxt(os.path.join(self.sz_data_directory, self.ymap_cov_file))
        except Exception:
            self.covmat = np.diag(self.sigma_tot_plc**2)
        super().initialize()

    def get_requirements(self):
        return {"Cl_sz": {}, "Cl_sz_foreground": {}}

    def _get_data(self):       return self.ell_plc, self.y2AndFg_plc
    def _get_cov(self):        return self.covmat
    def _get_theory(self, **p):
        theory = self.provider.get_Cl_sz()             # {'ell','1h','2h'}
        cl = np.asarray(theory['1h']) + np.asarray(theory['2h'])
        fg = self.provider.get_Cl_sz_foreground()
        return cl + fg if fg is not None else cl
```

For a fast/differentiable variant that bypasses cobaya's theory wiring, see **[`/class-sz:build-likelihood`](../build-likelihood/SKILL.md)** — it scaffolds a Likelihood that calls `cl_yy_from_params` directly.

## Workflow recipes

- **Quick C_ell^yy plot** → use `/class-sz:tszfast` (JAX pipeline, fast)
- **Build a new likelihood for a bandpower dataset** → use `/class-sz:build-likelihood`
- **End-to-end MCMC run on ACT/Planck data** → invoke the `class-sz-engineer` subagent

## Common pitfalls

- `compute()` does the full CLASS Boltzmann solve — slow. Use `compute_class_szfast()` for emulator-mode cosmology, or set `use_class_sz_fast_mode: 1` in the cobaya extra_args
- `use_class_sz_no_cosmo_mode: 1` skips cosmology entirely; the cosmology params must then live in `extra_args`, NOT in the cobaya sampled `params` block
- The `multipoles` extra_arg is a file path — class_sz reads it line-by-line as ell centers. The bandpower data file must use the SAME multipoles
- Trispectrum requires `'output': 'tSZ_tSZ_1h,tSZ_Trispectrum'` (note the leading `tSZ_tSZ_1h` for the 1h diagonal)
- JAX pipeline currently supports only `arnaud10` and `battaglia12`. For GNFW with explicit P0/beta sampling, you can still use `arnaud10` (the parameters are equivalent up to relabeling — see [reference.md](reference.md))
- Emulators are loaded from `~/.classy_szfast/` on first use; reinstall with `classy_szfast.install_emulators()` if missing

## Detailed parameter reference

For full lists of supported `output` strings, all `extra_args` knobs, emulator switches (`cosmo_model`), mass-function options, and the `ProfileParamsA10`/`ProfileParamsB12` field lists, see [reference.md](reference.md).
