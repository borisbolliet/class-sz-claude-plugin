---
description: class_sz / classy_szfast — independent Boltzmann + halo-model theory code extending CLASS. Covers full cosmology (matter Pk, CMB Cls, lensing, H(z)) plus halo-model observables (tSZ Cl^yy, kSZ, CIB, galaxy auto/cross, HOD, cluster counts). Two calculation pipelines (classic Class_sz, JAX cl_yy_from_params), pressure profiles (GNFW / Arnaud / Battaglia), mass functions, cobaya integration via classy_szfast.classy_sz.classy_sz, and the standalone Cl^yy power-spectrum likelihood pattern. Use when writing or debugging class_sz calculations, scaffolding a tSZ bandpower likelihood, or running cobaya MCMCs that fit Cl^yy (or any halo-model observable).
when_to_use: User mentions class_sz, classy_sz, classy_szfast, tSZ / Cl^yy power spectrum, kSZ², CIB, galaxy×lensing, HOD, cluster counts, halo mass function, matter Pk via emulators, Cl_sz, SZLikelihood, GNFW / Arnaud / Battaglia pressure profile, P0GNFW, betaGNFW, cl_yy_from_params, CosmoParams, ProfileParamsA10/B12, ACT / Planck tSZ bandpowers.
allowed-tools: Read Grep Glob Bash(~/pyvenvs/py312-class_sz/bin/python *) Bash(~/pyvenvs/py312-class_sz/bin/cobaya-run *) Bash(~/pyvenvs/py312-class_sz/bin/cobaya-install *) Bash(~/pyvenvs/py312-class_sz/bin/getdist *)
---

# class_sz / classy_szfast assistant

Help with [class_sz](https://github.com/CLASS-SZ/class_sz) (independent Boltzmann + halo-model theory code extending CLASS) and [classy_szfast](https://github.com/CLASS-SZ/classy_szfast) (Python wrapper + CosmoPower emulators + JAX differentiable pipeline).

**class_sz is a full theory code**, not just a halo-model add-on — it can replace CAMB/CLASS in any cobaya run. Capabilities (notebooks in `docs/notebooks/` of the source repo):

- **Cosmology**: matter Pk (linear + nonlinear), CMB Cls (TT/TE/EE), CMB lensing, H(z), σ8, halo mass function. Uses CosmoPower emulators in fast mode for ~ms-level cosmology evaluation.
- **Halo-model observables**: tSZ Cl^yy (1h + 2h + trispectrum), kSZ × tracers, CIB auto/cross, galaxy auto/cross, galaxy×lensing, tSZ×lensing, HOD, cluster counts (binned and unbinned).
- **Differentiable**: `classy_szfast.differentiable.cl_yy_from_params` is a fully JAX-jittable / grad-able Cl^yy pipeline (~200 evals/s on CPU). More observables being added.

**Local venv:** `~/pyvenvs/py312-class_sz/bin/python` has classy_sz, classy_szfast, cobaya, getdist, jax. Always invoke that python when running examples. `soliket` may or may not be installed; prefer standalone likelihoods to avoid the dependency.

## Three calculation pipelines — pick by use case

| Pipeline | Module | When to use |
| --- | --- | --- |
| **`classy_szlite`** ⭐ PREFERRED | `import classy_szlite as csl` | Pure-JAX, minimal deps (jax + numpy + mcfit), ede-v2 default. Covers CMB Cls, Pk linear/nonlinear, distances, derived params, halo-model Cl^yy. `cl_yy_factory` gives **~5 ms/eval** for fixed-cosmology MCMC. Use this for any new tSZ work. Repo: https://github.com/CLASS-SZ/classy_szlite |
| **Classic** | `from classy_sz import Class as Class_sz` | Full halo-model surface for observables not in classy_szlite (cluster counts, kSZ², CIB cross-spectra, etc.). Production cobaya runs via `classy_szfast.classy_sz.classy_sz`. |
| **`classy_szfast.differentiable`** | `from classy_szfast.differentiable import cl_yy_from_params` | Older JAX path with broader cosmo_model support (lcdm/mnu/neff/wcdm/ede/ede-v2). Use only if classy_szlite doesn't support what you need. |

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

## Canonical Cl^yy power-spectrum likelihood — standalone (no SOLikeT)

For fitting tSZ Cl^yy **bandpower** data (a power spectrum measurement; not a y-map pixel likelihood, despite the historical `ymap_ps.py` filename in SOLikeT), the cleanest pattern is a `cobaya.likelihood.Likelihood` subclass that loads bandpowers + covariance directly and computes the Gaussian `logp` itself. No SOLikeT dependency, no inheritance chain, ~50 lines. This is what `/class-sz:build-likelihood` scaffolds by default.

```python
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory
import numpy as np, os
from typing import Optional

class SZLikelihood(Likelihood):
    sz_data_directory: Optional[str] = None
    ymap_ps_file:      Optional[str] = None    # 3 cols: ell, D_ell × 1e12, σ
    ymap_cov_file:     Optional[str] = None    # N×N covariance

    def initialize(self):
        D = np.loadtxt(os.path.join(self.sz_data_directory, self.ymap_ps_file))
        self.ell, self.y, self.sigma = D[:,0], D[:,1], D[:,2]
        if self.ymap_cov_file:
            self.cov = np.loadtxt(os.path.join(self.sz_data_directory, self.ymap_cov_file))
        else:
            self.cov = np.diag(self.sigma**2)
        self.inv_cov = np.linalg.inv(self.cov)
        sign, logdet = np.linalg.slogdet(self.cov)
        self.log_norm = -0.5*logdet - 0.5*len(self.y)*np.log(2*np.pi)

    def get_requirements(self):
        return {"Cl_sz": {}, "Cl_sz_foreground": {}}

    def logp(self, **p):
        t = self.provider.get_Cl_sz()                         # {'ell','1h','2h'}
        cl = np.asarray(t['1h']) + np.asarray(t['2h'])
        fg = self.provider.get_Cl_sz_foreground()
        if fg is not None: cl = cl + np.asarray(fg)
        r = self.y - cl
        return -0.5 * float(r @ self.inv_cov @ r) + self.log_norm


class SZForegroundTheory(Theory):
    params = {"A_CIB": 0.0, "A_RS": 0.0, "A_IR": 0.0}
    foreground_data_directory: Optional[str] = None
    foreground_data_file: Optional[str] = "data_fg-ell-cib_rs_ir_cn-total-planck-collab-15.txt"

    def initialize(self):
        D = np.loadtxt(os.path.join(self.foreground_data_directory, self.foreground_data_file))
        self.A_CIB_MODEL, self.A_RS_MODEL = D[:,1], D[:,2]
        self.A_IR_MODEL,  self.A_CN_MODEL = D[:,3], D[:,4]

    def calculate(self, state, want_derived=False, **p):
        A_CN = 0.9033                                        # Bolliet+18 (1712.00788)
        if p["A_CIB"]==0 and p["A_RS"]==0 and p["A_IR"]==0:
            state["Cl_sz_foreground"] = None
        else:
            state["Cl_sz_foreground"] = (p["A_CIB"]*self.A_CIB_MODEL +
                p["A_RS"]*self.A_RS_MODEL + p["A_IR"]*self.A_IR_MODEL + A_CN*self.A_CN_MODEL)

    def get_Cl_sz_foreground(self):
        return self._current_state["Cl_sz_foreground"]
```

Run cobaya-run from the workdir so the module is on `sys.path`. The YAML references the bare module name (`mymod.SZLikelihood`), not `soliket.ymap.…`.

For a differentiable variant that bypasses the cobaya theory wiring entirely, see **[`/class-sz:build-likelihood --jax`](../build-likelihood/SKILL.md)** — it calls `cl_yy_from_params` directly inside `logp`.

### Legacy: SOLikeT inheritance pattern

If you're reproducing a chain that already references `soliket.ymap.ymap_ps.SZLikelihood` (and you have `soliket` installed), you can keep the inheritance form:

```python
from soliket.gaussian import GaussianLikelihood
class SZLikelihood(GaussianLikelihood):
    # ... same fields ...
    def _get_data(self):  return self.ell, self.y
    def _get_cov(self):   return self.covmat
    def _get_theory(self, **p):
        t = self.provider.get_Cl_sz()
        cl = np.asarray(t['1h']) + np.asarray(t['2h'])
        fg = self.provider.get_Cl_sz_foreground()
        return cl + np.asarray(fg) if fg is not None else cl
```

Prefer standalone for new work — fewer dependencies, no version skew, no SOLikeT install footguns.

## Workdir convention

Self-contained layout for a tSZ fit:

```
<workdir>/
├── ymap_ps.py             # the standalone likelihood module (or whatever name you pick)
├── <run-name>.yaml        # cobaya input
├── data/                  # bandpowers, cov, multipoles, foreground template
└── chains/                # cobaya output
```

Always `cd` into `<workdir>` before invoking cobaya-run, so the likelihood module is importable. The `/class-sz:build-likelihood` skill produces exactly this layout.

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
