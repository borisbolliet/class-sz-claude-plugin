---
description: classy_szlite — pure-JAX cosmology code (CMB Cls, matter Pk linear/nonlinear, distances, derived params, halo-model tSZ Cl^yy). Backed by the v2 CosmoPower emulators (same as ACT DR6 extended-cosmology analyses). Differentiable via jax.grad, JIT-friendly, and built around a fast-path `cl_yy_factory` closure for fixed-cosmology MCMC over profile parameters. Use when writing or debugging classy_szlite calculations, scaffolding a tSZ Cl^yy bandpower likelihood, running cobaya MCMCs, or using NUTS / HMC samplers on the GNFW pressure profile.
when_to_use: User mentions classy_szlite, csl, CosmoParams, ProfileParamsA10, cl_yy, cl_yy_factory, cl_TTTEEE, Pk, Pnl, distances, derived, halo-model integrals, GNFW / Arnaud pressure profile, P0GNFW, betaGNFW, tSZ Cl^yy bandpower likelihood, ACT-DR6 tSZ, may26 cobaya fit, NUTS / HMC over profile parameters, JAX gradient probes through cl_yy, cobaya Theory wiring via classy_szlite.
allowed-tools: Read Grep Glob Bash(~/pyvenvs/py312-class_sz/bin/python *) Bash(~/pyvenvs/py312-class_sz/bin/cobaya-run *) Bash(~/pyvenvs/py312-class_sz/bin/cobaya-install *) Bash(~/pyvenvs/py312-class_sz/bin/getdist *)
---

# classy_szlite assistant

Help with [classy_szlite](https://github.com/CLASS-SZ/classy_szlite) — a
pure-JAX cosmology code backed by the high-accuracy `v2` CosmoPower
emulators (same emulators as the [ACT DR6 extended-cosmology
analysis](https://arxiv.org/abs/2503.14454) and the [ACT DR6 + DESI DR2
EDE / H_0 analysis by Poulin et al.](https://arxiv.org/abs/2505.08051)).

**Runtime deps:** `jax + numpy + mcfit`. No tensorflow, no keras, no
cosmopower at runtime (the `_v2_plain.npz` emulator files are pure-numpy
loadable). Forward pass + gradients are pure JAX.

**Local venv:** `~/pyvenvs/py312-class_sz/bin/python` has classy_szlite,
cobaya, numpyro, getdist, jax. Always invoke that python when running
examples.

## API surface (everything top-level on `classy_szlite`)

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import classy_szlite as csl

# --- parameter containers (NamedTuple pytrees) ---
cosmo   = csl.CosmoParams()                                # Planck-18 defaults
profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)

# --- cosmology calculations ---
csl.derived(cosmo)                # {'sigma_8', 'Omega_m', 'S8', 'der_full': ndarray(17,)}
csl.cl_TTTEEE(cosmo)              # {'ell', 'tt', 'te', 'ee'}  dimensionless D_ell
csl.Pk(cosmo, [0., 0.5, 1., 2.])  # (k, P_lin(z, k)) — k in h/Mpc, P in (Mpc/h)³
csl.Pnl(cosmo, [0., 0.5])         # (k, P_nl(z, k))  — HMcode
csl.distances(cosmo, [0.1, 1.0])  # (Hz/c [1/Mpc], chi [Mpc], Da [Mpc])

# --- halo-model tSZ Cl^yy ---
ell = jnp.geomspace(2, 9000, 80)
cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell)              # full pipeline, ~18 ms warm

# --- FAST PATH for MCMC over profile, fixed cosmology ---
ev = csl.cl_yy_factory(cosmo, ell)                         # one-shot CosmoGrids + HaloGrids
cl_1h, cl_2h = ev(profile)                                 # ~5 ms / call
```

All functions are JAX-traceable; you can `jax.grad`, `jax.jacfwd`,
`jax.vmap` any of them. The `CosmoParams` / `ProfileParamsA10` are JAX
pytrees so gradients return the same container type.

## When to use which entry point

| Task | Function | Per-eval cost |
| --- | --- | --- |
| Fixed-cosmology MCMC over profile params (the dominant tSZ use case) | `cl_yy_factory(cosmo, ell)(profile)` | **~5 ms** |
| One-off Cl^yy at arbitrary cosmology | `cl_yy(cosmo, profile, ell)` | ~18 ms |
| Fisher matrix w.r.t. profile | `jax.jacfwd(ev)(profile)` | ~20 ms |
| Fisher / NUTS over cosmology too | full pipeline + `jax.grad` | ~50 ms |
| CMB-only, Pk-only, distances-only | the dedicated function | 1–3 ms |

**Don't wrap `cl_yy_factory(...)` in `jax.jit`** — its internals call
`mcfit.TophatVar` (σ(R) over a fixed log-k grid) which isn't jit-safe.
The closure is already fast and `jax.grad` works through it directly.

## Pressure profile

Currently classy_szlite ships **Arnaud 2010 GNFW** only. Container
fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `P0`    | 8.130  | central pressure normalisation |
| `c500`  | 1.156  | concentration at r500c |
| `gamma` | 0.3292 | inner slope |
| `alpha` | 1.062  | transition slope |
| `beta`  | 5.4807 | outer slope |
| `B`     | 1.25   | hydrostatic mass bias (M_true = M_obs / B) |

In the cobaya YAML the canonical sampled names are `P0GNFW`,
`betaGNFW`, etc. (the cobaya conventions); the scaffold maps them onto
the JAX field names.

## Cosmology container

`CosmoParams` fields and defaults (Planck-18 ΛCDM at the
LCDM-equivalent point of the `v2` ede emulators):

| Field | Default | Notes |
| --- | --- | --- |
| `omega_b`    | 0.02242  | physical baryon density |
| `omega_cdm`  | 0.11933  | physical CDM density |
| `H0`         | 67.66    | Hubble constant |
| `tau_reio`   | 0.054    | reionisation optical depth |
| `ln10_10_As` | 3.047    | log primordial amplitude |
| `n_s`        | 0.9665   | scalar tilt |
| `m_ncdm`     | 0.02 eV  | per-species neutrino mass (3 deg ν → Σmν = 0.06 eV) |
| `N_ur`       | 0.00441  | ultra-relativistic species |
| `fEDE`       | 0.001    | EDE fraction (silently LCDM-equivalent at this value) |
| `log10z_c`   | 3.562    | EDE critical redshift |
| `thetai_scf` | 2.83     | initial scalar-field angle |
| `r`          | 0.0      | tensor-to-scalar ratio |

For ΛCDM work, override only the standard 6 (`omega_b, omega_cdm, H0,
tau_reio, ln10_10_As, n_s`) — the rest stay at the LCDM-equivalent
defaults silently. For EDE / w-CDM / ν-CDM / Neff-CDM exploration, set
the relevant extra fields explicitly (the `v2` emulator covers all of
them).

## Halo-model Cl^yy convergence

Default integration grids (`n_z=100, n_m=200, m_min=1e10, m_max=3.5e15`)
give max |ΔC/C| ≤ 10⁻³ across ℓ ∈ [100, 8000] against an `n_z=400,
n_m=600` reference. Full sweep in the
[convergence study](https://classy-szlite.readthedocs.io/en/latest/convergence.html).

| Use case | n_z | n_m | target accuracy |
| --- | --- | --- | --- |
| MCMC / production (default) | 100 | 200 | ≤ 10⁻³ |
| Forecast / pre-MCMC         |  50 | 100 | ≤ 3×10⁻³ |
| Reference / cross-check     | 300 | 200 | ≤ 10⁻⁴ |
| Quick smoke-test            |  25 |  50 | ~1% |

## Canonical Cl^yy bandpower likelihood (standalone, no SOLikeT)

For fitting tSZ Cl^yy **bandpower** data (a power-spectrum
measurement; not a y-map pixel likelihood, despite the historical
`ymap_ps.py` filename in some legacy code), the cleanest pattern is a
`cobaya.likelihood.Likelihood` subclass that loads bandpowers +
covariance directly and computes the Gaussian `logp` itself. This is
what `/class-sz:build-likelihood` scaffolds by default.

```python
from cobaya.likelihood import Likelihood
from cobaya.theory     import Theory
import numpy as np, os
from typing import Optional
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import classy_szlite as csl


class ClyyLikelihood(Likelihood):
    sz_data_directory: Optional[str] = None
    ymap_ps_file:      Optional[str] = None    # 3 cols: ell, D_ell × 1e12, σ
    ymap_cov_file:     Optional[str] = None    # N × N covariance

    def initialize(self):
        D = np.loadtxt(os.path.join(self.sz_data_directory, self.ymap_ps_file))
        self.ell, self.y, self.sigma = D[:, 0], D[:, 1], D[:, 2]
        self.cov = np.loadtxt(os.path.join(self.sz_data_directory, self.ymap_cov_file)) \
                   if self.ymap_cov_file else np.diag(self.sigma**2)
        self.inv_cov = np.linalg.inv(self.cov)
        sign, logdet = np.linalg.slogdet(self.cov)
        self.log_norm = -0.5 * logdet - 0.5 * len(self.y) * np.log(2*np.pi)

    def get_requirements(self):
        return {"Cl_sz": {}, "Cl_sz_foreground": {}}

    def logp(self, **p):
        t  = self.provider.get_Cl_sz()                                  # {'ell','1h','2h'}
        cl = np.asarray(t['1h']) + np.asarray(t['2h'])
        fg = self.provider.get_Cl_sz_foreground()
        if fg is not None:
            cl = cl + np.asarray(fg)
        r = self.y - cl
        return -0.5 * float(r @ self.inv_cov @ r) + self.log_norm


class ClyyTheory(Theory):
    """classy_szlite-backed Cl_sz provider with cl_yy_factory fast path."""
    multipoles_file: Optional[str] = None
    n_z:        int   = 100
    n_m:        int   = 200
    delta_crit: float = 500.0

    # Standard 6 cosmology — fixed for this Theory
    omega_b:    float = 0.0226
    omega_cdm:  float = 0.118
    H0:         float = 68.22
    tau_reio:   float = 0.0561
    ln10_10_As: float = 3.06
    n_s:        float = 0.9743

    params = {"P0GNFW": 8.13, "c500": 1.156, "gammaGNFW": 0.3292,
              "alphaGNFW": 1.062, "betaGNFW": 5.48, "B": 1.25}

    def initialize(self):
        ell_array = np.loadtxt(self.multipoles_file)
        self._ell = jnp.asarray(ell_array)
        self.ell_np = ell_array
        self._dl_factor = jnp.asarray(ell_array * (ell_array+1) / (2*np.pi) * 1e12)

        cosmo = csl.CosmoParams(
            omega_b=self.omega_b, omega_cdm=self.omega_cdm,
            H0=self.H0, tau_reio=self.tau_reio,
            ln10_10_As=self.ln10_10_As, n_s=self.n_s,
        )
        self._csl  = csl
        self._eval = csl.cl_yy_factory(
            cosmo, self._ell,
            n_z=self.n_z, n_m=self.n_m, delta_crit=self.delta_crit,
        )

    def get_can_provide(self): return ["Cl_sz"]

    def calculate(self, state, want_derived=True, **p):
        prof = self._csl.ProfileParamsA10(
            P0=p["P0GNFW"], c500=p["c500"], gamma=p["gammaGNFW"],
            alpha=p["alphaGNFW"], beta=p["betaGNFW"], B=p["B"],
        )
        cl1, cl2 = self._eval(prof)
        state["Cl_sz"] = {
            "ell": self.ell_np,
            "1h":  np.asarray(self._dl_factor * cl1),
            "2h":  np.asarray(self._dl_factor * cl2),
        }

    def get_Cl_sz(self):
        return self._current_state["Cl_sz"]


class ClyyForegroundTheory(Theory):
    """CIB / RS / IR / CN foreground templates with amplitude knobs."""
    params = {"A_CIB": 0.0, "A_RS": 0.0, "A_IR": 0.0}
    foreground_data_directory: Optional[str] = None
    foreground_data_file: Optional[str] = "data_fg-ell-cib_rs_ir_cn-total-planck-collab-15.txt"

    def initialize(self):
        D = np.loadtxt(os.path.join(self.foreground_data_directory, self.foreground_data_file))
        self.A_CIB_MODEL, self.A_RS_MODEL = D[:,1], D[:,2]
        self.A_IR_MODEL,  self.A_CN_MODEL = D[:,3], D[:,4]

    def calculate(self, state, want_derived=False, **p):
        A_CN = 0.9033                                       # Bolliet+18 (1712.00788)
        if p["A_CIB"]==0 and p["A_RS"]==0 and p["A_IR"]==0:
            state["Cl_sz_foreground"] = None
        else:
            state["Cl_sz_foreground"] = (p["A_CIB"]*self.A_CIB_MODEL +
                p["A_RS"]*self.A_RS_MODEL + p["A_IR"]*self.A_IR_MODEL +
                A_CN*self.A_CN_MODEL)

    def get_Cl_sz_foreground(self):
        return self._current_state["Cl_sz_foreground"]
```

Always run `cobaya-run` from the **workdir** (where the likelihood
module lives), so the bare module name resolves on `sys.path`.

## Gradient-based sampling (NUTS / HMC)

Because the forward pass is pure JAX, gradient-based samplers
(`numpyro` NUTS, `blackjax`, `flowMC`) work out of the box. For a
fixed-cosmology profile-only fit:

```python
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

ev = csl.cl_yy_factory(cosmo, jnp.asarray(ell))
dl_factor = jnp.asarray(ell * (ell + 1) / (2 * np.pi) * 1e12)

def model():
    P0   = numpyro.sample("P0",   dist.Uniform(0.0, 20.0))
    beta = numpyro.sample("beta", dist.Uniform(0.0, 10.0))
    prof = csl.ProfileParamsA10(P0=P0, c500=1.156, gamma=0.3292,
                                 alpha=1.062, beta=beta, B=1.25)
    cl1, cl2 = ev(prof)
    mu = dl_factor * (cl1 + cl2)
    numpyro.factor("loglike", -0.5 * (y - mu) @ inv_cov @ (y - mu))

mcmc = MCMC(NUTS(model, dense_mass=True),
            num_warmup=500, num_samples=2000, num_chains=4)
mcmc.run(jax.random.PRNGKey(0))
mcmc.print_summary()
```

A full runnable example (with corner plot + cobaya-MH overlay) is at
[`examples/nuts_clyy_profile.py`](https://github.com/CLASS-SZ/classy_szlite/blob/main/examples/nuts_clyy_profile.py)
in the classy_szlite repo. Typical numbers on a single-thread laptop:
8000 samples × 4 chains in ~64 s, R-hat = 1.01, zero divergences.

## Workdir convention

Self-contained layout for a tSZ fit:

```
<workdir>/
├── <run-name>.py             # standalone likelihood + Theory module
├── <run-name>.yaml           # cobaya input
├── data/                     # bandpowers, cov, multipoles, foreground template
└── chains/                   # cobaya output
```

`cd` into `<workdir>` before invoking `cobaya-run` so the likelihood
module is importable. The `/class-sz:build-likelihood` skill produces
exactly this layout.

## Workflow recipes

- **Quick Cl^yy plot** → use `/class-sz:tszfast`
- **Build a new likelihood for a bandpower dataset** → use `/class-sz:build-likelihood`
- **End-to-end MCMC (cobaya RW-MH or NumPyro NUTS)** → invoke the
  `class-sz-engineer` subagent

## Common pitfalls

- **CWD footgun**: don't run `cobaya-run` / python from `~/GitHub` —
  PEP 420 namespace-package resolution can shadow editable installs
  via the `cobaya/` clone subfolder. Always `cd <workdir>` first.
- **Multipoles must match**: the `multipoles_file` passed to the Theory
  must use the SAME ell centres as the bandpower data file. Mismatch
  is silent.
- **JAX 64-bit**: `jax.config.update("jax_enable_x64", True)` is set on
  classy_szlite import — cosmology likelihoods need double precision;
  single precision will give noticeably biased posteriors at the
  bandpower covariance level.
- **Don't `jax.jit` the factory closure** — internals call
  `mcfit.TophatVar` which is not jit-safe. The closure is already fast.
- **`m_ncdm` is per-species** (3 degenerate ν by convention), so
  `Σmν = 3 × m_ncdm`. The `derived()` function uses this convention
  for `Omega_m`.

## Detailed parameter reference

For full field lists, emulator file conventions, and the LCDM /
m_ν / w-CDM / N_eff / EDE coverage details, see
[reference.md](reference.md).
