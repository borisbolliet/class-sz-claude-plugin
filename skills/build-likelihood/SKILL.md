---
description: Scaffold a cobaya likelihood for a tSZ Cl^yy power-spectrum bandpower dataset using class_sz. Produces a Python likelihood module + working cobaya YAML ready to MCMC. Defaults to a standalone implementation (no SOLikeT dependency); --soliket and --jax flags switch to alternative styles.
disable-model-invocation: true
argument-hint: "<likelihood-name> [--soliket | --jax] [--data-dir DIR]"
allowed-tools: Read Write Edit Glob Grep Bash(~/pyvenvs/py312-class_sz/bin/python *) Bash(~/pyvenvs/py312-class_sz/bin/cobaya-run -t *) Bash(~/pyvenvs/py312-class_sz/bin/cobaya-run --test *)
---

# Build a cobaya likelihood for a Cl^yy power-spectrum bandpower dataset

Scaffold a Gaussian likelihood + foreground theory module and a working cobaya YAML for a **tSZ Cl^yy power-spectrum bandpower dataset** (the data is binned bandpowers + an N×N covariance, NOT y-map pixels — the historical SOLikeT name `ymap_ps.py` is misleading; "ps" already means power spectrum, the `ymap` prefix is legacy carryover). Output: a Python file and a YAML, both pointing at a local workdir.

Arguments in `$ARGUMENTS`:
- `$0` — likelihood class name in CamelCase (e.g. `ACTYMapLikelihood`)
- Optional flag (mutually exclusive):
  - **(default) standalone** — `cobaya.likelihood.Likelihood` + `cobaya.theory.Theory` directly. No SOLikeT dep. Simplest, most portable, what you want for a fresh workdir. Matches the validated `~/Desktop/class-sz-plugin-tests/ymap_ps.py` shape.
  - `--soliket` — legacy `soliket.gaussian.GaussianLikelihood` inheritance pattern. Use only when you need other SOLikeT features (cash, sacc, ccl), or when reproducing a chain that explicitly references `soliket.ymap.ymap_ps.SZLikelihood`.
  - `--jax` — `cobaya.likelihood.Likelihood` that calls `classy_szfast.differentiable.cl_yy_from_params` directly inside `logp`. Skips the class_sz theory wrapper entirely. Faster and differentiable, but A10/B12 profiles only and currently no `Cl_sz_foreground` plumbing.
- `--data-dir DIR` — directory with the bandpower / cov / multipoles / foreground files (otherwise ask the user).

## Workdir convention

Default target: a self-contained workdir laid out as

```
<workdir>/
├── <likelihood-name>.py          # the scaffolded module (this skill writes it)
├── <likelihood-name>.yaml        # the scaffolded cobaya input (also written)
├── data/                         # bandpowers / cov / multipoles / foreground template
└── chains/                       # cobaya output
```

If `<workdir>` already has a `data/` subdir with the expected files, use them. Otherwise ask the user where the data lives and propose copying it into `data/`.

**Always run cobaya-run from the workdir** so the scaffolded Python module is on `sys.path`. The skill prints the exact command.

## Steps

1. **Read the data.** `np.loadtxt` the bandpower file (3-col: ell, D_ell·1e12, σ); check its shape matches the cov file (N×N) and the multipoles file (N rows). Mismatched ell centers between data and multipoles is a silent bug — verify.

2. **Ask** for any missing details:
   - data directory if not given via `--data-dir`
   - bandpower / cov / multipoles / foreground file names (offer to glob)
   - pressure profile: `GNFW` (default) or `B12`
   - sampled params (default: `P0GNFW`, `betaGNFW`; rest fixed)
   - cosmology fixed (default, `use_class_sz_no_cosmo_mode: 1`) or sampled
   - include the foreground theory? (default: yes, with all amplitudes fixed at 0 — file must still exist)

3. **Write the likelihood module** to `<workdir>/<likelihood-name>.py`. Pick skeleton by flag:

### Default — standalone

```python
"""<Likelihood-name> — standalone Gaussian likelihood for a tSZ bandpower dataset."""
from __future__ import annotations
import os, numpy as np
from typing import Optional
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory


class <LikelihoodName>(Likelihood):
    sz_data_directory: Optional[str] = None
    ymap_ps_file:      Optional[str] = None
    ymap_cov_file:     Optional[str] = None

    def initialize(self):
        D = np.loadtxt(os.path.join(self.sz_data_directory, self.ymap_ps_file))
        self.ell, self.y, self.sigma = D[:,0], D[:,1], D[:,2]
        if self.ymap_cov_file:
            cov = np.loadtxt(os.path.join(self.sz_data_directory, self.ymap_cov_file))
            assert cov.shape == (len(self.y), len(self.y))
            self.cov = cov
        else:
            self.cov = np.diag(self.sigma ** 2)
        self.inv_cov = np.linalg.inv(self.cov)
        sign, logdet = np.linalg.slogdet(self.cov)
        if sign <= 0:
            raise ValueError(f"Covariance not positive-definite (sign={sign})")
        self.log_norm = -0.5 * logdet - 0.5 * len(self.y) * np.log(2*np.pi)
        self.log.info(f"Loaded {len(self.y)} bandpowers from {self.ymap_ps_file}")

    def get_requirements(self):
        return {"Cl_sz": {}, "Cl_sz_foreground": {}}

    def logp(self, **p):
        t = self.provider.get_Cl_sz()                       # {'ell','1h','2h'}
        cl = np.asarray(t['1h']) + np.asarray(t['2h'])
        fg = self.provider.get_Cl_sz_foreground()
        if fg is not None:
            cl = cl + np.asarray(fg)
        r = self.y - cl
        return -0.5 * float(r @ self.inv_cov @ r) + self.log_norm


class <LikelihoodName>ForegroundTheory(Theory):
    params = {"A_CIB": 0.0, "A_RS": 0.0, "A_IR": 0.0}
    foreground_data_directory: Optional[str] = None
    foreground_data_file: Optional[str] = "data_fg-ell-cib_rs_ir_cn-total-planck-collab-15.txt"

    def initialize(self):
        D = np.loadtxt(os.path.join(self.foreground_data_directory, self.foreground_data_file))
        self.A_CIB_MODEL, self.A_RS_MODEL = D[:,1], D[:,2]
        self.A_IR_MODEL,  self.A_CN_MODEL = D[:,3], D[:,4]

    def calculate(self, state, want_derived=False, **p):
        A_CIB, A_RS, A_IR = p["A_CIB"], p["A_RS"], p["A_IR"]
        A_CN = 0.9033                                       # Bolliet+18 arXiv:1712.00788
        if A_CIB == 0 and A_RS == 0 and A_IR == 0:
            state["Cl_sz_foreground"] = None
        else:
            state["Cl_sz_foreground"] = (
                A_CIB*self.A_CIB_MODEL + A_RS*self.A_RS_MODEL +
                A_IR*self.A_IR_MODEL + A_CN*self.A_CN_MODEL)

    def get_Cl_sz_foreground(self):
        return self._current_state["Cl_sz_foreground"]
```

### `--soliket` — legacy

```python
from soliket.gaussian import GaussianLikelihood
import numpy as np, os
from typing import Optional

class <LikelihoodName>(GaussianLikelihood):
    sz_data_directory: Optional[str] = None
    ymap_ps_file:      Optional[str] = None
    ymap_cov_file:     Optional[str] = None

    def initialize(self):
        D = np.loadtxt(os.path.join(self.sz_data_directory, self.ymap_ps_file))
        self.ell, self.y, self.sigma = D[:,0], D[:,1], D[:,2]
        try:    self.covmat = np.loadtxt(os.path.join(self.sz_data_directory, self.ymap_cov_file))
        except: self.covmat = np.diag(self.sigma**2)
        super().initialize()

    def get_requirements(self): return {"Cl_sz": {}, "Cl_sz_foreground": {}}
    def _get_data(self): return self.ell, self.y
    def _get_cov(self):  return self.covmat
    def _get_theory(self, **p):
        t = self.provider.get_Cl_sz()
        cl = np.asarray(t['1h']) + np.asarray(t['2h'])
        fg = self.provider.get_Cl_sz_foreground()
        return cl + np.asarray(fg) if fg is not None else cl
```

Caveats for `--soliket`: requires `soliket` install in the active venv. Be aware of the `~/GitHub` cwd-collision footgun (Python's namespace-package machinery shadows the editable cobaya/soliket installs if a folder with the same name is in cwd). Run from the workdir, not from `~/GitHub`.

### `--jax` — local JAX-backed Theory provider (no class_sz wrapper)

Three classes: the same standalone Likelihood + ForegroundTheory as the default path, PLUS a local `<Name>Theory` that wraps `classy_szfast.differentiable.cl_yy_from_params` and exposes `Cl_sz` (drop-in replacement for `classy_szfast.classy_sz.classy_sz`). This is the pattern validated in `~/Desktop/class-sz-plugin-tests/clyy_v2.py:ClyyTheoryV2`: ~12 ms / warm eval at the bandpower ells, vs ~58 ms for the classy_szfast.classy_sz wrapper — about 4× faster per MCMC step.

```python
import os, numpy as np
from typing import Optional
from cobaya.theory import Theory

class <LikelihoodName>Theory(Theory):
    """JAX-backed cobaya Theory provider for tSZ Cl^yy. Drop-in for
    classy_szfast.classy_sz.classy_sz when only Cl_sz is needed.
    Returns D_ell × 1e12 (matches bandpower-data convention)."""

    profile:    str   = "arnaud10"         # or "battaglia12" (not wired here)
    delta_crit: float = 500.0              # 500 for arnaud10, 200 for battaglia12
    n_z:        int   = 100
    n_m:        int   = 200
    multipoles_file: Optional[str] = None  # must match the bandpower ell column

    # Fixed cosmology — override in YAML
    omega_b: float = 0.0226
    omega_cdm: float = 0.118
    H0: float = 68.22
    tau_reio: float = 0.07
    ln10_10_As: float = 3.06
    n_s: float = 0.9743

    # All 5 Arnaud-10 profile fields exposed as cobaya params; fix some,
    # sample others by setting priors in YAML.
    params = {
        "P0GNFW":    8.130,
        "c500":      1.156,
        "gammaGNFW": 0.3292,
        "alphaGNFW": 1.062,
        "betaGNFW":  5.4807,
    }

    def initialize(self):
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from classy_szfast.differentiable import (
            CosmoParams, ProfileParamsA10, cl_yy_from_params,
        )
        ell_array = np.loadtxt(self.multipoles_file)
        self._ell = jnp.asarray(ell_array)
        self.ell_np = ell_array
        # Convert dimensionless C_ell to D_ell × 1e12 (bandpower-data units)
        self._dl_factor = jnp.asarray(ell_array * (ell_array + 1) / (2*np.pi) * 1e12)
        self._cosmo = CosmoParams(self.omega_b, self.omega_cdm, self.H0,
                                  self.tau_reio, self.ln10_10_As, self.n_s)

        # IMPORTANT: do NOT wrap _fwd in jax.jit. classy_szfast's CosmoPower
        # emulators handle their own JIT internally; an outer jit propagates
        # the trace into emulator lazy-warmup paths that aren't fully trace-
        # safe. The ultrafast notebook follows the same no-outer-jit pattern.
        # See [[project-classy-szfast-no-outer-jit]] (project memory) and
        # the upstream fix at https://github.com/CLASS-SZ/classy_szfast
        # commit 9b4a0d1 (removes one obstacle; others remain).
        def _fwd(P0, c500, gamma, alpha, beta):
            prof = ProfileParamsA10(P0=P0, c500=c500, gamma=gamma,
                                    alpha=alpha, beta=beta)
            cl1, cl2 = cl_yy_from_params(self._ell, self._cosmo,
                profile_params=prof, profile=self.profile,
                delta_crit=self.delta_crit, n_z=self.n_z, n_m=self.n_m)
            return self._dl_factor * cl1, self._dl_factor * cl2
        self._fn = _fwd

    def get_can_provide(self): return ["Cl_sz"]

    def calculate(self, state, want_derived=True, **p):
        dl1, dl2 = self._fn(p["P0GNFW"], p["c500"],
                            p["gammaGNFW"], p["alphaGNFW"], p["betaGNFW"])
        state["Cl_sz"] = {"ell": self.ell_np,
                          "1h": np.asarray(dl1), "2h": np.asarray(dl2)}

    def get_Cl_sz(self):
        return self._current_state["Cl_sz"]
```

The Likelihood and ForegroundTheory are the same as the default (standalone) path. The YAML changes: replace the `classy_szfast.classy_sz.classy_sz` block with `<likelihood-module>.<LikelihoodName>Theory`, and fix the cosmology + halo-model grid sizes as theory attrs instead of `extra_args`.

Caveats:
- The multipoles file passed to the theory MUST match the data's ell column (the JAX pipeline evaluates exactly at those ells; no rebinning).
- Cold first call is ~3–4 s (emulator JIT compile). All subsequent calls are ~12 ms — fine for MCMC.
- Currently `arnaud10` only; Battaglia12 needs `ProfileParamsB12` + the 9 B12 fields.

4. **Write the cobaya YAML** to `<workdir>/<likelihood-name>.yaml`. The default (standalone) shape (adapt module name and data files; halo-fit style, fixed cosmology, sampling profile params):

```yaml
theory:
  <LikelihoodName>ForegroundTheory:
    foreground_data_directory: <ABSOLUTE WORKDIR>/data/
    foreground_data_file: data_fg-ell-cib_rs_ir_cn-total-planck-collab-15.txt
  classy_szfast.classy_sz.classy_sz:
    use_class_sz_fast_mode: 1
    use_class_sz_no_cosmo_mode: 1
    extra_args:
      output: tSZ_1h
      mass_function: T08M500c
      pressure_profile: GNFW
      multipoles: <ABSOLUTE WORKDIR>/data/<ls-file>.txt
      c500: 1.156
      gammaGNFW: 0.3292
      alphaGNFW: 1.062
      z_min: 0.005
      z_max: 3.0
      M_min: 1.0e10
      M_max: 3.5e15
      x_outSZ: 4.0
      use_fft_for_profiles_transform: 1
      N_samp_fftw: 1024
      x_min_gas_pressure_fftw: 1.0e-4
      x_max_gas_pressure_fftw: 1.0e6
      omega_b: 0.0226
      omega_cdm: 0.118
      H0: 68.22
      logA: 3.06
      n_s: 0.9743
likelihood:
  <likelihood-module>.<LikelihoodName>:
    sz_data_directory: <ABSOLUTE WORKDIR>/data/
    ymap_ps_file:      <bandpower-file>.txt
    ymap_cov_file:     <cov-file>.txt
params:
  A_CIB: 0
  A_RS:  0
  A_IR:  0
  P0GNFW:   { prior: {min: 0, max: 20}, ref: {dist: norm, loc: 8.13,   scale: 0.1}, proposal: 0.1 }
  betaGNFW: { prior: {min: 0, max: 10}, ref: {dist: norm, loc: 5.4807, scale: 0.1}, proposal: 0.1 }
sampler:
  mcmc:
    Rminus1_stop: 0.05            # 0.01 for production
    max_tries: 10000
    burn_in: 100
    learn_proposal: true
    blocking:
      - [1, [P0GNFW, betaGNFW]]
    proposal_scale: 1.9
    oversample_power: 0.4
    oversample_thin: true
output: <ABSOLUTE WORKDIR>/chains/<likelihood-name>_test
debug: True
timing: true
```

For `--soliket`: replace the standalone module reference with `soliket.ymap.ymap_ps.SZLikelihood` (or your `--soliket` scaffold's class) and the foreground theory similarly. For `--jax`: replace the `classy_szfast.classy_sz.classy_sz` block with the local JAX theory provider:

```yaml
theory:
  <likelihood-module>.<LikelihoodName>ForegroundTheory:
    foreground_data_directory: <ABSOLUTE WORKDIR>/data/
    foreground_data_file: data_fg-ell-cib_rs_ir_cn-total-planck-collab-15.txt
  <likelihood-module>.<LikelihoodName>Theory:        # ← local JAX theory
    profile: arnaud10
    delta_crit: 500.0
    n_z: 100
    n_m: 200
    multipoles_file: <ABSOLUTE WORKDIR>/data/<ls-file>.txt
    omega_b: 0.0226
    omega_cdm: 0.118
    H0: 68.22
    tau_reio: 0.07
    ln10_10_As: 3.06
    n_s: 0.9743
params:
  # Foreground amplitudes (fixed at 0 unless you want to sample them)
  A_CIB: 0
  A_RS:  0
  A_IR:  0
  # Profile params: fix all 5 Arnaud-10 fields except P0 and beta (sampled)
  c500:      1.156
  gammaGNFW: 0.3292
  alphaGNFW: 1.062
  P0GNFW:   { prior: {min: 0, max: 20}, ref: {dist: norm, loc: 8.13,   scale: 0.1}, proposal: 0.1 }
  betaGNFW: { prior: {min: 0, max: 10}, ref: {dist: norm, loc: 5.4807, scale: 0.1}, proposal: 0.1 }
# ... sampler, output, debug as before ...
```

Drop everything that was `classy_szfast.classy_sz.classy_sz`-specific (use_class_sz_fast_mode, use_class_sz_no_cosmo_mode, the entire extra_args block) — the local JAX theory takes its config directly as class attributes.

5. **Validate with `--test`** from the workdir:
   ```bash
   cd <workdir>
   ~/pyvenvs/py312-class_sz/bin/cobaya-run --test <likelihood-name>.yaml
   ```
   Success looks like `[run] Test initialization successful! You can probably run now without --test.` plus speed measurements.

6. **Report back**:
   - Files written (absolute paths)
   - `--test` result (✓ or first error line)
   - Suggested next step (full chain, MPI command, getdist post-processing)

## Pitfalls

- The likelihood module is referenced by dotted Python path in the YAML (`<module>.<class>`). For the workdir layout above, `<module>` is the bare filename without `.py`. Always run cobaya-run from the workdir.
- `use_class_sz_no_cosmo_mode: 1` requires cosmology values in `extra_args`. Don't also declare them in the sampled `params` block — cobaya will pass them through and class_sz will error.
- The `multipoles` extra_arg points class_sz at a file of ell centers. The bandpower data's ell column MUST match those centers — verify with `np.loadtxt` and `np.allclose(ell_multipoles, ell_data)`.
- For `--jax`: JAX must be installed in the venv (`~/pyvenvs/py312-class_sz/bin/python -c 'import jax'`).
- Don't run python from `~/GitHub` — the local `cobaya/` clone there shadows the editable cobaya install via PEP 420 namespace-package resolution, breaking `from cobaya import LoggedError` and every soliket import.
