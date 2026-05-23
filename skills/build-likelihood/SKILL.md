---
description: Scaffold a cobaya likelihood for a y-map bandpower dataset using class_sz. Produces a Python likelihood module (Gaussian, modeled on SOLikeT SZLikelihood) plus a working cobaya input.yaml ready to MCMC. Optionally uses the JAX cl_yy_from_params pipeline directly for speed/differentiability.
disable-model-invocation: true
argument-hint: "<likelihood-name> [--jax] [--data-dir DIR]"
allowed-tools: Read Write Edit Glob Grep Bash(~/pyvenvs/py312-class_sz/bin/python *) Bash(~/pyvenvs/py312-class_sz/bin/cobaya-run -t *) Bash(~/pyvenvs/py312-class_sz/bin/cobaya-run --test *)
---

# Build a cobaya likelihood for a y-map bandpower dataset

Scaffold a new SOLikeT-style cobaya likelihood that fits a tSZ y-map bandpower dataset using class_sz. Output: a Python module file and a working cobaya YAML.

Arguments in `$ARGUMENTS`:
- `$0` — likelihood class name in CamelCase (e.g. `ACTYMapLikelihood`)
- Optional flags:
  - `--jax`: skip the `classy_szfast.classy_sz.classy_sz` theory wrapper; call `cl_yy_from_params` directly inside the likelihood. Faster, supports gradients, but only A10/B12 profiles.
  - `--data-dir DIR`: where the bandpower / cov / ls files live (default: ask the user)

## Steps

1. **Ask** for any missing details:
   - data directory (`--data-dir` if not given)
   - bandpower file name (3-col: ell, D_ell × 1e12, sigma)
   - cov file name (N×N)
   - multipoles file name (1-col, matches the bandpowers)
   - pressure profile: `GNFW` (default) or `B12`
   - which params are sampled — default: `P0GNFW`, `betaGNFW`; rest fixed
   - whether cosmology is fixed (default: yes, use `use_class_sz_no_cosmo_mode: 1`) or sampled
   - whether to include the foreground theory (default: include `SZForegroundTheory` with all foreground amplitudes fixed at 0)

2. **Write the likelihood module** to `<likelihood-name>.py` in the user's working directory (or wherever they specify). Skeleton (non-JAX path) — adapt parameter names and defaults to what the user said:

```python
from soliket.gaussian import GaussianLikelihood
import numpy as np, os
from typing import Optional

class <LikelihoodName>(GaussianLikelihood):
    sz_data_directory: Optional[str] = None
    ymap_ps_file:      Optional[str] = None
    ymap_cov_file:     Optional[str] = None
    f_sky:             Optional[float] = 0.47

    def initialize(self):
        D = np.loadtxt(os.path.join(self.sz_data_directory, self.ymap_ps_file))
        self.ell, self.y, self.sigma = D[:,0], D[:,1], D[:,2]
        try:
            self.covmat = np.loadtxt(os.path.join(self.sz_data_directory, self.ymap_cov_file))
        except Exception:
            self.covmat = np.diag(self.sigma**2)
        super().initialize()

    def get_requirements(self):
        return {"Cl_sz": {}}                          # add "Cl_sz_foreground": {} if FG theory enabled

    def _get_data(self): return self.ell, self.y
    def _get_cov(self):  return self.covmat
    def _get_theory(self, **p):
        t = self.provider.get_Cl_sz()
        return np.asarray(t['1h']) + np.asarray(t['2h'])
```

For the `--jax` path, use this skeleton instead — cosmology fixed at init, only profile params sampled:

```python
from cobaya.likelihood import Likelihood
import numpy as np, os
from typing import Optional
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from classy_szfast.differentiable import CosmoParams, ProfileParamsA10, cl_yy_from_params

class <LikelihoodName>(Likelihood):
    sz_data_directory: Optional[str] = None
    ymap_ps_file:      Optional[str] = None
    ymap_cov_file:     Optional[str] = None
    # fixed cosmology — override in YAML
    omega_b: float = 0.0226
    omega_cdm: float = 0.118
    H0: float = 68.22
    tau_reio: float = 0.07
    ln10_10_As: float = 3.06
    n_s: float = 0.9743

    def initialize(self):
        D = np.loadtxt(os.path.join(self.sz_data_directory, self.ymap_ps_file))
        self.ell_d, self.y, self.sigma = D[:,0], D[:,1], D[:,2]
        try:    cov = np.loadtxt(os.path.join(self.sz_data_directory, self.ymap_cov_file))
        except: cov = np.diag(self.sigma**2)
        self.inv_cov = np.linalg.inv(cov)
        self._cosmo = CosmoParams(self.omega_b, self.omega_cdm, self.H0,
                                  self.tau_reio, self.ln10_10_As, self.n_s)
        self._ell  = jnp.asarray(self.ell_d)
        # JIT once for speed:
        self._fn = jax.jit(lambda p: cl_yy_from_params(self._ell, self._cosmo, profile_params=p,
                                                      profile='arnaud10', delta_crit=500.0))

    def get_requirements(self): return {}             # no theory needed

    def logp(self, **p):
        prof = ProfileParamsA10(P0=p['P0GNFW'], c500=1.156, gamma=0.3292,
                                alpha=1.062, beta=p['betaGNFW'])
        cl1, cl2 = self._fn(prof)
        dl = self._ell * (self._ell + 1) / (2*jnp.pi) * (cl1 + cl2) * 1e12
        r = np.asarray(self.y - dl)
        return float(-0.5 * r @ self.inv_cov @ r)
```

3. **Write the cobaya YAML** to `<likelihood-name>.input.yaml`. Non-JAX example (mirrors the fionapaper may26 layout):

```yaml
theory:
  classy_szfast.classy_sz.classy_sz:
    use_class_sz_fast_mode: 1
    use_class_sz_no_cosmo_mode: 1
    extra_args:
      output: tSZ_1h
      mass_function: T08M500c
      pressure_profile: GNFW
      multipoles: <ABSOLUTE PATH TO ls.txt>
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
  <module_path>.<LikelihoodName>:
    sz_data_directory: <DATA_DIR>
    ymap_ps_file:      <BANDPOWER_FILE>
    ymap_cov_file:     <COV_FILE>
params:
  P0GNFW:   { prior: {min: 0,  max: 20}, ref: {dist: norm, loc: 8.13,   scale: 0.1}, proposal: 0.1, latex: 'P_0^{GNFW}' }
  betaGNFW: { prior: {min: 0,  max: 10}, ref: {dist: norm, loc: 5.4807, scale: 0.1}, proposal: 0.1, latex: '\beta^{GNFW}' }
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
output: chains/<likelihood-name>_test
debug: True
timing: true
```

For `--jax`, drop the `theory:` block entirely (the JAX-path likelihood has no theory requirements) and move the cosmology values up into the likelihood YAML block.

4. **Validate with a dry run** — use `cobaya-run --test` to verify the model assembles without launching MCMC:
   ```bash
   ~/pyvenvs/py312-class_sz/bin/cobaya-run --test <likelihood-name>.input.yaml
   ```

5. **Report back**:
   - Files written (with absolute paths)
   - Result of `--test` (✓ or first error)
   - Suggested next step (`cobaya-install` if any packages missing, then `cobaya-run`)

## Pitfalls

- The likelihood class must be importable. If you write it to `~/Work/.../my_lik.py`, the YAML reference is `<dotted.module>.<ClassName>`. Either put it on `PYTHONPATH` or refer to it by relative file path that's resolvable from the cobaya run dir.
- `use_class_sz_no_cosmo_mode: 1` requires cosmology in `extra_args`. Don't also declare cosmology in the cobaya `params` block — cobaya will pass it through and class_sz will complain.
- For the `--jax` path the user MUST have JAX installed in the venv (`~/pyvenvs/py312-class_sz/bin/python -c 'import jax'`). Check before scaffolding.
- The multipoles file must use the SAME ell centers as the bandpower data file. Verify with `np.loadtxt`.
