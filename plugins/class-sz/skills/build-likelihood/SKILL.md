---
description: Scaffold a cobaya likelihood for a tSZ Cl^yy bandpower dataset using classy_szlite. Produces a Python file (Likelihood + Theory + Foreground Theory) and a working cobaya YAML, ready for cobaya-run. The Theory uses classy_szlite.cl_yy_factory so per-step cost is ~5 ms/eval after init.
disable-model-invocation: true
argument-hint: "<likelihood-name> [--data-dir DIR]"
allowed-tools: Read Write Edit Glob Grep Bash(~/pyvenvs/py312-class_sz/bin/python *) Bash(~/pyvenvs/py312-class_sz/bin/cobaya-run -t *) Bash(~/pyvenvs/py312-class_sz/bin/cobaya-run --test *)
---

# Build a cobaya likelihood for a Cl^yy bandpower dataset

Scaffold a **classy_szlite**-backed Gaussian likelihood + foreground
theory + JAX-backed Theory module and a working cobaya YAML for a
**tSZ Cl^yy bandpower dataset** (data is binned bandpowers + N×N
covariance — NOT a y-map pixel likelihood, despite the historical
`ymap_ps.py` filename in some legacy code).

Arguments in `$ARGUMENTS`:

- `$0` — likelihood class-name root, CamelCase (e.g. `Clyy`). The skill
  writes three classes: `<Root>Likelihood`, `<Root>Theory`,
  `<Root>ForegroundTheory`.
- `--data-dir DIR` — directory with the bandpower / cov / multipoles /
  foreground files (otherwise ask the user).

## Why classy_szlite

Pure JAX + numpy + mcfit — no tensorflow / cosmopower at runtime
(the `_v2_plain.npz` emulator files load with `allow_pickle=False`).
Per-step cost ~5 ms via the `cl_yy_factory` closure that precomputes
CosmoGrids + HaloGrids once at `initialize()`.

The standard-6 ΛCDM cosmology is surfaced on the Theory; the v2
emulator's extra fields (`fEDE`, `log10z_c`, `thetai_scf`, `r`,
`m_ncdm`, `N_ur`) stay at their silently LCDM-equivalent defaults
(`fEDE = 0.001`). If your fit needs them sampled, edit the Theory's
`initialize()` to expose them as class attrs and pass them through
to `csl.CosmoParams(...)`.

## Workdir convention

```
<workdir>/
├── <likelihood-name>.py       # 3 classes: Likelihood + Theory + Foreground
├── <likelihood-name>.yaml     # cobaya input
├── data/                      # bandpowers / cov / multipoles / foreground
└── chains/                    # cobaya output
```

If `<workdir>` already has a `data/` subdir with the expected files,
use them. Otherwise ask the user where the data lives and propose
copying it into `data/`.

**Always run cobaya-run from the workdir** so the module is on
`sys.path` (the bare module name resolves; not from `~/GitHub` where
PEP 420 namespace shadows editable installs).

## Steps

1. **Read the data.** `np.loadtxt` the bandpower file (3 cols: ell,
   D_ell × 1e12, σ); confirm shape matches the cov file (N × N) and
   the multipoles file (N rows). Mismatched ell centres between data
   and multipoles is a silent bug — verify.

2. **Ask** for missing details:
   - data directory if not given via `--data-dir`
   - bandpower / cov / multipoles / foreground file names
   - sampled params (default: `P0GNFW`, `betaGNFW`; rest fixed)
   - hydrostatic mass bias B (default: `1.25` for ACT-DR6 may26-style)

3. **Write the Python module** to `<workdir>/<likelihood-name>.py`.
   Reference shape (mirrors the validated
   `~/Desktop/class-sz-plugin-tests/clyy_v2.py`):

```python
"""<Root> — Cl^yy bandpower likelihood + classy_szlite Theory."""
from __future__ import annotations
import os, numpy as np
from typing import Optional
from cobaya.likelihood import Likelihood
from cobaya.theory     import Theory


class <Root>Likelihood(Likelihood):
    sz_data_directory: Optional[str] = None
    ymap_ps_file:      Optional[str] = None    # 3 cols: ell, D_ell · 1e12, σ
    ymap_cov_file:     Optional[str] = None    # N × N covariance

    def initialize(self):
        D = np.loadtxt(os.path.join(self.sz_data_directory, self.ymap_ps_file))
        self.ell, self.y, self.sigma = D[:, 0], D[:, 1], D[:, 2]
        self.cov = (np.loadtxt(os.path.join(self.sz_data_directory, self.ymap_cov_file))
                    if self.ymap_cov_file else np.diag(self.sigma**2))
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


class <Root>Theory(Theory):
    """classy_szlite-backed Cl_sz provider using cl_yy_factory fast path.

    Precomputes CosmoGrids + HaloGrids once at initialize() time
    (cosmology is fixed for this Theory); per-step calls only the
    cl_yy_1h_2h integration. ~5 ms / eval after warmup.

    Only the standard 6 LCDM cosmology params are surfaced as class
    attributes; v2 emulator extras (fEDE, log10z_c, thetai_scf, r,
    m_ncdm, N_ur) are silent at their LCDM-equivalent defaults.
    """
    multipoles_file: Optional[str] = None      # required
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

    params = {
        "P0GNFW":    8.130,
        "c500":      1.156,
        "gammaGNFW": 0.3292,
        "alphaGNFW": 1.062,
        "betaGNFW":  5.4807,
        "B":         1.25,
    }

    def initialize(self):
        import jax; jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import classy_szlite as csl

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


class <Root>ForegroundTheory(Theory):
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

4. **Write the cobaya YAML** to `<workdir>/<likelihood-name>.yaml`.
   Only the standard 6 cosmology params are exposed:

```yaml
theory:
  <module>.<Root>ForegroundTheory:
    foreground_data_directory: <ABSOLUTE WORKDIR>/data/
    foreground_data_file: data_fg-ell-cib_rs_ir_cn-total-planck-collab-15.txt
  <module>.<Root>Theory:
    n_z: 100
    n_m: 200
    delta_crit: 500.0
    multipoles_file: <ABSOLUTE WORKDIR>/data/<ls-file>.txt
    # Standard 6 cosmological parameters (fixed)
    omega_b:    0.0226
    omega_cdm:  0.118
    H0:         68.22
    tau_reio:   0.0561
    ln10_10_As: 3.06
    n_s:        0.9743

likelihood:
  <module>.<Root>Likelihood:
    sz_data_directory: <ABSOLUTE WORKDIR>/data/
    ymap_ps_file:      <bandpower-file>.txt
    ymap_cov_file:     <cov-file>.txt

params:
  A_CIB: 0
  A_RS:  0
  A_IR:  0
  c500:      1.156
  gammaGNFW: 0.3292
  alphaGNFW: 1.062
  B:         1.25
  P0GNFW:   { prior: {min: 0, max: 20}, ref: {dist: norm, loc: 8.13,   scale: 0.1}, proposal: 0.1 }
  betaGNFW: { prior: {min: 0, max: 10}, ref: {dist: norm, loc: 5.4807, scale: 0.1}, proposal: 0.1 }

sampler:
  mcmc:
    Rminus1_stop: 0.05            # 0.01 for production
    max_tries: 10000
    burn_in: 100
    learn_proposal: true
    blocking: [[1, [P0GNFW, betaGNFW]]]
    proposal_scale: 1.9

output: <ABSOLUTE WORKDIR>/chains/<likelihood-name>
debug: True
timing: true
```

5. **Validate with `--test`** from the workdir:

   ```bash
   cd <workdir>
   ~/pyvenvs/py312-class_sz/bin/cobaya-run --test <likelihood-name>.yaml
   ```

6. **Report back**:
   - Files written (absolute paths)
   - `--test` result + `Average evaluation time` for `<Root>Theory`
     (expect ~5–15 ms warm)
   - Suggested next: full chain
     (`cobaya-run` or `mpirun -np 4 cobaya-run`)

## Prerequisites

The user's venv must have `classy_szlite` installed:

```bash
~/pyvenvs/py312-class_sz/bin/pip install classy_szlite
# or from a local editable checkout
~/pyvenvs/py312-class_sz/bin/pip install -e ~/GitHub/classy_szlite
```

Verify:
```bash
~/pyvenvs/py312-class_sz/bin/python -c "import classy_szlite; print(classy_szlite.__version__)"
```

The emulator data must be at `~/class_sz_data/ede/` (or
`$CLASSY_SZLITE_DATA_DIR`). Get the files from
[cosmopower-organization/ede](https://github.com/cosmopower-organization/ede).

## Pitfalls

- Always run `cobaya-run` from `<workdir>` so the module is
  importable. Don't run from `~/GitHub` (PEP 420 namespace collision
  with the local cobaya / classy_szlite repo subfolders shadows the
  editable installs).
- The `multipoles_file` must use the SAME ell centres as the
  bandpower data file — verify with `np.loadtxt`.
- For **EDE-mode work** (sampling `fEDE` etc.), edit
  `<Root>Theory.initialize` to expose those as class attrs and pass
  them to `csl.CosmoParams(...)`. The default scaffold keeps EDE
  silent at the LCDM-equivalent point (`fEDE = 0.001`).
- `cl_yy_factory` precomputes CosmoGrids + HaloGrids ONCE — if you
  want to **sample cosmology too**, replace the factory call with the
  full `csl.cl_yy(...)` pipeline (~18 ms/eval) and rebuild the
  CosmoParams inside `calculate()`. Or use NumPyro NUTS instead of
  cobaya RW-MH for a 10× speedup; see `/class-sz:explain` →
  "Gradient-based sampling" for the recipe.
