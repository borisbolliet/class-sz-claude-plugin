# class_sz / classy_szfast — detailed reference

Load this when SKILL.md's surface isn't enough — e.g. exact parameter names, output strings, emulator IDs, profile-field details.

## Cosmology parameters (passed to `Class_sz.set` or `CosmoParams`)

| Name (classic) | JAX field | Default-ish | Notes |
| --- | --- | --- | --- |
| `omega_b` | `omega_b` | 0.02242 | physical baryon density |
| `omega_cdm` | `omega_cdm` | 0.11933 | physical CDM density |
| `H0` | `H0` | 67.66 | Hubble constant |
| `tau_reio` | `tau_reio` | 0.0561 | reionization optical depth |
| `ln10^{10}A_s` | `ln10_10_As` | 3.047 | log primordial amplitude |
| `n_s` | `n_s` | 0.9665 | scalar tilt |
| `cosmo_model` | (n/a — JAX uses LCDM only currently) | 0 | classic only: 0 = ΛCDM emulators, 1 = mnu-ΛCDM emulators |

Alternatives accepted by classic: `logA` (== `ln10^{10}A_s`), `omega_m` (then internally set CDM).

## Halo-model / precision parameters (classic only)

| Param | Default | What it controls |
| --- | --- | --- |
| `z_min` | 0.005 | min redshift in halo integral |
| `z_max` | 3.0 | max redshift |
| `M_min` | 1e10 | min halo mass (Msun) |
| `M_max` | 3.5e15 | max halo mass (Msun) |
| `ndim_redshifts` | 500 | z-grid points (decrease for speed) |
| `n_z_pressure_profile` | 100 | pressure-profile z-grid |
| `n_m_pressure_profile` | 100 | pressure-profile M-grid |
| `redshift_epsabs` / `_epsrel` | 1e-40 / 1e-4 | quadrature tolerances |
| `mass_epsabs` / `_epsrel` | 1e-40 / 1e-4 | quadrature tolerances |
| `use_fft_for_profiles_transform` | 1 | use FFTs for profile Fourier transforms |
| `N_samp_fftw` | 1024 | FFT samples |
| `x_min_gas_pressure_fftw` | 1e-4 | FFT x_min |
| `x_max_gas_pressure_fftw` | 1e6 | FFT x_max |
| `x_outSZ` | 4.0 | truncate profile beyond `x_outSZ * r_s` |

## `output` string — common combinations

Concatenate with commas, no spaces:

| Output | What you get |
| --- | --- |
| `tSZ_1h` | only the 1-halo C_ell^yy |
| `tSZ_tSZ_1h` | 1-halo (full form) |
| `tSZ_tSZ_2h` | 2-halo |
| `tSZ_Trispectrum` | non-Gaussian C_ell^yy covariance |
| `kSZ_kSZ_1h` | kinetic SZ 1-halo |
| `tSZ_lensing_1h,tSZ_lensing_2h` | tSZ × CMB lensing |
| `gxg_1h,gxg_2h` | galaxy auto |
| `gxk_1h,gxk_2h` | galaxy × CMB lensing |
| `cibxcib_1h,cibxcib_2h` | CIB auto |
| `mPk` | matter P(k) at requested z |
| `sz_cluster_counts` / `sz_cluster_counts_binned` | cluster-count likelihoods |

Each adds methods on the `Class_sz` instance: `cl_sz()`, `tllprime_sz()`, `cl_kSZ_kSZ_1h()`, `cl_gxk()`, `cl_cib_cib()`, etc.

## Multipoles

Two ways to set:
1. **File**: `'multipoles': '/path/to/ls.txt'` — one ell per line; output `cl_sz()['ell']` matches
2. **Grid**: `'ell_min': 2, 'ell_max': 8000, 'dlogell': 0.2` (or `'dell'`)

The bandpower data file used by the likelihood **must use the same multipoles** as the theory `multipoles` file, otherwise binning will silently mismatch.

## Pressure-profile parameter lists

**GNFW / Arnaud 2010** (`pressure_profile: GNFW`, JAX `profile='arnaud10'`):
- `P0GNFW` — central pressure normalization (Arnaud10 best fit: 8.130)
- `c500` — concentration at r500 (1.156)
- `gammaGNFW` — inner slope (0.3292)
- `alphaGNFW` — transition slope (1.062)
- `betaGNFW` — outer slope (5.4807)
- `B` — mass bias (1 − b); 1.0 → 1.41 in literature; ACT papers use 1/B with B=1.25

JAX equivalent fields on `ProfileParamsA10`: `P0`, `c500`, `gamma`, `alpha`, `beta`. (The two are equivalent up to relabeling — `P0GNFW` ↔ `P0`, `gammaGNFW` ↔ `gamma`, etc.)

**Battaglia 2012** (`pressure_profile: B12`, JAX `profile='battaglia12'`, `delta_crit=200`):
Nine fitting coefficients on `ProfileParamsB12`:
- `P0_A`, `P0_am`, `P0_az` — normalization vs mass and z
- `xc_A`, `xc_am`, `xc_az` — core scale
- `beta_A`, `beta_am`, `beta_az` — outer slope

P0(M,z) = P0_A · (M/1e14)^P0_am · (1+z)^P0_az  (analogous for xc, beta).

## Emulators / `cosmo_model`

Set via `'cosmo_model'`:
- `0` — pure ΛCDM (default; uses CosmoPower trained on LCDM)
- `1` — ΛCDM + Σm_ν (mnu-ΛCDM emulators)
- Newer values may unlock w0wa, extensions — check `classy_szfast.cosmosis_emulators`

Switching `cosmo_model` mid-run is not supported — recreate the `Class_sz()` instance.

## cobaya wrapper internals

`classy_szfast.classy_sz.classy_sz` inherits from `cobaya.theories.classy.classy` and overrides:

| Method | What it does |
| --- | --- |
| `initialize` | imports `from classy_sz import Class` |
| `must_provide` | registers a `Collector(method='cl_sz')` if `Cl_sz` is requested; also `sz_binned_cluster_counts`, `sz_unbinned_cluster_counts` |
| `get_Cl_sz` | returns deep-copied `_current_state["Cl_sz"]` |

Options on the YAML side:
- `use_class_sz_fast_mode: 1` — call `compute_class_szfast()` instead of `compute()`
- `use_class_sz_no_cosmo_mode: 1` — bypass cosmology (cosmo params come from `extra_args`)
- `extra_args` — dict passed via `c.set(...)` before each `compute*`

## File-layout convention for a bandpower likelihood

| File | Cols | What |
| --- | --- | --- |
| `ls_<tag>.txt` | 1 | ell centres (one per bandpower) |
| `mean_<tag>.txt` | 1 | mean D_ell^yy × 1e12 per bandpower |
| `data_ps-ell-y2-erry2_<tag>.txt` | 3 | ell · D_ell · sigma (Gaussian only) |
| `cov_<tag>.txt` | N×N | full covariance matrix |

The fionapaper ACT-DR4-yy set uses `lmin=2000`, 8 bandpowers (ells 1000–6000), with a covariance file `cov_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt`.

## Common cobaya YAML extras for class_sz runs

```yaml
sampler:
  mcmc:
    blocking:
      - [1, [P0GNFW, betaGNFW]]      # explicit block; no fast/slow needed when cosmology is fixed
    proposal_scale: 1.9
    oversample_power: 0.4
    oversample_thin: true
```

When cosmology is sampled (no `use_class_sz_no_cosmo_mode`), add `drag: true` and let cobaya auto-block; the cosmology call is the slow component.

## Foregrounds (SOLikeT `SZForegroundTheory`)

Reads `data_fg-ell-cib_rs_ir_cn-total-planck-collab-15.txt` (5 cols: ell, CIB, RS, IR, CN templates) and returns:
```
Cl_sz_foreground = A_CIB * CIB + A_RS * RS + A_IR * IR + 0.9033 * CN
```
The `0.9033` is the correlated-noise template amplitude from Bolliet et al. 1712.00788, hard-coded.

## Useful classy_szfast functions

| Function | Purpose |
| --- | --- |
| `classy_szfast.install_emulators()` | (re)download CosmoPower emulators into `~/.classy_szfast/` |
| `classy_szfast.differentiable.cl_yy_from_params` | JAX C_ell^yy |
| `classy_szfast.differentiable.CosmoParams` | NamedTuple of cosmology |
| `classy_szfast.differentiable.ProfileParamsA10` / `B12` | NamedTuple of profile params |

## Reference papers
- Bolliet et al. 2018 (arXiv:1712.00788) — Planck tSZ likelihood
- Komatsu+Seljak 2001 — halo-model tSZ
- Arnaud et al. 2010 — GNFW pressure profile
- Battaglia et al. 2012 — B12 pressure profile
- CLASS_SZ paper: Bolliet et al. (in prep — see CLASS-SZ GitHub for current refs)
