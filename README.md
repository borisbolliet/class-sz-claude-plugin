# classy_szlite Assistant (Claude Code plugin)

Specialized [Claude Code](https://code.claude.com) assistance for
[**classy_szlite**](https://github.com/CLASS-SZ/classy_szlite) — a pure-JAX
cosmology code that exposes:

- CMB angular power spectra (TT, TE, EE)
- Linear and nonlinear matter Pk
- Cosmological distances (H(z), comoving, angular-diameter)
- Derived parameters (σ8, Ω_m, S8, ...)
- Halo-model tSZ Cl^yy (Arnaud-10 GNFW pressure profile)

…all backed by the high-accuracy **`v2` CosmoPower emulators** used in
the [ACT DR6 extended-cosmology analyses](https://arxiv.org/abs/2503.14454),
and entirely differentiable with `jax.grad` / `jax.jacfwd`.

Designed to compose with the
[cobaya-claude-plugin](https://github.com/borisbolliet/cobaya-claude-plugin)
— enable both to get a full tSZ MCMC workflow.

## What you get

- **`/class-sz:explain`** — auto-loads when the conversation is about
  classy_szlite, tSZ Cl^yy bandpower fits, GNFW / Arnaud pressure
  profiles, the `cl_yy_factory` fast path, JAX gradient probes, or
  cobaya wiring of a classy_szlite-backed theory. Bundles a focused
  classy_szlite API + parameter reference.

- **`/class-sz:tszfast [param=value …]`** — runs a quick `Cl^yy`
  calculation through `classy_szlite.cl_yy_factory`; supports
  parameter sweeps and `jax.grad` probes. Useful for sanity-checks
  before a long MCMC.

- **`/class-sz:build-likelihood <Name> [--data-dir DIR]`** — scaffolds
  a standalone Gaussian likelihood (no SOLikeT dep) + classy_szlite
  Theory + foreground Theory + a working cobaya YAML for a **tSZ
  Cl^yy bandpower** dataset (binned bandpowers + N×N covariance — not
  a y-map pixel likelihood). Uses `cl_yy_factory` so the per-step
  cost is ~5 ms/eval after init.

- **`class-sz-engineer` subagent** — specialist for end-to-end
  classy_szlite + cobaya work (scaffold, install, run, summarise).
  Use for heavy multi-step tasks where you don't want install / chain
  output flooding the main thread.

## Install

Local testing (no marketplace needed):

```bash
claude --plugin-dir ~/GitHub/class-sz-claude-plugin
```

From a marketplace (once published):

```
/plugin marketplace add <owner>/<marketplace-repo>
/plugin install class-sz@<marketplace-name>
```

## Environment

The skills and the `class-sz-engineer` subagent assume:

- Python venv: `~/pyvenvs/py312-class_sz/bin/python` (with
  `classy_szlite`, `cobaya`, `numpyro`, `getdist`, `jax`)
- classy_szlite source (optional, for editable install):
  `~/GitHub/classy_szlite`
- Emulator data: `~/class_sz_data/ede/` (or the path in
  `$CLASSY_SZLITE_DATA_DIR`) — see
  [cosmopower-organization/ede](https://github.com/cosmopower-organization/ede)
- Reference workflow data (ACT-DR6 may26 setup):
  `~/Desktop/class-sz-plugin-tests/` (data/, clyy_v2.py, clyy_v2.yaml,
  may26.proposal.covmat)

Adjust paths in `agents/class-sz-engineer.md` and the skill
`allowed-tools` if your venv lives elsewhere.

## Try it

```
/class-sz:explain
How do I sample profile parameters at fixed cosmology with NUTS?

/class-sz:tszfast P0=10 beta=5.0

/class-sz:build-likelihood ACTClyy --data-dir "~/Desktop/class-sz-plugin-tests/data"
```

## Layout

```
.claude-plugin/plugin.json
skills/
  explain/SKILL.md                # always-on classy_szlite knowledge
  explain/reference.md            # loaded on demand
  tszfast/SKILL.md                # /class-sz:tszfast
  build-likelihood/SKILL.md       # /class-sz:build-likelihood
agents/
  class-sz-engineer.md            # subagent
```

## License

MIT
