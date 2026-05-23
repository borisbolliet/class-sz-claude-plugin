# CLASS_SZ Assistant (Claude Code plugin)

Specialized [Claude Code](https://code.claude.com) assistance for [class_sz](https://github.com/CLASS-SZ/class_sz) (CLASS extension for halo-model SZ observables) and [classy_szfast](https://github.com/CLASS-SZ/classy_szfast) (Python wrapper, CosmoPower emulators, JAX pipeline).

Designed to compose with [cobaya-claude-plugin](https://github.com/borisbolliet/cobaya-claude-plugin) — enable both to get a full tSZ MCMC workflow.

## What you get

- **`/class-sz:explain`** — knowledge skill. Auto-loads when the conversation is about class_sz, classy_szfast, the tSZ power spectrum, the SOLikeT SZLikelihood pattern, GNFW / Arnaud / Battaglia pressure profiles, halo-model integrals, or cobaya wiring via `classy_szfast.classy_sz.classy_sz`. Bundles two pipeline cheat-sheets (classic + JAX) and a detailed parameter reference.
- **`/class-sz:tszfast [profile=arnaud10|battaglia12] [param=value …]`** — runs a quick C_ell^yy calculation through the JAX `cl_yy_from_params` pipeline; supports sweeps and `jax.grad` probes. Useful for sanity-checking before a long cobaya run.
- **`/class-sz:build-likelihood <Name> [--jax] [--data-dir DIR]`** — scaffolds a SOLikeT-style Gaussian likelihood + working cobaya YAML for a y-map bandpower dataset, ready to MCMC. The `--jax` variant skips the cobaya theory wrapper and calls the JAX pipeline directly from the likelihood (faster, differentiable, A10/B12 only).
- **`class-sz-engineer` subagent** — specialist for end-to-end class_sz + cobaya work (scaffold, install, run, summarize). Use for heavy multi-step tasks where you don't want install/chain output flooding the main thread.

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
- Python venv: `~/pyvenvs/py312-class_sz/bin/python` (with `classy_sz`, `classy_szfast`, `cobaya`, `soliket`, `getdist`, `jax`).
- class_sz source: `/Users/boris/GitHub/class_sz/`
- Reference workflow data (ACT-DR4-yy bandpowers, fionapaper run): `/Users/boris/Library/CloudStorage/GoogleDrive-boris.bolliet@gmail.com/My Drive/yy-2026/fionapaper/`

Adjust paths in `agents/class-sz-engineer.md` and the skill `allowed-tools` if your venv lives elsewhere.

## Try it

```
/class-sz:explain
What is the difference between use_class_sz_fast_mode and use_class_sz_no_cosmo_mode?

/class-sz:tszfast profile=battaglia12

/class-sz:build-likelihood ACTYMapLikelihood --data-dir "/Users/boris/.../fionapaper/act_dr4_yy"
```

## Layout

```
.claude-plugin/plugin.json
skills/
  explain/SKILL.md                # always-on knowledge
  explain/reference.md            # loaded on demand
  tszfast/SKILL.md                # /class-sz:tszfast
  build-likelihood/SKILL.md       # /class-sz:build-likelihood
agents/
  class-sz-engineer.md            # subagent
```

## Notes / open items

- **No MCP server.** The classy_sz CLI surface (Python + cobaya-run) is invoked through Bash; MCP would only buy a typed surface or a long-lived process (e.g. a warm `Class_sz()` instance), neither of which we need yet.
- **`disable-model-invocation: true` on `/class-sz:build-likelihood`** — scaffolding writes files, so Claude shouldn't decide on its own when to scaffold. The other two skills are model-invocable.
- **`paths` filter** is not set on `/class-sz:explain` — it's available everywhere, auto-triggered by description match.

## License

MIT
