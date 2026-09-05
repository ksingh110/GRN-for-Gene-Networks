# GRN for Gene Networks

An evolutionary search workflow for constructing gene regulatory networks whose simulated expression trajectories match a target pattern. The project supports initial network generation, mutation, refinement, logic-gate selection, structural diagrams, Antimony export, and expression-trajectory plots.

## Repository layout

```text
.
├── src/                    # Python source and CLI entry point
├── examples/results/       # Small retained example outputs
├── archive/legacy_runs/    # Historical run archives
├── config.ini              # Default experiment configuration
├── requirements.txt        # Python dependencies
└── README.md
```

Generated run data is intentionally kept outside the source tree and ignored by Git.

## Installation

Python 3.11 or 3.12 is recommended. Create an isolated environment and install the dependencies:

```bash
git clone https://github.com/ksingh110/GRN-for-Gene-Networks.git
cd GRN-for-Gene-Networks
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Graphviz is also required for the hierarchical network layout. On macOS:

```bash
brew install graphviz
```

## Running experiments

Start the interactive CLI from the repository root:

```bash
python src/cli.py --interactive
```

Run one successful evolution search with the configuration defaults:

```bash
python src/cli.py --successes 1
```

All logic gates are enabled by default. Gates can be excluded for an experiment:

```bash
python src/cli.py --exclude-logic-gates NOR XOR
```

Refine a retained example network:

```bash
python src/cli.py --mode refine_only --refine-seed examples/results/generation_success_2.json
```

Use `python src/cli.py --help` for the complete command-line interface. Experiment defaults can be changed in `config.ini`, and `--seed` can be used for reproducible stochastic runs.

## Outputs

Each CLI invocation creates a timestamped directory under `results_cli/`. Successful networks are stored in numbered `generationSuccess_*` folders with their JSON representation, Antimony model, structural diagram, and expression plot. Failed runs are not persisted. Intermediate generation data may be written under `generations/` and `generationRefine/`.

## Research artifacts

- `examples/results/` contains compact, human-inspectable output examples.
- `archive/legacy_runs/` contains the original historical run archives retained for provenance.

For a publication release, create an immutable tagged version and archive that tag with the manuscript's exact configuration, random seed, and environment details.
