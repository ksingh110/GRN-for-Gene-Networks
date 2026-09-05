import argparse
import configparser
import json
import multiprocessing as mp
import os
import random
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from generationCreation import (
    compute_generation_initial,
    generationCreate,
)

from refineGeneration import (
    extract_seed_network,
    load_seed_network,
    run_refinement,
)
from diagram import diagram_network
from visualize import visualize_network, saveAntimony
from randomNetwork import prune_unconnected_genes


VERSION = "0.3.0"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

OSCILLATOR = [5.0, 30.0, 5.0, 30.0, 5.0, 30.0, 5.0, 30.0, 5.0, 30.0]
SWITCH = [30.0, 30.0, 30.0, 30.0, 30.0,5.0, 5.0, 5.0, 5.0, 5.0]
DIP = [30.0, 30.0, 30.0, 30.0, 5.0,5.0, 30.0, 30.0, 30.0, 30.0]
ALL_LOGIC_GATES = ["AND", "OR", "NOR", "NAND", "XOR", "EQ"]


def parse_logic_gates(value):
    if isinstance(value, str):
        values = value.replace(",", " ").split()
    else:
        values = [part for value_part in value for part in value_part.replace(",", " ").split()]

    values = [value.upper() for value in values]

    if values == ["ALL"]:
        return list(ALL_LOGIC_GATES)

    if values == ["NONE"]:
        return []

    invalid = [value for value in values if value not in ALL_LOGIC_GATES]
    if invalid:
        raise ValueError("Unknown logic gates: " + ", ".join(invalid))

    return list(dict.fromkeys(values))


DEFAULTS = {
    "mode": "evolve",
    "target_successes": 1,
    "trials": None,
    "target": 100.0,
    "target_pattern": "OSCILLATOR",
    "exclude_logic_gates": [],
    "logic_gates": list(ALL_LOGIC_GATES),
    "max_generations": 500,
    "extend_threshold": 750.0,
    "extend_by": 300,
    "elite_percent": 10,

    # Refinement settings
    "refine": False,
    "refine_seed_path": "",
    "refine_target": 100.0,
    "refine_max_generations": 300,
    "refine_extend_threshold": 120.0,
    "refine_extend_by": 300,
    "refine_elite_percent": 10,

    # General settings
    "results_dir": "results_cli",
    "prefix": "",
    "print_every": 50,
    "workers": max(1, (os.cpu_count() or 1) - 1),
    "seed": None,
}


def load_config(path):
    cfg = dict(DEFAULTS)

    if not path or not os.path.exists(path):
        return cfg

    parser = configparser.ConfigParser()
    parser.read(path)

    if "network_evolver" not in parser:
        return cfg

    section = parser["network_evolver"]

    cfg["mode"] = section.get(
        "mode",
        cfg["mode"],
    ).strip().lower()

    if cfg["mode"] not in {"evolve", "refine_only"}:
        raise ValueError(
            "mode must be evolve or refine_only"
        )

    cfg["target_successes"] = section.getint(
        "target_successes",
        cfg["target_successes"],
    )

    trials_raw = section.get("trials", "").strip()
    cfg["trials"] = (
        int(trials_raw)
        if trials_raw
        else None
    )

    cfg["target"] = section.getfloat(
        "target",
        cfg["target"],
    )

    cfg["target_pattern"] = section.get(
        "target_pattern",
        cfg["target_pattern"],
    ).strip().upper()

    cfg["exclude_logic_gates"] = parse_logic_gates(
        section.get("exclude_logic_gates", "NONE")
    )
    cfg["logic_gates"] = [
        gate for gate in ALL_LOGIC_GATES
        if gate not in cfg["exclude_logic_gates"]
    ]

    if cfg["target_pattern"] not in {
        "OSCILLATOR",
        "SWITCH",
        "DIP",
    }:
        raise ValueError(
            "target_pattern must be "
            "OSCILLATOR, SWITCH, or DIP"
        )

    cfg["max_generations"] = section.getint(
        "max_generations",
        cfg["max_generations"],
    )

    cfg["extend_threshold"] = section.getfloat(
        "extend_threshold",
        cfg["extend_threshold"],
    )

    cfg["extend_by"] = section.getint(
        "extend_by",
        cfg["extend_by"],
    )

    cfg["elite_percent"] = section.getint(
        "elite_percent",
        cfg["elite_percent"],
    )

    cfg["refine"] = section.getboolean(
        "refine",
        cfg["refine"],
    )

    cfg["refine_seed_path"] = section.get(
        "refine_seed_path",
        cfg["refine_seed_path"],
    ).strip()

    cfg["refine_target"] = section.getfloat(
        "refine_target",
        cfg["refine_target"],
    )

    cfg["refine_max_generations"] = section.getint(
        "refine_max_generations",
        cfg["refine_max_generations"],
    )

    cfg["refine_extend_threshold"] = section.getfloat(
        "refine_extend_threshold",
        cfg["refine_extend_threshold"],
    )

    cfg["refine_extend_by"] = section.getint(
        "refine_extend_by",
        cfg["refine_extend_by"],
    )

    cfg["refine_elite_percent"] = section.getint(
        "refine_elite_percent",
        cfg["refine_elite_percent"],
    )

    cfg["results_dir"] = section.get(
        "results_dir",
        cfg["results_dir"],
    ).strip()

    cfg["prefix"] = section.get(
        "prefix",
        cfg["prefix"],
    )

    cfg["print_every"] = section.getint(
        "print_every",
        cfg["print_every"],
    )

    workers_raw = section.get("workers", "").strip()
    cfg["workers"] = (
        int(workers_raw)
        if workers_raw
        else cfg["workers"]
    )

    seed_raw = section.get("seed", "").strip()
    cfg["seed"] = (
        int(seed_raw)
        if seed_raw
        else None
    )

    return cfg


def build_parser():
    parser = argparse.ArgumentParser(
        prog="network_evolver",
        description=(
            "Evolve gene networks, optionally refine successful "
            "networks, or refine an existing seed network."
        ),
    )

    parser.add_argument(
        "--config",
        default="config.ini",
        help="Path to config file",
    )

    parser.add_argument(
        "--mode",
        choices=["evolve", "refine_only"],
        default=None,
        help="Run normal evolution or refine only an existing network",
    )

    refine_group = parser.add_mutually_exclusive_group()

    refine_group.add_argument(
        "--refine",
        dest="refine",
        action="store_true",
        help="Refine every successful evolved network once",
    )

    refine_group.add_argument(
        "--no-refine",
        dest="refine",
        action="store_false",
        help="Do not refine successful evolved networks",
    )

    parser.set_defaults(refine=None)

    parser.add_argument(
        "--refine-seed",
        dest="refine_seed_path",
        default=None,
        help="Network JSON file used in refine_only mode",
    )

    parser.add_argument(
        "--refine-target",
        type=float,
        default=None,
        help="Refinement convergence threshold",
    )

    parser.add_argument(
        "--refine-max-generations",
        type=int,
        default=None,
        help="Base generation limit for refinement",
    )

    parser.add_argument(
        "--refine-extend-threshold",
        type=float,
        default=None,
        help="Fitness threshold that extends refinement",
    )

    parser.add_argument(
        "--refine-extend-by",
        type=int,
        default=None,
        help="Number of extra refinement generations",
    )

    parser.add_argument(
        "--refine-elite-percent",
        type=int,
        default=None,
        help="Number of refinement elites retained",
    )

    parser.add_argument(
        "--successes",
        type=int,
        default=None,
        help="Stop after this many successful evolution trials",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Run exactly this many evolution trials",
    )

    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Normal evolution convergence threshold",
    )

    parser.add_argument(
        "--target-pattern",
        type=str.upper,
        choices=["OSCILLATOR", "SWITCH", "DIP"],
        default=None,
        help="Target pattern",
    )

    parser.add_argument(
        "--exclude-logic-gates",
        nargs="+",
        default=None,
        help="Logic gates to exclude: AND OR NOR NAND XOR EQ, ALL, or NONE",
    )

    parser.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="Base generation limit per evolution trial",
    )

    parser.add_argument(
        "--elite-percent",
        type=int,
        default=None,
        help="Number of normal-evolution elites retained",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fixed random-number seed",
    )

    parser.add_argument(
        "-silent",
        action="store_true",
        help="Only print major results",
    )

    parser.add_argument(
        "-nds",
        type=int,
        default=None,
        help="Generations between progress dots in silent mode",
    )

    parser.add_argument(
        "-pop",
        type=str,
        default=None,
        help="Output best fitness per generation to this filename",
    )

    parser.add_argument(
        "-w",
        action="store_true",
        help="Wait for the user to press Enter before exiting",
    )

    parser.add_argument(
        "-printDefaults",
        action="store_true",
        help="Print configuration values and exit",
    )

    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Open the interactive configuration menu",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser


def apply_command_line_arguments(cfg, args):
    if args.mode is not None:
        cfg["mode"] = args.mode

    if args.refine is not None:
        cfg["refine"] = args.refine

    if args.refine_seed_path is not None:
        cfg["refine_seed_path"] = args.refine_seed_path

    if args.refine_target is not None:
        cfg["refine_target"] = args.refine_target

    if args.refine_max_generations is not None:
        cfg["refine_max_generations"] = (
            args.refine_max_generations
        )

    if args.refine_extend_threshold is not None:
        cfg["refine_extend_threshold"] = (
            args.refine_extend_threshold
        )

    if args.refine_extend_by is not None:
        cfg["refine_extend_by"] = args.refine_extend_by

    if args.refine_elite_percent is not None:
        cfg["refine_elite_percent"] = (
            args.refine_elite_percent
        )

    if args.trials is not None:
        cfg["trials"] = args.trials
        cfg["target_successes"] = None

    if args.successes is not None:
        cfg["target_successes"] = args.successes
        cfg["trials"] = None

    if args.target is not None:
        cfg["target"] = args.target

    if args.target_pattern is not None:
        cfg["target_pattern"] = args.target_pattern

    if args.exclude_logic_gates is not None:
        cfg["exclude_logic_gates"] = parse_logic_gates(args.exclude_logic_gates)
        cfg["logic_gates"] = [
            gate for gate in ALL_LOGIC_GATES
            if gate not in cfg["exclude_logic_gates"]
        ]

    if args.max_generations is not None:
        cfg["max_generations"] = args.max_generations

    if args.elite_percent is not None:
        cfg["elite_percent"] = args.elite_percent

    if args.workers is not None:
        cfg["workers"] = args.workers

    if args.seed is not None:
        cfg["seed"] = args.seed

    return cfg


def prompt_int(label, default):
    raw = input(
        f"New value for {label} [{default}]: "
    ).strip()

    if raw == "":
        return default

    try:
        return int(raw)
    except ValueError:
        print(
            f"  Invalid integer. Keeping {default}."
        )
        return default


def prompt_optional_int(label, default):
    display = default if default is not None else "unused"

    raw = input(
        f"New value for {label} [{display}]: "
    ).strip()

    if raw == "":
        return default

    if raw.lower() in {"none", "unused", "off"}:
        return None

    try:
        return int(raw)
    except ValueError:
        print(
            f"  Invalid integer. Keeping {display}."
        )
        return default


def prompt_float(label, default):
    raw = input(
        f"New value for {label} [{default}]: "
    ).strip()

    if raw == "":
        return default

    try:
        return float(raw)
    except ValueError:
        print(
            f"  Invalid number. Keeping {default}."
        )
        return default


def prompt_str(label, default):
    raw = input(
        f"New value for {label} [{default}]: "
    ).strip()

    if raw == "":
        return default

    return raw


def prompt_bool(label, default):
    default_text = "yes" if default else "no"

    raw = input(
        f"New value for {label} [{default_text}]: "
    ).strip().lower()

    if raw == "":
        return default

    if raw in {"yes", "y", "true", "1", "on"}:
        return True

    if raw in {"no", "n", "false", "0", "off"}:
        return False

    print(
        f"  Enter yes or no. Keeping {default_text}."
    )

    return default


def prompt_mode(label, default):
    raw = input(
        f"New value for {label} [{default}]: "
    ).strip().lower()

    if raw == "":
        return default

    if raw not in {"evolve", "refine_only"}:
        print(
            "  Mode must be evolve or refine_only. "
            f"Keeping {default}."
        )
        return default

    return raw


def prompt_target_pattern(label, default):
    raw = input(
        f"New value for {label} [{default}]: "
    ).strip().upper()

    if raw == "":
        return default

    if raw not in {"OSCILLATOR", "SWITCH", "DIP"}:
        print(
            "  Pattern must be OSCILLATOR, SWITCH, or DIP. "
            f"Keeping {default}."
        )
        return default

    return raw


def prompt_logic_gates(label, default):
    display = ",".join(default) if default else "NONE"
    raw = input(f"New value for {label} [{display}]: ").strip()

    if raw == "":
        return default

    try:
        return parse_logic_gates(raw)
    except ValueError as error:
        print(f"  {error}. Keeping {display}.")
        return default


MENU_ITEMS = [
    ("Mode (evolve/refine_only)", "mode", "mode"),
    ("Refine successful networks", "refine", "bool"),
    ("Refine-only seed path", "refine_seed_path", "str"),
    ("Target pattern", "target_pattern", "target_pattern"),
    ("Excluded logic gates", "exclude_logic_gates", "logic_gates"),
    ("Run mode (successes/trials)", "_run_mode", "str"),
    (
        "Target successes (if successes)",
        "target_successes",
        "optional_int",
    ),
    (
        "Number of trials (if trials)",
        "trials",
        "optional_int",
    ),
    ("Random seed mode", "_seed_mode", "str"),
    ("Fixed random seed", "seed", "optional_int"),
    ("Evolution fitness target", "target", "float"),
    ("Evolution max generations", "max_generations", "int"),
    ("Evolution extend threshold", "extend_threshold", "float"),
    ("Evolution extend by", "extend_by", "int"),
    ("Evolution elite count", "elite_percent", "int"),
    ("Refinement fitness target", "refine_target", "float"),
    (
        "Refinement max generations",
        "refine_max_generations",
        "int",
    ),
    (
        "Refinement extend threshold",
        "refine_extend_threshold",
        "float",
    ),
    ("Refinement extend by", "refine_extend_by", "int"),
    (
        "Refinement elite count",
        "refine_elite_percent",
        "int",
    ),
    ("CPU workers", "workers", "int"),
    ("Results directory", "results_dir", "str"),
    ("Output prefix", "prefix", "str"),
    ("Print every N generations", "print_every", "int"),
]


def run_mode_display(cfg):
    if cfg["target_successes"] is not None:
        return "successes"

    return "trials"


def seed_mode_display(cfg):
    if cfg["seed"] is not None:
        return "fixed"

    return "random"


def run_interactive_setup(cfg):
    while True:
        print()
        print("Network Evolver Configuration")
        print("=" * 62)

        for index, (label, key, _) in enumerate(
            MENU_ITEMS,
            start=1,
        ):
            if key == "_run_mode":
                value = run_mode_display(cfg)

            elif key == "_seed_mode":
                value = seed_mode_display(cfg)

            else:
                value = cfg[key]

                if key == "exclude_logic_gates":
                    value = ",".join(value) if value else "NONE"

                if value is None or value == "":
                    value = "(unused)"

            print(
                f" {index:2d}) {label:38s}: {value}"
            )

        print("-" * 62)
        print(" R) Run with these settings")
        print(" Q) Quit")
        print("-" * 62)

        choice = input(
            "Select an option: "
        ).strip().lower()

        if choice == "r":
            print()
            print("Starting run...")
            print()
            return cfg

        if choice == "q":
            print("Exiting.")
            sys.exit(0)

        try:
            item_index = int(choice) - 1

            if not 0 <= item_index < len(MENU_ITEMS):
                raise ValueError

        except ValueError:
            print("  Invalid menu option.")
            continue

        label, key, kind = MENU_ITEMS[item_index]

        if key == "_run_mode":
            new_mode = prompt_str(
                "run mode (successes/trials)",
                run_mode_display(cfg),
            ).lower()

            if new_mode.startswith("t"):
                cfg["trials"] = cfg["trials"] or 1
                cfg["target_successes"] = None
            else:
                cfg["target_successes"] = (
                    cfg["target_successes"] or 1
                )
                cfg["trials"] = None

            continue

        if key == "_seed_mode":
            new_mode = prompt_str(
                "seed mode (random/fixed)",
                seed_mode_display(cfg),
            ).lower()

            if new_mode.startswith("f"):
                if cfg["seed"] is None:
                    cfg["seed"] = random.randrange(2**32)
            else:
                cfg["seed"] = None

            continue

        if kind == "int":
            cfg[key] = prompt_int(label, cfg[key])

        elif kind == "optional_int":
            cfg[key] = prompt_optional_int(
                label,
                cfg[key],
            )

        elif kind == "float":
            cfg[key] = prompt_float(label, cfg[key])

        elif kind == "bool":
            cfg[key] = prompt_bool(label, cfg[key])

        elif kind == "mode":
            cfg[key] = prompt_mode(label, cfg[key])

        elif kind == "target_pattern":
            cfg[key] = prompt_target_pattern(
                label,
                cfg[key],
            )

        elif kind == "logic_gates":
            cfg[key] = prompt_logic_gates(label, cfg[key])

        else:
            cfg[key] = prompt_str(label, cfg[key])


def validate_config(cfg):
    cfg["exclude_logic_gates"] = parse_logic_gates(cfg["exclude_logic_gates"])
    cfg["logic_gates"] = [
        gate for gate in ALL_LOGIC_GATES
        if gate not in cfg["exclude_logic_gates"]
    ]

    if cfg["mode"] not in {"evolve", "refine_only"}:
        raise ValueError(
            "mode must be evolve or refine_only"
        )

    if cfg["target_pattern"] not in {
        "OSCILLATOR",
        "SWITCH",
        "DIP",
    }:
        raise ValueError(
            "target_pattern must be OSCILLATOR, SWITCH, or DIP"
        )

    if cfg["workers"] < 1:
        raise ValueError(
            "workers must be at least 1"
        )

    if not 1 <= cfg["elite_percent"] <= 99:
        raise ValueError(
            "elite_percent must be between 1 and 99"
        )

    if not 1 <= cfg["refine_elite_percent"] <= 99:
        raise ValueError(
            "refine_elite_percent must be between 1 and 99"
        )

    if cfg["mode"] == "evolve":
        if (
            cfg["trials"] is None
            and cfg["target_successes"] is None
        ):
            cfg["target_successes"] = 1

    if cfg["mode"] == "refine_only":
        if not cfg["refine_seed_path"]:
            raise ValueError(
                "refine_seed_path is required in refine_only mode"
            )

        if not os.path.exists(cfg["refine_seed_path"]):
            raise FileNotFoundError(
                "Refinement seed file does not exist: "
                f"{cfg['refine_seed_path']}"
            )

    return cfg


def get_target_pattern(pattern_name):
    pattern_name = pattern_name.upper()

    if pattern_name == "OSCILLATOR":
        return OSCILLATOR

    if pattern_name == "SWITCH":
        return SWITCH

    if pattern_name == "DIP":
        return DIP

    raise ValueError(
        f"Unknown target pattern: {pattern_name}"
    )


def get_random_seed(cfg):
    if cfg["seed"] is not None:
        return cfg["seed"]

    return random.randrange(2**32)


def run_evolution_trial(
    cfg,
    trial_index,
    pool,
    random_seed,
    target_pattern,
    silent,
    nds,
    population_file,
):
    random.seed(random_seed)

    print(f"Seed: {random_seed}")

    compute_generation_initial(
        1,
        target_pattern,
        mp_pool=pool,
        logic_gates=cfg["logic_gates"],
    )

    with open(
        "generations/generation_1.json"
    ) as file:
        data = json.load(file)

    best_fitness = (
        data["networks"][0]["fitness"]["score"]
    )

    generation_number = 1
    generation_limit = cfg["max_generations"]
    limit_extended = False
    start_time = time.perf_counter()
    population_history = [(0, best_fitness)]

    while (
        best_fitness > cfg["target"]
        and generation_number <= generation_limit
    ):
        data = generationCreate(
            cfg["elite_percent"],
            data,
            generation_number,
            target_pattern,
            mp_pool=pool,
            logic_gates=cfg["logic_gates"],
        )

        new_best = (
            data["networks"][0]["fitness"]["score"]
        )

        best_fitness = min(
            best_fitness,
            new_best,
        )

        population_history.append(
            (generation_number, best_fitness)
        )

        if (
            not silent
            and generation_number
            % cfg["print_every"] == 0
        ):
            print(
                f"Generation: {generation_number:4d}, "
                f"fitness: {best_fitness:.3f}"
            )

        elif (
            silent
            and nds
            and generation_number % nds == 0
        ):
            print(".", end="", flush=True)

        if (
            best_fitness < cfg["extend_threshold"]
            and not limit_extended
        ):
            generation_limit += cfg["extend_by"]
            limit_extended = True

        generation_number += 1

    if silent and nds:
        print()

    elapsed = time.perf_counter() - start_time
    converged = best_fitness <= cfg["target"]
    best_network = {
        **data["networks"][0],
        **prune_unconnected_genes(data["networks"][0]),
    }
    if population_file:
        with open(population_file, "a") as file:
            for generation, fitness in population_history:
                file.write(
                    f"{trial_index},"
                    f"{generation},"
                    f"{fitness:.6f}\n"
                )

    return {
        "trial": trial_index,
        "seed": random_seed,
        "target_pattern": cfg["target_pattern"],
        "excluded_logic_gates": cfg["exclude_logic_gates"],
        "logic_gates": cfg["logic_gates"],
        "converged": converged,
        "generations_used": generation_number - 1,
        "elapsed_seconds": elapsed,
        "final_fitness": best_fitness,
        "best_network": best_network,
        "pop_history": population_history,
    }


def run_refinement_attempt(
    cfg,
    seed_network,
    pool,
    target_pattern,
    refinement_index,
    silent,
):
    random_seed = get_random_seed(cfg)

    print()
    print(f"Refinement: {refinement_index + 1}")
    print("-" * 20)
    print(f"Seed: {random_seed}")
    print(
        f"Starting network: "
        f"{len(seed_network['genes'])} genes, "
        f"{len(seed_network['edges'])} edges"
    )

    result = run_refinement(
        seed_network=seed_network,
        target_pattern=target_pattern,
        pool=pool,
        random_seed=random_seed,
        target=cfg["refine_target"],
        max_generations=cfg["refine_max_generations"],
        extend_threshold=cfg[
            "refine_extend_threshold"
        ],
        extend_by=cfg["refine_extend_by"],
        elite_count=cfg["refine_elite_percent"],
        print_every=cfg["print_every"],
        silent=silent,
        logic_gates=cfg["logic_gates"],
    )
    status = (
        "success"
        if result["converged"]
        else "failure"
    )

    output_name = (
        f"{cfg['prefix']}refine_success.json"
    )

    output_path = os.path.join(
        cfg["results_dir"],
        output_name,
    )

    output = {
        "refinement": refinement_index,
        "seed": result["seed"],
        "target_pattern": cfg["target_pattern"],
        "excluded_logic_gates": cfg["exclude_logic_gates"],
        "logic_gates": cfg["logic_gates"],
        "converged": result["converged"],
        "generations_used": result[
            "generations_used"
        ],
        "elapsed_seconds": result[
            "elapsed_seconds"
        ],
        "final_fitness": result["final_fitness"],
        "raw_sse": result["raw_sse"],
        "network_size": result["network_size"],
        "best_network": result["best_network"],
    }

    if result["converged"]:
        with open(output_path, "w") as file:
            json.dump(output, file, indent=2)

        diagram_network(output_path, os.path.splitext(output_path)[0] + "_diagram.png")
        visualize_network(output_path, os.path.splitext(output_path)[0] + "_visualize.png")
        saveAntimony(output_path, os.path.splitext(output_path)[0] + "_antimony.txt")


    print(
        f"Refinement [{status}] "
        f"fitness={result['final_fitness']:.3f} "
        f"raw_sse={result['raw_sse']:.3f} "
        f"size={result['network_size']} "
        f"generations={result['generations_used']} "
        f"time={result['elapsed_seconds']:.2f}s"
    )

    if result["converged"]:
        print(f"Written to: {output_path}")

    return output

def print_configuration(cfg, config_path):
    print(f"Network evolver version: {VERSION}")
    print("-" * 50)

    if os.path.exists(config_path):
        print(".....Configuration file loaded")
    else:
        print(
            ".....No configuration file found, "
            "using defaults"
        )

    print(f".....Mode: {cfg['mode']}")
    print(f".....Target pattern: {cfg['target_pattern']}")
    print(f".....Excluded logic gates: {','.join(cfg['exclude_logic_gates']) if cfg['exclude_logic_gates'] else 'NONE'}")
    print(f".....Enabled logic gates: {','.join(cfg['logic_gates']) if cfg['logic_gates'] else 'NONE'}")
    print(f".....Results directory: {cfg['results_dir']}/")
    print(f".....Workers: {cfg['workers']}")

    if cfg["mode"] == "evolve":
        if cfg["target_successes"] is not None:
            print(
                ".....Run until successes: "
                f"{cfg['target_successes']}"
            )
        else:
            print(
                f".....Fixed trials: {cfg['trials']}"
            )

        print(
            ".....Refine successful networks: "
            f"{cfg['refine']}"
        )

    else:
        print(
            ".....Refinement seed: "
            f"{cfg['refine_seed_path']}"
        )

    if cfg["seed"] is None:
        print(".....Random seed: random per attempt")
    else:
        print(
            f".....Random seed: fixed ({cfg['seed']})"
        )

    print()
    print(
        "Current time = "
        f"{datetime.now().strftime('%m/%d/%Y %I:%M:%S %p')}"
    )


def run_refine_only(
    cfg,
    pool,
    target_pattern,
    silent,
):
    seed_network = load_seed_network(
        cfg["refine_seed_path"]
    )

    return run_refinement_attempt(
        cfg=cfg,
        seed_network=seed_network,
        pool=pool,
        target_pattern=target_pattern,
        refinement_index=0,
        silent=silent,
    )


def run_evolution_mode(
    cfg,
    args,
    pool,
    target_pattern,
):
    all_results = []
    all_refinement_results = []

    success_count = 0
    trial_index = 0

    summary_path = os.path.join(
        cfg["results_dir"],
        f"{cfg['prefix']}success_summary.json",
    )

    refinement_summary_path = os.path.join(
        cfg["results_dir"],
        f"{cfg['prefix']}refine_summary.json",
    )

    population_file = None

    if args.pop:
        population_file = os.path.join(
            cfg["results_dir"],
            f"{cfg['prefix']}{args.pop}",
        )

        if os.path.exists(population_file):
            os.remove(population_file)

    while True:
        if (
            cfg["trials"] is not None
            and trial_index >= cfg["trials"]
        ):
            break

        if (
            cfg["target_successes"] is not None
            and success_count
            >= cfg["target_successes"]
        ):
            break

        print()
        print(f"Run: {trial_index + 1}")
        print("-" * 20)

        random_seed = get_random_seed(cfg)

        result = run_evolution_trial(
            cfg=cfg,
            trial_index=trial_index,
            pool=pool,
            random_seed=random_seed,
            target_pattern=target_pattern,
            silent=args.silent,
            nds=args.nds,
            population_file=None,
        )

        status = (
            "success"
            if result["converged"]
            else "failure"
        )

        print(
            f"End of run: {trial_index+1} "
            f"[{status}] "
            f"fitness={result['final_fitness']:.3f} "
            f"generations={result['generations_used']} "
            f"time={result['elapsed_seconds']:.2f}s"
        )

        if result["converged"]:
            success_count += 1
            all_results.append(result)
            success_dir = os.path.join(cfg["results_dir"], f"generationSuccess_{success_count}")
            os.makedirs(success_dir, exist_ok=True)
            network_path = os.path.join(success_dir, f"{cfg['prefix']}success.json")
            output_base = os.path.splitext(network_path)[0]
            image_path_diagram = f"{output_base}_diagram.png"
            image_path_visualize = f"{output_base}_visualize.png"
            txt_path_antimony = f"{output_base}_antimony.txt"

            with open(network_path, "w") as file:
                json.dump(result, file, indent=2)

            diagram_network(network_path, image_path_diagram)
            visualize_network(network_path, image_path_visualize)
            saveAntimony(network_path, txt_path_antimony)
            with open(summary_path, "w") as file:
                json.dump(all_results, file, indent=2)

            if population_file:
                with open(population_file, "a") as file:
                    for generation, fitness in result["pop_history"]:
                        file.write(f"{success_count},{generation},{fitness:.6f}\n")

            if cfg["refine"]:
                seed_network = extract_seed_network(
                    result["best_network"]
                )

                refinement_result = (
                    run_refinement_attempt(
                        cfg={**cfg, "results_dir": success_dir},
                        seed_network=seed_network,
                        pool=pool,
                        target_pattern=target_pattern,
                        refinement_index=success_count - 1,
                        silent=args.silent,
                    )
                )

                if refinement_result["converged"]:
                    all_refinement_results.append(refinement_result)

                    with open(refinement_summary_path, "w") as file:
                        json.dump(all_refinement_results, file, indent=2)

        trial_index += 1

    converged_count = success_count

    if trial_index:
        success_percentage = (
            100
            * converged_count
            / trial_index
        )
    else:
        success_percentage = 0.0

    print()
    print("-" * 30)
    print(f"Trials complete: {trial_index}")
    print(
        f"Converged: {converged_count} "
        f"({success_percentage:.1f}%)"
    )

    if cfg["refine"]:
        refine_success_count = sum(
            1
            for result in all_refinement_results
            if result["converged"]
        )

        print(
            "Refinements complete: "
            f"{len(all_refinement_results)}"
        )

        print(
            "Refinements converged: "
            f"{refine_success_count}"
        )

    print(
        f"Results written to: "
        f"{cfg['results_dir']}/"
    )


def main():
    args = build_parser().parse_args()

    cfg = load_config(args.config)
    cfg = apply_command_line_arguments(cfg, args)

    if args.interactive or len(sys.argv) == 1:
        cfg = run_interactive_setup(cfg)

    cfg = validate_config(cfg)

    if args.printDefaults:
        for key, value in cfg.items():
            print(f"{key:28s} = {value}")
        return

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg["results_dir"] = os.path.join(cfg["results_dir"], run_timestamp)
    os.makedirs(cfg["results_dir"], exist_ok=True)

    os.makedirs(
        "generations",
        exist_ok=True,
    )

    os.makedirs(
        "generationRefine",
        exist_ok=True,
    )

    target_pattern = get_target_pattern(
        cfg["target_pattern"]
    )

    print_configuration(
        cfg,
        args.config,
    )

    pool = mp.Pool(
        processes=cfg["workers"]
    )

    try:
        if cfg["mode"] == "refine_only":
            run_refine_only(
                cfg=cfg,
                pool=pool,
                target_pattern=target_pattern,
                silent=args.silent,
            )

        else:
            run_evolution_mode(
                cfg=cfg,
                args=args,
                pool=pool,
                target_pattern=target_pattern,
            )

    finally:
        pool.close()
        pool.join()
if __name__ == "__main__":
    main()
