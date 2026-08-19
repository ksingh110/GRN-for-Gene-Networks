import argparse
import configparser
import json
import os
import sys
import time
import random
import multiprocessing as mp
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generationCreation import compute_generation_initial, generationCreate
from test import visualize_network

VERSION = "0.1.0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

DEFAULTS = {
    "target_successes": 1,
    "trials": None,
    "target": 100.0,
    "max_generations": 500,
    "extend_threshold": 750.0,
    "extend_by": 300,
    "elite_percent": 10,
    "results_dir": "results",
    "prefix": "",
    "print_every": 50,
    "workers": max(1, os.cpu_count()-1),
    "seed": None,
}


def load_config(path):
    cfg = dict(DEFAULTS)
    if path and os.path.exists(path):
        parser = configparser.ConfigParser()
        parser.read(path)
        if "network_evolver" in parser:
            section = parser["network_evolver"]
            cfg["target_successes"] = section.getint("target_successes", cfg["target_successes"])
            trials_raw = section.get("trials", "")
            cfg["trials"] = int(trials_raw) if trials_raw.strip() else None
            cfg["target"] = section.getfloat("target", cfg["target"])
            cfg["max_generations"] = section.getint("max_generations", cfg["max_generations"])
            cfg["extend_threshold"] = section.getfloat("extend_threshold", cfg["extend_threshold"])
            cfg["extend_by"] = section.getint("extend_by", cfg["extend_by"])
            cfg["elite_percent"] = section.getint("elite_percent", cfg["elite_percent"])
            cfg["results_dir"] = section.get("results_dir", cfg["results_dir"])
            cfg["prefix"] = section.get("prefix", cfg["prefix"])
            cfg["print_every"] = section.getint("print_every", cfg["print_every"])
            workers_raw = section.get("workers", "")
            cfg["workers"] = int(workers_raw) if workers_raw.strip() else cfg["workers"]
            seed_raw = section.get("seed", "")
            cfg["seed"] = int(seed_raw) if seed_raw.strip() else None
    return cfg


def build_parser():
    p = argparse.ArgumentParser(
        prog="network_evolver",
    )
    p.add_argument("--config", default="config.ini", help="Path to config file (default: config.ini)")
    p.add_argument("-v", "--verbose", action="store_true", help="Make output more verbose")
    p.add_argument("-silent", action="store_true", help="Silent mode (only final results printed)")
    p.add_argument("-nds", type=int, default=None, help="Number of generations between progress dots in silent mode")
    p.add_argument("-pop", type=str, default=None, help="Output data to filename describing best-fitness per generation")
    p.add_argument("-w", action="store_true", help="Wait for the user to hit a key before exiting")
    p.add_argument("--successes", type=int, default=None, help="Stop once this many trials converge (default: 1)")
    p.add_argument("--trials", type=int, default=None, help="Run exactly this many trials instead of running until N successes")
    p.add_argument("--target", type=float, default=None, help="Fitness threshold to count as converged")
    p.add_argument("--max-generations", type=int, default=None, help="Base generation limit per trial")
    p.add_argument("--elite-percent", type=int, default=None, help="Number of elites carried over each generation")
    p.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: cpu_count - 1)")
    p.add_argument("--seed", type=int, default=None, help="Fixed random seed for every trial (omit for a fresh random seed each trial)")
    p.add_argument("-printDefaults", action="store_true", help="List the config values and exit")
    p.add_argument("-i", "--interactive", action="store_true", help="Prompt for each config value before running")
    return p


def prompt_int(label, default):
    raw = input(f"New value for {label} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  Not a valid integer, keeping {default}")
        return default


def prompt_float(label, default):
    raw = input(f"New value for {label} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  Not a valid number, keeping {default}")
        return default


def prompt_str(label, default):
    raw = input(f"New value for {label} [{default}]: ").strip()
    return raw if raw != "" else default


MENU_ITEMS = [
    ("Run mode (successes/trials)", "_run_mode", "str"),
    ("Target successes (if mode=successes)", "target_successes", "int"),
    ("Number of trials (if mode=trials)", "trials", "int"),
    ("Seed mode (random/fixed)", "_seed_mode", "str"),
    ("Fixed seed value (if mode=fixed)", "seed", "int"),
    ("Fitness target", "target", "float"),
    ("Max generations per trial", "max_generations", "int"),
    ("Extend threshold", "extend_threshold", "float"),
    ("Extend by (generations)", "extend_by", "int"),
    ("Elite percent", "elite_percent", "int"),
    ("CPU workers", "workers", "int"),
    ("Results directory", "results_dir", "str"),
    ("Print every N generations", "print_every", "int"),
]


def _run_mode_display(cfg):
    return "successes" if cfg["target_successes"] is not None else "trials"


def _seed_mode_display(cfg):
    return "fixed" if cfg["seed"] is not None else "random"


def run_interactive_setup(cfg):
    if cfg["target_successes"] is None and cfg["trials"] is None:
        cfg["target_successes"] = 1

    while True:
        for idx, (label, key, _) in enumerate(MENU_ITEMS, start=1):
            if key == "_run_mode":
                value = _run_mode_display(cfg)
            elif key == "_seed_mode":
                value = _seed_mode_display(cfg)
            else:
                value = cfg[key]
                if value is None:
                    value = "(unused)"
            print(f" {idx:2d}) {label:36s}: {value}")
        print("-" * 55)
        print(" R) Run with these settings")
        print(" Q) Quit")
        print("-" * 55)

        choice = input("Select an option: ").strip().lower()

        if choice == "r":
            print()
            print("Config set. Starting run...")
            print()
            return cfg
        if choice == "q":
            print("Exiting.")
            sys.exit(0)

        try:
            item_idx = int(choice) - 1
            if item_idx < 0 or item_idx >= len(MENU_ITEMS):
                raise ValueError
        except ValueError:
            print("  Not a valid option, try again.")
            continue

        label, key, kind = MENU_ITEMS[item_idx]

        if key == "_run_mode":
            new_mode = prompt_str("run mode ('successes' or 'trials')", _run_mode_display(cfg))
            if new_mode.strip().lower().startswith("t"):
                cfg["trials"] = cfg["trials"] or 10
                cfg["target_successes"] = None
            else:
                cfg["target_successes"] = cfg["target_successes"] or 1
                cfg["trials"] = None
            continue

        if key == "_seed_mode":
            new_mode = prompt_str("seed mode ('random' or 'fixed')", _seed_mode_display(cfg))
            if new_mode.strip().lower().startswith("f"):
                cfg["seed"] = cfg["seed"] if cfg["seed"] is not None else random.randrange(2**32)
            else:
                cfg["seed"] = None
            continue

        if kind == "int":
            cfg[key] = prompt_int(label, cfg[key])
        elif kind == "float":
            cfg[key] = prompt_float(label, cfg[key])
        else:
            cfg[key] = prompt_str(label, cfg[key])


def run_trial(cfg, trial_idx, pool, seed, silent, nds, pop_file):
    random.seed(seed)
    print("Seed: " + str(seed))
    compute_generation_initial(1, mp_pool=pool)
    with open("generations_2/generation_1.json") as f:
        data = json.load(f)

    best_fitness = data["networks"][0]["fitness"]["score"]
    iteration = 1
    limit = cfg["max_generations"]
    limit_extended = False
    start_time = time.perf_counter()
    pop_history = [(0, best_fitness)]

    while best_fitness > cfg["target"] and iteration <= limit:
        data = generationCreate(cfg["elite_percent"], data, iteration, mp_pool=pool)
        new_best = data["networks"][0]["fitness"]["score"]
        best_fitness = min(best_fitness, new_best)
        pop_history.append((iteration, best_fitness))

        if not silent and iteration % cfg["print_every"] == 0:
            print(f"Generation: {iteration:4d}, fitness: {best_fitness:.3f}")
        elif silent and nds and iteration % nds == 0:
            print(".", end="", flush=True)

        if best_fitness < cfg["extend_threshold"] and not limit_extended:
            limit += cfg["extend_by"]
            limit_extended = True

        iteration += 1

    if silent and nds:
        print()

    elapsed = time.perf_counter() - start_time
    converged = best_fitness <= cfg["target"]

    if pop_file:
        with open(pop_file, "a") as f:
            for gen, fit in pop_history:
                f.write(f"{trial_idx},{gen},{fit:.6f}\n")

    return {
        "trial": trial_idx,
        "seed": seed,
        "converged": converged,
        "generations_used": iteration - 1,
        "elapsed_seconds": elapsed,
        "final_fitness": best_fitness,
        "best_network": data["networks"][0],
        "pop_history": pop_history,
    }


def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    if args.trials is not None:
        cfg["trials"] = args.trials
        cfg["target_successes"] = None
    if args.successes is not None:
        cfg["target_successes"] = args.successes
        cfg["trials"] = None
    if args.target is not None:
        cfg["target"] = args.target
    if args.max_generations is not None:
        cfg["max_generations"] = args.max_generations
    if args.elite_percent is not None:
        cfg["elite_percent"] = args.elite_percent
    if args.workers is not None:
        cfg["workers"] = args.workers
    if args.seed is not None:
        cfg["seed"] = args.seed

    if args.printDefaults:
        for k, v in cfg.items():
            print(f"{k:20s} = {v}")
        return

    if args.interactive or len(sys.argv) == 1:
        cfg = run_interactive_setup(cfg)

    if cfg["trials"] is None and cfg["target_successes"] is None:
        cfg["target_successes"] = 1

    silent = args.silent
    os.makedirs(cfg["results_dir"], exist_ok=True)
    os.makedirs("generations_2", exist_ok=True)

    n_workers = cfg["workers"]

    run_mode_desc = (
        f"until {cfg['target_successes']} success(es)" if cfg["target_successes"] is not None
        else f"fixed {cfg['trials']} trials"
    )
    seed_mode_desc = f"fixed ({cfg['seed']})" if cfg["seed"] is not None else "random per trial"

    print(f"Network evolver version: {VERSION}")
    print("-" * 40)
    print(".....Configuration file loaded" if os.path.exists(args.config) else ".....No configuration file found, using defaults")
    print(f".....Writing results to: {cfg['results_dir']}/")
    print(f".....Run mode: {run_mode_desc}")
    print(f".....Seed mode: {seed_mode_desc}")
    print(f".....Objective: oscillator")
    print(f".....CPU usage: {n_workers} worker process(es) (of {os.cpu_count()} available cores)")
    print()
    print(f"Current time = {datetime.now().strftime('%m/%d/%Y %I:%M:%S %p')}")

    pool = mp.Pool(processes=n_workers)
    all_results = []
    summary_path = os.path.join(cfg["results_dir"], "success_summary.json")
    pop_file = os.path.join(cfg["results_dir"], f"{cfg['prefix']}{args.pop}") if args.pop else None
    if pop_file and os.path.exists(pop_file):
        os.remove(pop_file)

    try:
        trial_idx = 0
        success_count = 0

        while True:
            if cfg["trials"] is not None and trial_idx >= cfg["trials"]:
                break
            if cfg["target_successes"] is not None and success_count >= cfg["target_successes"]:
                break

            print()
            print(f"Run: {trial_idx + 1}")
            print("-" * 20)

            seed = cfg["seed"] if cfg["seed"] is not None else random.randrange(2**32)
            result = run_trial(cfg, trial_idx, pool, seed, silent, args.nds, pop_file)
            all_results.append(result)

            status = "success" if result["converged"] else "fail"
            print(f"End of run: {trial_idx}  [{status}]  fitness={result['final_fitness']:.3f}  "
                  f"generations={result['generations_used']}  time={result['elapsed_seconds']:.2f}s")

            out_name = "success" if result["converged"] else "failure"
            net_path = os.path.join(cfg["results_dir"], f"{cfg['prefix']}{out_name}_{trial_idx}.json")
            with open(net_path, "w") as f:
                json.dump(result["best_network"], f)

            with open(summary_path, "w") as f:
                json.dump(all_results, f, indent=2)

            if result["converged"]:
                traj_path = os.path.join(cfg["results_dir"], f"{cfg['prefix']}trajectory_{trial_idx}.png")
                visualize_network(result["best_network"], save_path=traj_path)

                pop_history = result["pop_history"]
                gens = [g for g, fit in pop_history if g % 50 == 0]
                fits = [fit for g, fit in pop_history if g % 50 == 0]
                if pop_history[-1][0] not in gens:
                    gens.append(pop_history[-1][0])
                    fits.append(pop_history[-1][1])

                plt.figure(figsize=(8, 5))
                plt.plot(gens, fits, marker="o")
                plt.xlabel("Generation")
                plt.ylabel("Best fitness")
                plt.title(f"Trial {trial_idx} — fitness vs. generation")
                plt.tight_layout()
                plt.savefig(os.path.join(cfg["results_dir"], f"{cfg['prefix']}fitness_progress_{trial_idx}.png"), dpi=300)
                plt.close()

                success_count += 1

            trial_idx += 1

    finally:
        pool.close()
        pool.join()

    n_converged = sum(1 for r in all_results if r["converged"])
    print()
    print("-" * 20)
    print(f"Trials complete: {len(all_results)}")
    print(f"Converged: {n_converged} ({100 * n_converged / len(all_results):.1f}%)")
    print(f"Results written to: {cfg['results_dir']}/")

    if args.w:
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()