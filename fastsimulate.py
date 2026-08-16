import numpy as np
from scipy.integrate import solve_ivp

SINGLE_INPUT_LAWS = ["activation", "inhibition"]
TWO_INPUT_LAWS = ["AND", "OR", "NOR", "NAND", "XOR", "EQ"]


class SimResult:
    def __init__(self, colnames, rows):
        self.colnames = colnames
        self._rows = rows

    def __getitem__(self, idx):
        return self._rows[idx]

    def __len__(self):
        return len(self._rows)


def _rate_term(e, y, idx):
    law = e["rate_law"]

    if law == "activation":
        S = max(0.0, y[idx[e["regulator"]]])
        Vf, Ks, n = e["Vf"], e["Ks"], e["n"]
        return (Vf * S ** n) / (Ks + S ** n)

    if law == "inhibition":
        S = max(0.0, y[idx[e["regulator"]]])
        Vf, Ks, n = e["Vf"], e["Ks"], e["n"]
        return Vf / (Ks + S ** n)

    A = max(0.0, y[idx[e["regulator"]]])
    B = max(0.0, y[idx[e["regulator2"]]])

    Vf, K1, K2, K3 = e["Vf"], e["K1"], e["K2"], e["K3"]
    n1, n2 = e["n1"], e["n2"]

    A_n = A ** n1
    B_n = B ** n2

    if law == "AND":
        return Vf * (K1 * K2 * A_n * B_n) / (
            1 + K1 * A_n + K2 * B_n + K1 * K2 * A_n * B_n
        )

    if law == "OR":
        return Vf * (K1 * A_n + K2 * B_n) / (
            1 + K1 * A_n + K2 * B_n
        )

    if law == "NOR":
        return Vf / (
            1 + K1 * A_n + K2 * B_n + K3 * A_n * B_n
        )

    if law == "NAND":
        return Vf * (1 + K1 * A_n + K2 * B_n) / (
            1 + K1 * A_n + K2 * B_n + K3 * A_n * B_n
        )

    if law == "XOR":
        return Vf * (K1 * A_n + K2 * B_n) / (
            1 + K1 * A_n + K2 * B_n + K3 * A_n * B_n
        )

    if law == "EQ":
        return Vf * (1 + K1 * A_n * B_n) / (
            1 + K1 * A_n + K2 * B_n + K3 * A_n * B_n
        )

    raise ValueError(f"unknown rate law: {law}")


def _build_derivative_fn(net, degradation_rates):
    genes = net["genes"]
    idx = {g: i for i, g in enumerate(genes)}

    regs_by_target = {g: [] for g in genes}

    for e in net["edges"]:
        regs_by_target[e["target"]].append(e)

    error_holder = {}

    def dydt(t, y):
        try:
            dy = np.zeros(len(genes))

            for i, g in enumerate(genes):
                regs = regs_by_target[g]

                synth = (
                    sum(_rate_term(e, y, idx) for e in regs)
                    if regs
                    else 0.0
                )

                dy[i] = synth - degradation_rates[i] * y[i]

            return dy

        except Exception as exc:
            error_holder["error"] = exc
            return np.zeros(len(genes))

    return dydt, error_holder


def simulate_network(
    net,
    t_end=50,
    n_points=10,
    rtol=1e-2,
    atol=1e-4
):
    genes = net["genes"]

    y0 = [net["y0"][g] for g in genes]
    degradation_rates = [net["degradation_rates"][g] for g in genes]

    dydt, error_holder = _build_derivative_fn(
        net,
        degradation_rates
    )

    t_eval = np.linspace(
        0,
        t_end,
        n_points
    )

    sol = solve_ivp(
        dydt,
        [0, t_end],
        y0,
        t_eval=t_eval,
        method="LSODA",
        rtol=rtol,
        atol=atol
    )

    if "error" in error_holder:
        raise error_holder["error"]

    if not sol.success:
        raise RuntimeError(
            f"ODE integration failed: {sol.message}"
        )

    colnames = ["time"] + genes

    rows = [
        [sol.t[ti]]
        + [
            sol.y[gi][ti]
            for gi in range(len(genes))
        ]
        for ti in range(len(sol.t))
    ]

    return SimResult(colnames, rows)