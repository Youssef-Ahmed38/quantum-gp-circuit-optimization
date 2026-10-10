"""Minimal IBM Quantum free-tier protocol for C2 (H1 calibration, H2 holdout).

DEFAULT IS A DRY RUN: circuits are built, checked for ideal identity, hashed,
and the QPU-time budget is estimated. Nothing is sent to IBM unless you run

    python benchmarks/ibm_dd_protocol.py submit-h1 --confirm-qpu-use
    python benchmarks/ibm_dd_protocol.py submit-h2 --confirm-qpu-use

with ``QISKIT_IBM_TOKEN`` set and ``qiskit-ibm-runtime`` installed.

H1  CPMG spectroscopy (|+>, n in {0,2,4,8}, w in {4,16,40,80} us) and T1 (|1>)
    on three qubits in parallel; 3 randomized blocks x 1,000 shots.
H2  GHZ-5 with a 20 us idle layer, then un-preparation; policies: no DD,
    Qiskit-like XX, Qiskit-like XY4 (as X/Y CPMG), GP-DD from the frozen H1 fit,
    and GP-DD with cross-window protection; 3 randomized blocks x 2,000 shots.
    Primary endpoint: P(000000).
The H2 schedule file is SHA-256 hashed before submission (pre-registration).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator

from qcgp.ddgp import DDProblem, IdleModel, alap, fit_idle_model, fixed_pattern, solve_dd_allocation
from qcgp.hardware import schedule_to_circuit
from qcgp.noisesim import gate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "ibm_dd_protocol"
REP_DELAY_US = 250.0
WINDOWS = [4.0, 16.0, 40.0, 80.0]
COUNTS = [0, 2, 4, 8]
TP = 0.036
SUF = ""
MAX_PULSES = max(COUNTS)  # never extrapolate beyond the pulse counts calibrated in H1
CX_NATIVE = 0.140  # H-CZ-H on Heron: 36 + 68 + 36 ns


def h1_circuits(qubits=3):
    circuits, labels = [], []
    for w in WINDOWS:
        for n in COUNTS:
            c = QuantumCircuit(qubits, qubits, name=f"cpmg_w{w:g}_n{n}")
            c.h(range(qubits))
            for q in range(qubits):
                cursor = 0.
                for i in range(n):
                    centre = w * (2 * i + 1) / (2 * n)
                    c.delay(centre - TP / 2 - cursor, q, unit="us")
                    c.x(q)
                    cursor = centre + TP / 2
                c.delay(w - cursor, q, unit="us")
            c.h(range(qubits))
            c.measure(range(qubits), range(qubits))
            circuits.append(c)
            labels.append(dict(kind="cpmg", window=w, pulses=n))
        c = QuantumCircuit(qubits, qubits, name=f"t1_w{w:g}")
        c.x(range(qubits))
        c.delay(w, range(qubits), unit="us")
        c.x(range(qubits))
        c.measure(range(qubits), range(qubits))
        circuits.append(c)
        labels.append(dict(kind="t1", window=w, pulses=0))
    return circuits, labels


def h2_problem(idle_models):
    """GHZ-5 on qubits 0-4, parity copy to ancilla 5, a 20 us ancilla wait
    (measurement / feed-forward latency stand-in), uncompute. The data qubits
    idle while the ancilla waits because the uncompute chain depends on it."""
    gates = [gate("h", 0)] + [gate("cx", i, i + 1) for i in range(4)]
    gates += [gate("cx", 4, 5), gate("block", 5), gate("cx", 4, 5)]
    gates += [gate("cx", i, i + 1) for i in reversed(range(4))] + [gate("h", 0)]
    timed = [g for g in gates if g.timed]
    base = np.array([20.0 if g.block else (TP if len(g.qubits) == 1 else CX_NATIVE) for g in timed])
    models = list(idle_models)[:5]
    models += [models[-1]] * (6 - len(models))
    return DDProblem(6, tuple(gates), base, base.copy(), base.copy(), np.zeros(len(base)), np.zeros(len(base)),
                     tuple(models), TP)


def strip_measure(c):
    d = c.remove_final_measurements(inplace=False)
    return QuantumCircuit.from_instructions([i for i in d.data if i.operation.name != "delay"],
                                            qubits=d.qubits)


def budget(circuits, shots, blocks):
    durations = []
    for c in circuits:
        total = 0.
        for q in range(c.num_qubits):
            t = sum(float(i.operation.duration) for i in c.data
                    if i.operation.name == "delay" and c.find_bit(i.qubits[0]).index == q)
            total = max(total, t)
        durations.append(total)
    return float(sum((d + REP_DELAY_US) * shots * blocks for d in durations) * 1e-6)


def dry_run_h1():
    circuits, labels = h1_circuits()
    for c in circuits:
        if not Operator(strip_measure(c)).equiv(Operator(QuantumCircuit(c.num_qubits))):
            raise AssertionError(f"{c.name} is not the identity")
    seconds = budget(circuits, 1000, 3)
    return circuits, labels, seconds


def h2_policies(idle_models):
    problem = h2_problem(idle_models)
    from qcgp.ddgp import asap
    d, s, M = asap(problem)
    policies = {
        "no_dd": (d, s, M, np.zeros(len(problem.windows()))),
        "xx_qiskit_like": (d, s, M, fixed_pattern(problem, d, s, M, 2)),
        "xy4_like": (d, s, M, fixed_pattern(problem, d, s, M, 4)),
    }
    gp = solve_dd_allocation(problem, M, optimize_durations=False, allow_dd=True, max_pulses=MAX_PULSES)
    policies["gp_dd"] = (gp["durations"], gp["starts"], gp["makespan"], gp["pulses"])
    from qcgp.ddgp import correlated_protection
    policies["gp_dd_correlated"] = (gp["durations"], gp["starts"], gp["makespan"],
                                    correlated_protection(problem, gp["durations"], gp["starts"], gp["makespan"],
                                                          max_pulses=MAX_PULSES))
    circuits = {k: schedule_to_circuit(problem, *v) for k, v in policies.items()}
    for name, c in circuits.items():
        c.name = name
        if not Operator(strip_measure(c)).equiv(Operator(QuantumCircuit(c.num_qubits))):
            raise AssertionError(f"{name} is not the identity")
    return problem, policies, circuits


def h3_policies(idle_models):
    """Control arms isolating what the GP decided (pre-registered as H3).

    gp_dd: the H2 GP schedule, unchanged.
    cpmg8_uniform: 8 pulses in every window where they fit (matches the GP's per-window
        count on protected windows, so it controls for "more pulses").
    gp_dd_protect_q54: the GP schedule with qubit 54 (logical qubit 1) also given the
        uniform count, isolating the decision to leave that qubit undecoupled.
    xy4_like: the best standard schedule in H2 and R1 (common reference).
    """
    problem, h2, _ = h2_policies(idle_models)
    d, s, M, gp_counts = h2["gp_dd"]
    uniform = fixed_pattern(problem, d, s, M, MAX_PULSES)
    protect = np.array(gp_counts, dtype=float).copy()
    for k, (q, a, b) in enumerate(problem.windows()):
        if q == 1:
            protect[k] = uniform[k]
    policies = {"gp_dd": h2["gp_dd"], "cpmg8_uniform": (d, s, M, uniform),
                "gp_dd_protect_q54": (d, s, M, protect), "xy4_like": h2["xy4_like"]}
    circuits = {k: schedule_to_circuit(problem, *v) for k, v in policies.items()}
    for name, c in circuits.items():
        c.name = name
        if not Operator(strip_measure(c)).equiv(Operator(QuantumCircuit(c.num_qubits))):
            raise AssertionError(f"{name} is not the identity")
    return problem, policies, circuits


def placeholder_models():
    """Heron-like idle models for dry runs only (replace with the frozen H1 fit)."""
    return [IdleModel(2e-4, 1.0, 1 / 300, 3e-4) for _ in range(6)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["dry-run", "fit-h1", "submit-h1", "submit-h2", "submit-h3", "fetch"])
    parser.add_argument("--job-file", help="job record JSON for fetch")
    parser.add_argument("--counts", help="H1 counts JSON for fit-h1")
    parser.add_argument("--confirm-qpu-use", action="store_true")
    parser.add_argument("--backend", default="least-busy")
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--tag", default="", help="suffix for replication job records, e.g. r1")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--exclude", default="", help="comma-separated physical qubits to avoid (second chain)")
    args = parser.parse_args()
    global SUF
    SUF = f"-{args.tag}" if args.tag else ""
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "dry-run":
        circuits, labels, h1_seconds = dry_run_h1()
        frozen = OUT / f"h1_frozen_fit{SUF}.json"
        models = ([IdleModel(**m) for m in json.loads(frozen.read_text())["models"]]
                  if frozen.exists() else placeholder_models())
        problem, policies, h2 = h2_policies(models)
        h2_seconds = budget(list(h2.values()), 2000, 3)
        with open(OUT / f"h2_circuits_dry_run{SUF}.qpy", "wb") as handle:
            qpy.dump(list(h2.values()), handle)
        digest = hashlib.sha256((OUT / f"h2_circuits_dry_run{SUF}.qpy").read_bytes()).hexdigest()
        report = dict(h1_circuits=len(circuits), h1_estimated_qpu_seconds=h1_seconds,
                      h2_circuits=len(h2), h2_estimated_qpu_seconds=h2_seconds,
                      h2_gp_pulses=[int(x) for x in policies["gp_dd"][3]],
                      h2_xx_pulses=int(np.sum(policies["xx_qiskit_like"][3])),
                      h2_sha256=digest, models="frozen H1 fit" if frozen.exists() else "placeholder (no H1 data yet)",
                      submitted=False)
        (OUT / f"dry_run{SUF}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return
    if args.command == "fit-h1":
        data = json.loads(Path(args.counts).read_text())
        models, diagnostics = [], []
        for q in range(3):
            rows = [dict(window=r["window"], pulses=r["pulses"],
                         hazard=-np.log(max(2 * r["p0"][q] - 1, 1e-9))) for r in data
                    if r["kind"] == "cpmg" and r["pulses"] > 0]
            # Free precession oscillates under a coherent offset; keep n = 0 points only up to
            # the first zero crossing (contrast falling and above 0.2), where a hazard is defined.
            previous = 1.
            for r in sorted((r for r in data if r["kind"] == "cpmg" and r["pulses"] == 0), key=lambda r: r["window"]):
                contrast = 2 * r["p0"][q] - 1
                if contrast <= 0.2 or contrast >= previous:
                    break
                rows.append(dict(window=r["window"], pulses=0, hazard=-np.log(contrast)))
                previous = contrast
            model, cost, spam = fit_idle_model(rows, spam_offset=True, static_term=True)
            # T1: slope of log survival across windows (the intercept absorbs SPAM).
            t1 = [(r["window"], np.log(max(r["p0"][q], 1e-9))) for r in data if r["kind"] == "t1"]
            rate = float(max(-np.polyfit([x for x, _ in t1], [y for _, y in t1], 1)[0], 0.)) if len(t1) > 1 else 0.
            model = replace(model, markov=max(model.markov - rate / 2, 0.), relaxation=rate)
            models.append(asdict(model))
            diagnostics.append(dict(qubit=q, log_residual_cost=cost, spam_hazard=spam, t1_us=1 / rate if rate else None))
        mean = {k: float(np.mean([m[k] for m in models])) for k in models[0]}
        models = models + [dict(mean), dict(mean), dict(mean)]
        # Pulse-error floor: backend-reported X error on each H2 physical qubit (independent calibration).
        h1 = json.loads((OUT / f"submit-h1{SUF}_job.json").read_text())
        backend = runtime_service().backend(h1["backend"])
        layout = h1.get("chain6") or extend_layout(backend, h1["layout"], 6)
        floors = [float(backend.target["x"][(p,)].error or 0.) for p in layout]
        for m, floor in zip(models, floors):
            m["pulse_error"] = max(m["pulse_error"], floor)
        payload = dict(models=models, diagnostics=diagnostics, source=args.counts, h2_layout=layout,
                       x_error_floors=floors, max_pulses=MAX_PULSES,
                       note=("H2 qubits 3-5 use the mean of the three fitted H1 qubits; pulse error is floored at "
                             "the backend-reported X error; pulse counts are capped at the H1-calibrated maximum"))
        (OUT / f"h1_frozen_fit{SUF}.json").write_text(json.dumps(payload, indent=2))
        print("Frozen fit written; rerun dry-run to build and hash H2.")
        return
    if args.command == "fetch":
        fetch(args.job_file)
        return
    if not args.confirm_qpu_use:
        raise SystemExit("Refusing to submit without --confirm-qpu-use")
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2
    service = runtime_service()
    backend = least_busy(service) if args.backend == "least-busy" else service.backend(args.backend)
    if args.command == "submit-h1":
        circuits, labels, _ = dry_run_h1()
        shots, blocks = 1000, 3
    elif args.command == "submit-h3":
        frozen = OUT / f"h1_frozen_fit{SUF}.json"
        _, _, h3 = h3_policies([IdleModel(**m) for m in json.loads(frozen.read_text())["models"]])
        circuits, labels = list(h3.values()), list(h3)
        shots, blocks = 6000, 4
    else:
        frozen = OUT / f"h1_frozen_fit{SUF}.json"
        if not frozen.exists():
            raise SystemExit("H2 requires the frozen H1 fit (pre-registration)")
        _, _, h2 = h2_policies([IdleModel(**m) for m in json.loads(frozen.read_text())["models"]])
        circuits, labels = list(h2.values()), list(h2)
        shots, blocks = 2000, 3
    shots = args.shots or shots
    blocks = args.blocks or blocks
    rng = np.random.default_rng(args.seed)
    batch = [(block, int(index)) for block in range(blocks) for index in rng.permutation(len(circuits))]
    exclude = {int(x) for x in args.exclude.split(",") if x.strip()}
    chain6 = choose_layout(backend, 6, exclude)
    layout = chain6[:circuits[0].num_qubits]
    if args.command in ("submit-h2", "submit-h3"):
        h1 = json.loads((OUT / f"submit-h1{SUF}_job.json").read_text())
        if h1["backend"] != backend.name:
            raise SystemExit(f"H2 must run on the H1 backend {h1['backend']}")
        layout = h1.get("chain6") or extend_layout(backend, h1["layout"], circuits[0].num_qubits)
    compiled = transpile([circuits[i] for _, i in batch], backend, optimization_level=0,
                         initial_layout=layout, scheduling_method="alap")
    job = SamplerV2(mode=backend).run(compiled, shots=shots)
    record = dict(job_id=job.job_id(), command=args.command, order=batch, labels=labels, backend=backend.name,
                  layout=layout, shots=shots, submitted=time.strftime("%Y-%m-%dT%H:%M:%S"))
    record["blocks"] = blocks
    if args.command == "submit-h1":
        record["chain6"] = chain6
    suffix = f"-{args.tag}" if args.tag else ""
    (OUT / f"{args.command}{suffix}_job.json").write_text(json.dumps(record, indent=2, default=str))
    print("Submitted", job.job_id(), "on", backend.name, "layout", layout)


def runtime_service():
    """Saved account (preferred) or QISKIT_IBM_TOKEN; the token never appears in code or logs."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    if "QISKIT_IBM_TOKEN" in os.environ:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["QISKIT_IBM_TOKEN"])
    return QiskitRuntimeService()


def least_busy(service):
    return service.least_busy(operational=True, simulator=False)


def choose_layout(backend, width, exclude=frozenset()):
    """A connected chain of physical qubits with the lowest summed readout error."""
    props = backend.target
    coupling = backend.coupling_map
    best, best_score = None, None
    for start in (q for q in range(backend.num_qubits) if q not in exclude):
        path, current = [start], start
        while len(path) < width:
            options = [q for q in coupling.neighbors(current) if q not in path and q not in exclude]
            if not options:
                break
            current = min(options)
            path.append(current)
        if len(path) < width:
            continue
        score = 0.
        for q in path:
            entry = props["measure"].get((q,)) if "measure" in props else None
            score += entry.error if entry is not None and entry.error is not None else 1.
        if best_score is None or score < best_score:
            best, best_score = path, score
    return best


def extend_layout(backend, seed, width):
    """Keep the calibrated H1 qubits first and extend the chain through unused neighbours."""
    path = list(seed)
    while len(path) < width:
        options = [q for q in backend.coupling_map.neighbors(path[-1]) if q not in path]
        if not options:
            raise SystemExit("Cannot extend the H1 chain to the H2 width")
        path.append(min(options))
    return path


def fetch(job_file):
    """Download results and aggregate per-circuit marginal P(0) for every qubit."""
    record = json.loads(Path(job_file).read_text())
    service = runtime_service()
    job = service.job(record["job_id"])
    status = str(job.status())
    if "DONE" not in status.upper():
        print("Job status:", status)
        return
    result = job.result()
    totals = {}
    for position, (block, index) in enumerate(record["order"]):
        counts = result[position].data.c.get_counts()
        bucket = totals.setdefault(index, {})
        for key, value in counts.items():
            bucket[key] = bucket.get(key, 0) + value
    rows = []
    for index, counts in sorted(totals.items()):
        shots = sum(counts.values())
        width = len(next(iter(counts)))
        p0 = [sum(v for k, v in counts.items() if k[::-1][q] == "0") / shots for q in range(width)]
        label = record["labels"][index]
        row = dict(label) if isinstance(label, dict) else dict(policy=label)
        row.update(p0=p0, shots=shots, all_zero=counts.get("0" * width, 0) / shots, counts=counts)
        rows.append(row)
    out = Path(job_file).with_name(Path(job_file).stem.replace("_job", "") + "_counts.json")
    out.write_text(json.dumps(rows, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
