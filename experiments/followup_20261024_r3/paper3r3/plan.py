"""Frozen third-round scientific matrix. Pure stdlib on the login node."""
from pathlib import Path
import hashlib
import json

PACKAGE = Path(__file__).resolve().parent.parent
SEEDS = (20261018, 20261019, 20261020)
BUDGETS = (20, 4, 8, 12, 16)
STRESS = ("label_noise10", "label_noise20", "gain10", "gain25", "rotation10", "rotation25")
ADAPTERS = ("sgd10", "sgd100", "head_sgd", "head_newton", "tangent_diag", "tangent_joint",
            "tangent_profiled", "tangent_information", "tangent_protected", "information_protected", "relinearized")
JOINT_BASES = {"profiled": "tangent_profiled", "information": "tangent_information", "protected": "information_protected"}
JOINT_ADAPTERS = ("sgd10", "tangent_profiled", "tangent_information", "information_protected")
COMPARISONS = (
    ("legacy:tangent_profiled", "legacy:sgd10"),
    ("legacy:tangent_profiled", "legacy:sgd100"),
    ("legacy:tangent_profiled", "legacy:head_newton"),
    ("legacy:tangent_profiled", "legacy:tangent_diag"),
    ("legacy:tangent_profiled", "legacy:tangent_joint"),
    ("legacy:tangent_information", "legacy:tangent_profiled"),
    ("legacy:tangent_protected", "legacy:tangent_profiled"),
    ("legacy:information_protected", "legacy:tangent_information"),
    ("profiled:tangent_profiled", "matched_sgd:tangent_profiled"),
    ("protected:information_protected", "matched_sgd:information_protected"),
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _code():
    names = ["algorithm.py", "baseline.py", "data.py", "diagnostics.py", "worker.py"]
    files = [PACKAGE / "paper3r3" / name for name in names] + sorted((PACKAGE / "p3work").glob("*.py"))
    return {str(path.relative_to(PACKAGE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def basis_key(kind, rank, seed):
    return f"basis-{kind}-r{rank}-seed{seed}"


def configuration(variant, *, rank=4, radius=.15, legacy=False, **extra):
    # Every primary optimizer/geometry comparison has exactly the same priors.
    # A separately named original-penalty control is retained, never pooled.
    return {"variant": variant, "rank": rank, "radius": radius, "tangent_steps": 10,
            "theta_l2": .01, "head_l2": .001 if legacy else .1, "damping": 1e-4,
            "information_scale": 1., "kl_epsilon": .01, **extra}


def _basis(kind, rank, seed):
    variant = "sgd10" if kind in ("legacy", "matched_sgd") else JOINT_BASES[kind]
    return {"key": basis_key(kind, rank, seed), "family": "basis", "group": f"basis:{kind}:r{rank}",
            "seed": seed, "variant": variant, "config": configuration(variant, rank=rank, legacy=kind == "legacy"),
            "updates": 1000, "requires": [], "reuse_completed_earlier_P": kind == "legacy"}


def _evaluation(kind, variant, seed, *, family="core", setting="r4-rho0p15", config=None, group=None,
                baseline_group=None, scenarios=("clean",), budgets=BUDGETS):
    config = config or configuration(variant)
    group = group or f"{kind}:{variant}"
    return {"key": f"{family}-{kind}-{group.replace(':', '-')}-{setting}-seed{seed}",
            "family": family, "group": group, "seed": seed, "variant": variant,
            "basis_key": basis_key(kind, config["rank"], seed),
            "requires": [basis_key(kind, config["rank"], seed)], "config": config,
            "setting": setting, "baseline_group": baseline_group or "legacy:sgd10",
            "partition": "holdout", "budgets": list(budgets), "scenarios": list(scenarios),
            "diagnostics_budget": 20}


def make_plan(root, output, gpus=3, cpus=12):
    tasks = [_basis("legacy", 4, seed) for seed in SEEDS]
    # First yield same-dictionary optimizer comparisons; then jointly learn the
    # new dictionaries. This keeps useful early results independent of later fits.
    for seed in SEEDS:
        tasks.append(_evaluation("legacy", "sgd10", seed, group="legacy:original_sgd10",
                                 config=configuration("sgd10", legacy=True)))
        for variant in ADAPTERS:
            extra = STRESS if variant in ("sgd10", "head_newton", "tangent_profiled", "tangent_information", "information_protected") else ()
            tasks.append(_evaluation("legacy", variant, seed, scenarios=("clean", *extra)))
    # Match training head priors when attributing a benefit to dictionary
    # learning. Legacy dictionaries retain their original .001 head prior.
    tasks += [_basis("matched_sgd", 4, seed) for seed in SEEDS]
    for seed in SEEDS:
        for variant in ("tangent_profiled", "information_protected"):
            tasks.append(_evaluation("matched_sgd", variant, seed,
                                     baseline_group=f"legacy:{variant}"))
    tasks += [_basis(kind, 4, seed) for kind in JOINT_BASES for seed in SEEDS]
    for kind in JOINT_BASES:
        for seed in SEEDS:
            for variant in JOINT_ADAPTERS:
                extra = STRESS if kind == "protected" and variant == "information_protected" else ()
                tasks.append(_evaluation(kind, variant, seed, scenarios=("clean", *extra),
                                         baseline_group=f"legacy:{variant}"))
    tasks += [_basis("legacy", rank, seed) for rank in (2, 8) for seed in SEEDS]
    sensitivity = []
    for rank, radius in ((2, .15), (8, .15), (4, .05), (4, .35)):
        sensitivity.append((f"rank{rank}-radius{radius}", {"rank": rank, "radius": radius}, ("tangent_profiled", "information_protected")))
    for penalty in (.001, .01, 1.):
        sensitivity.append((f"headprior{penalty}", {"head_l2": penalty}, ("tangent_profiled", "information_protected")))
    for steps in (1, 3, 20):
        sensitivity.append((f"iterations{steps}", {"tangent_steps": steps}, ("tangent_profiled", "information_protected")))
    for epsilon in (.002, .05):
        sensitivity.append((f"KL{epsilon}", {"kl_epsilon": epsilon}, ("information_protected",)))
    for scale in (.1, 10.):
        sensitivity.append((f"information{scale}", {"information_scale": scale}, ("information_protected",)))
    for setting, changes, variants in sensitivity:
        for variant in variants:
            for seed in SEEDS:
                tasks.append(_evaluation("legacy", variant, seed, family="sensitivity", setting=setting,
                    group=f"sensitivity:{variant}:{setting}", config=configuration(variant, **changes),
                    baseline_group=f"legacy:{variant}", budgets=(20,)))
    for seed in SEEDS:
        tasks.append({"key": f"reference-supportEA-seed{seed}", "family": "reference", "group": "support_EA",
                      "variant": "support_EA", "seed": seed, "requires": [],
                      "config": configuration("head_newton"), "budgets": list(BUDGETS),
                      "scenarios": ["clean"], "partition": "holdout", "baseline_group": "legacy:head_newton"})
    counts = {family: sum(task["family"] == family for task in tasks) for family in ("basis", "core", "sensitivity", "reference")}
    assert counts == {"basis": 21, "core": 78, "sensitivity": 72, "reference": 3}
    assert len({task['key'] for task in tasks}) == len(tasks)
    specification = {"version": "paper3-round3-profiled-information-v1", "seeds": list(SEEDS),
                     "budgets": list(BUDGETS), "updates": 1000, "episodes_per_update": 4,
                     "validate_every": 50, "checkpoint_every": 10, "outer_lr": .01,
                     "comparisons": [list(pair) for pair in COMPARISONS], "tasks": tasks,
                     "primary_partition": "holdout", "primary_budget": 20, "primary_scenario": "clean",
                     "interpretation": "exploratory internal replication; cohort already used by earlier rounds",
                     "counts": counts, "external_data_required_for_core": False}
    return {**specification, "root": str(Path(root).resolve()), "output": str(Path(output).resolve()),
            "gpus": gpus, "cpus": cpus, "recipe_sha256": _digest({"specification": specification, "code": _code()})}


def make_external_plan(root, output, external_manifest, basis_run, gpus=3, cpus=12):
    """Separate optional published-order transfer; never changes a running core plan."""
    manifest = Path(external_manifest).expanduser().resolve()
    json.loads(manifest.read_text())
    basis_run = Path(basis_run).expanduser().resolve()
    base_plan = json.loads((basis_run / "plan.json").read_text())
    tasks = []
    pairs = [("legacy", "sgd10"), ("legacy", "head_newton"), ("legacy", "tangent_profiled"),
             ("legacy", "tangent_information"), ("legacy", "information_protected"), ("protected", "information_protected")]
    for kind, variant in pairs:
        for seed in SEEDS:
            task = _evaluation(kind, variant, seed, family="transfer", budgets=(20, 4, 8, 12, 16))
            task.update(requires=[], basis_output=str(basis_run), external_manifest=str(manifest), partition="weibo_published_order")
            tasks.append(task)
    spec = {"version": "paper3-round3-weibo-published-order-v1", "seeds": list(SEEDS), "tasks": tasks,
            "budgets": list(BUDGETS), "comparisons": [], "primary_partition": "weibo_published_order",
            "primary_budget": 20, "primary_scenario": "clean", "basis_recipe_sha256": base_plan["recipe_sha256"],
            "external_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "interpretation": "descriptive external published-order LR-subtask robustness; physical acquisition clock unverified"}
    return {**spec, "root": str(Path(root).resolve()), "output": str(Path(output).resolve()), "gpus": gpus, "cpus": cpus,
            "external_manifest": str(manifest), "basis_output": str(basis_run),
            "recipe_sha256": _digest({"specification": spec, "code": _code()})}
