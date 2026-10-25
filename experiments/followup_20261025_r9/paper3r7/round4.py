"""Support cross-fitting, convex probability averaging and robust Lie calibration.

The frozen Round 3 implementation lives in ``core``. Cross-fitting uses only the
acquired support; calibration never accepts future query data or query labels.
Selection is discrete and piecewise differentiated. GCE has a nonconvex fixed
identity-tangent objective and is never described as a globally convex learner.
"""
from dataclasses import dataclass, field, fields, replace
import itertools
import math
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from . import core

# Preserve the previous numerical API for audited controls and their tests.
for _name in dir(core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(core, _name)

NEW_VARIANTS = ("xfit_shrink", "xfit_stack", "xfit_select", "xfit_select_stack",
                "insample_shrink", "insample_stack", "fixed_half", "uniform_stack",
                "gce_profiled")
VARIANTS = core.VARIANTS + NEW_VARIANTS


@dataclass(frozen=True)
class Config(core.Config):
    base_variant: str = "tangent_profiled"
    cv_folds: int = 5
    cv_scheme: str = "blocked"
    cv_rule: str = "one_se"
    stacking_l2: float = .01
    cv_head_grid: tuple = (.01, .1, 1.)
    cv_radius_grid: tuple = (.05, .15)
    gce_q: float = .7

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError("Unknown round-four variant")
        previous = {item.name: getattr(self, item.name) for item in fields(core.Config)}
        previous["variant"] = self.variant if self.variant in core.VARIANTS else "tangent_profiled"
        core.Config(**previous)
        if self.base_variant not in ("tangent_profiled", "information_protected", "gce_profiled"):
            raise ValueError("Unknown cross-fitting base learner")
        if self.cv_folds < 2 or self.cv_scheme not in ("blocked", "stratified", "loo"):
            raise ValueError("Require at least two folds and a supported support-only fold scheme")
        if self.cv_rule not in ("one_se", "minimum"):
            raise ValueError("Unknown fixed CV selection rule")
        if not math.isfinite(self.stacking_l2) or self.stacking_l2 <= 0:
            raise ValueError("Positive finite stacking ridge required")
        if not math.isfinite(self.gce_q) or not 0 <= self.gce_q <= 1:
            raise ValueError("GCE q must lie in [0, 1]")
        for name in ("cv_head_grid", "cv_radius_grid"):
            values = tuple(getattr(self, name))
            if not values or any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("CV prior/radius grid must be nonempty, finite and positive")
            object.__setattr__(self, name, values)


@dataclass
class Fitted(core.Fitted):
    # None denotes the frozen source. Components have no recursive mixtures.
    components: tuple | None = None
    mixture_weights: torch.Tensor | None = None


def support_folds(support, count=5, scheme="blocked"):
    """Ordered held-index sets; blocked bins use *acquired* positions before QC."""
    n = len(support.y)
    if not n:
        return []
    device = support.y.device
    if scheme == "loo":
        return [torch.tensor([i], device=device) for i in range(n)]
    count = min(int(count), n)
    if scheme == "stratified":
        bins = [[] for _ in range(count)]
        for label in (0, 1):
            indices = (support.y == label).nonzero(as_tuple=False).flatten().tolist()
            for number, index in enumerate(indices):
                bins[number % count].append(index)
    elif scheme == "blocked":
        acquired = getattr(support, "acquired_event_uids", ())
        valid = getattr(support, "event_uids", ())
        if acquired and len(valid) == n:
            positions = {uid: index for index, uid in enumerate(acquired)}
            indices = [positions[uid] for uid in valid]
            width = len(acquired)
        else:
            indices, width = list(range(n)), n
        bins = [[] for _ in range(count)]
        for index, position in enumerate(indices):
            bins[min(count - 1, position * count // width)].append(index)
    else:
        raise ValueError("Unknown support fold scheme")
    return [torch.tensor(sorted(indices), device=device) for indices in bins if indices]


def shrink_weight(probability, source_probability, y, ridge):
    """Exact scalar Brier+ridgeto-source minimizer over [0, 1]."""
    a = core._class_weights(y)
    delta = probability - source_probability
    numerator = (a * delta * (y.to(delta) - source_probability)).sum()
    denominator = (a * delta.square()).sum() + ridge
    return (numerator / denominator).clamp(0., 1.)


def simplex_brier_weights(probabilities, y, ridge):
    """Exact tiny simplex QP by all active faces, with source-centered ridge.

    Differentiation follows the selected face. Ties/boundaries are piecewise
    smooth; no higher-order or globally smooth hypergradient claim is made.
    """
    p = probabilities.double()
    if p.ndim != 2 or p.shape[1] not in (2, 3) or p.shape[0] != len(y):
        raise ValueError("Expected N x (2 or 3) component probabilities")
    a = core._class_weights(y)
    k = p.shape[1]
    eye = torch.eye(k, dtype=p.dtype, device=p.device)
    reference = eye[0]
    h = p.T @ (a[:, None] * p) + (ridge / 2) * eye
    rhs = p.T @ (a * y.to(p)) + (ridge / 2) * reference
    best, best_value = None, math.inf
    for size in range(1, k + 1):
        for face in itertools.combinations(range(k), size):
            indices = torch.tensor(face, device=p.device)
            sub = h[indices][:, indices]
            one = p.new_ones(size)
            inverse_rhs = torch.linalg.solve(sub, rhs[indices])
            inverse_one = torch.linalg.solve(sub, one)
            multiplier = (inverse_rhs.sum() - 1) / inverse_one.sum()
            active = inverse_rhs - multiplier * inverse_one
            if bool((active.detach() < -1e-10).any()):
                continue
            active = active.clamp_min(0.)
            active = active / active.sum()
            value = p.new_zeros(k).scatter(0, indices, active)
            objective = (a * (p @ value - y.to(p)).square()).sum()
            objective = objective + (ridge / 2) * (value - reference).square().sum()
            score = float(objective.detach())
            if score < best_value:
                best, best_value = value, score
    if best is None:
        raise FloatingPointError("No finite simplex face")
    return best


def _gce_objective(margin, y, theta, head, config):
    signed = (2 * y.to(margin) - 1) * margin
    log_probability = F.logsigmoid(signed)
    losses = -log_probability if config.gce_q == 0 else -torch.expm1(config.gce_q * log_probability) / config.gce_q
    return ((core._class_weights(y) * losses).sum() + config.theta_l2 * theta.square().sum()
            + config.head_l2 * head.square().sum() / len(head))


def _gce_fit(m0, coordinates, features, b, y, config):
    if config.gce_q == 0:
        weights, diagnostics = core._surrogate_fit(m0, coordinates, features, b, y,
                                                   replace(config, variant="tangent_profiled"))
        diagnostics.update(gce_q=0., robust_objective="CE_limit_exact")
        return weights, diagnostics
    j, z = coordinates @ b, core._augmented(features)
    theta, head = b.new_zeros(b.shape[1]), b.new_zeros(z.shape[1])
    values, fractions = [], []
    for _ in range(config.tangent_steps):
        margin = m0 + j @ theta + z @ head
        probability = margin.sigmoid()
        robust = F.logsigmoid((2 * y.to(margin) - 1) * margin).mul(config.gce_q).exp()
        a = core._class_weights(y)
        residual = a * robust * (probability - y.to(margin))
        curvature = a * robust * probability * (1 - probability)
        lt, lh = 2 * config.theta_l2, 2 * config.head_l2 / len(head)
        gt, gh = j.T @ residual + lt * theta, z.T @ residual + lh * head
        htt = j.T @ (curvature[:, None] * j) + (lt + config.damping) * torch.eye(len(theta), device=b.device, dtype=b.dtype)
        hth = j.T @ (curvature[:, None] * z)
        hhh = z.T @ (curvature[:, None] * z) + (lh + config.damping) * torch.eye(len(head), device=b.device, dtype=b.dtype)
        schur = htt - hth @ torch.linalg.solve(hhh, hth.T)
        schur = (schur + schur.T) / 2
        gradient = gt - hth @ torch.linalg.solve(hhh, gh)
        proposed_theta = core.ball_quadratic(schur, schur @ theta - gradient, config.radius)
        proposed_head = head - torch.linalg.solve(hhh, gh + hth.T @ (proposed_theta - theta))
        before = _gce_objective(margin, y, theta, head, config)
        if not values:
            values.append(float(before.detach()))
        accepted = 0.
        for power in range(config.line_search_steps):
            fraction = 2. ** -power
            tt, hh = theta + fraction * (proposed_theta - theta), head + fraction * (proposed_head - head)
            after = _gce_objective(m0 + j @ tt + z @ hh, y, tt, hh, config)
            if bool(after.detach() <= before.detach() + 1e-12):
                theta, head, accepted = tt, hh, fraction
                break
        values.append(float(_gce_objective(m0 + j @ theta + z @ head, y, theta, head, config).detach()))
        fractions.append(accepted)
    margin = m0 + j @ theta + z @ head
    robust = F.logsigmoid((2 * y.to(margin) - 1) * margin).mul(config.gce_q).exp()
    return core.FastWeights(theta, head[:-1], head[-1]), {
        "surrogate": "fixed_identity_tangent_nonconvex_GCE", "gce_q": config.gce_q,
        "gce_objective_history": values, "surrogate_line_search_fractions": fractions,
        "robust_target_probability_weight_mean": float(robust.detach().mean()),
        "robust_target_probability_weight_class0": float(robust[y == 0].detach().mean()),
        "robust_target_probability_weight_class1": float(robust[y == 1].detach().mean()),
        "optimizer_updates": config.tangent_steps, "theta_norm": float(theta.detach().norm())}


def _cached_fit(source, geometry, b, support, cached, config, *, variant=None, protected_baseline=None):
    variant = variant or config.base_variant
    if not all(bool((support.y == label).any()) for label in (0, 1)):
        return core.Fitted(core.zero_fast_weights(source, b.shape[1], device=b.device),
                           {"optimizer_updates": 0}, "insufficient_class_support")
    m0, coordinates, features = cached
    if variant == "head_newton":
        weights, diagnostics = core._head_newton(m0, features, b, support.y, config)
    elif variant == "gce_profiled":
        weights, diagnostics = _gce_fit(m0, coordinates, features, b, support.y, config)
    else:
        weights, diagnostics = core._surrogate_fit(m0, coordinates, features, b, support.y,
                                                  replace(config, variant=variant))
    fraction = 1.
    if variant == "information_protected":
        weights, fraction, kl = core._protect(source, geometry, b, weights, support, config, protected_baseline)
        diagnostics.update(protection_fraction=fraction, actual_source_support_kl=kl)
    return core.Fitted(weights, diagnostics, protection_fraction=fraction)


def _probability(source, geometry, b, fitted, x):
    logits = source.source_logits(x) if fitted is None else core.predict(source, geometry, b, fitted, x)
    return logits.double().softmax(1)[:, 1]


def _subsupport(support, indices):
    # Internal training-fold view: no query tensors, event replacement or labels.
    return SimpleNamespace(x=support.x[indices], y=support.y[indices])


def _oof(source, geometry, b, support, cached, config, folds, *, include_head=False):
    n = len(support.y)
    all_indices = torch.arange(n, device=b.device)
    source_probability = cached[0].sigmoid()
    spatial = source_probability.clone()
    head = source_probability.clone()
    fallback_count, fit_count = 0, 0
    for held in folds:
        mask = torch.ones(n, device=b.device, dtype=torch.bool)
        mask[held] = False
        train = all_indices[mask]
        training = _subsupport(support, train)
        training_cache = tuple(value[train] for value in cached)
        state = _cached_fit(source, geometry, b, training, training_cache, config)
        fit_count += 1
        fallback_count += int(state.fallback_reason is not None)
        spatial = spatial.index_copy(0, held, _probability(source, geometry, b, state, support.x[held]))
        if include_head:
            state_head = _cached_fit(source, geometry, b, training, training_cache, config, variant="head_newton")
            head = head.index_copy(0, held, _probability(source, geometry, b, state_head, support.x[held]))
            fit_count += 1
    return source_probability, head, spatial, fallback_count, fit_count


def _candidate_score(probability, y, folds):
    a = core._class_weights(y)
    losses = (probability - y.to(probability)).square()
    risk = (a * losses).sum()
    group_mass = torch.stack([a[index].sum() for index in folds])
    group_risk = torch.stack([(a[index] * losses[index]).sum() / a[index].sum() for index in folds])
    mass_square = group_mass.square().sum()
    if len(folds) > 1 and bool(mass_square < 1):
        variance = (group_mass * (group_risk - risk).square()).sum() / (1 - mass_square)
        se = (variance * mass_square).clamp_min(0).sqrt()
    else:
        se = risk * 0
    return risk, se


def _wrap(state, *, components=None, mixture_weights=None, cv=None, create_graph=False):
    weights = state.weights
    if not all(bool(torch.isfinite(value).all()) for value in (weights.theta, weights.delta_w, weights.delta_b)):
        raise FloatingPointError("Nonfinite cross-fitting component state")
    if mixture_weights is not None and not bool(torch.isfinite(mixture_weights).all()):
        raise FloatingPointError("Nonfinite probability mixture weights")
    if not create_graph:
        weights = core.FastWeights(*(value.detach() for value in (weights.theta, weights.delta_w, weights.delta_b)))
        if components is not None:
            components = tuple(None if item is None else _wrap(item) for item in components)
        if mixture_weights is not None:
            mixture_weights = mixture_weights.detach()
    diagnostics = dict(state.diagnostics)
    if cv is not None:
        diagnostics["cv"] = cv
    return Fitted(weights, diagnostics, state.fallback_reason, state.protection_fraction,
                  components, mixture_weights)


def fit(source, geometry, b, support, config, *, create_graph=False, protected_baseline=None):
    """Fit acquired support only; query inputs/labels are not accepted."""
    if config.variant in core.VARIANTS:
        return core.fit(source, geometry, b, support, config, create_graph=create_graph,
                        protected_baseline=protected_baseline)
    core._assert_frozen(source)
    if b.shape != (99, config.rank):
        raise ValueError("B must match the configured full Lie basis")
    if not all(bool((support.y == label).any()) for label in (0, 1)):
        return Fitted(core.zero_fast_weights(source, b.shape[1], device=b.device),
                      {"cv": {"status": "FALLBACK", "fallback_reason": "insufficient_class_support"}},
                      "insufficient_class_support")
    work_b = b if create_graph else b.detach()
    with torch.enable_grad():
        cached = core.support_tangent(source, geometry, support)
        if config.variant == "gce_profiled":
            state = _cached_fit(source, geometry, work_b, support, cached, config, variant="gce_profiled")
            return _wrap(state, create_graph=create_graph)
        needs_head = config.variant in ("xfit_stack", "xfit_select_stack", "insample_stack", "uniform_stack")
        cv = {"cv_scheme": config.cv_scheme, "status": "COMPLETE", "selection_rule": config.cv_rule,
              "valid_folds": 0, "n_candidate_fits": 0, "fold_fallbacks": 0,
              "selection_labels": "acquired_support_only", "criterion": "class_balanced_Brier"}
        selected_config = config
        selected_oof = None
        folds = support_folds(support, config.cv_folds, config.cv_scheme)
        if config.variant.startswith("xfit"):
            if "select" in config.variant:
                candidates = []
                for penalty, radius in itertools.product(config.cv_head_grid, config.cv_radius_grid):
                    candidate = replace(config, head_l2=penalty, radius=radius)
                    p0, ph, ps, failures, calls = _oof(source, geometry, work_b, support, cached, candidate,
                                                     folds, include_head=needs_head)
                    risk, se = _candidate_score(ps, support.y, folds)
                    candidates.append((candidate, (p0, ph, ps), failures, risk, se))
                    cv["n_candidate_fits"] += calls
                best = min(range(len(candidates)), key=lambda index: (float(candidates[index][3].detach()), index))
                minimum, error = candidates[best][3:5]
                threshold = minimum + error if config.cv_rule == "one_se" else minimum
                eligible = [item for item in candidates if float(item[3].detach()) <= float(threshold.detach()) + 1e-14]
                chosen = min(eligible, key=lambda item: (item[0].radius, -item[0].head_l2)) if config.cv_rule == "one_se" else candidates[best]
                selected_config, selected_oof, failures, _, _ = chosen
                cv.update(minimum_cv_brier=float(minimum.detach()), one_se=float(error.detach()),
                          selection_threshold=float(threshold.detach()), candidate_count=len(candidates),
                          candidate_scores=[{"head_l2": item[0].head_l2, "radius": item[0].radius,
                                             "brier": float(item[3].detach()), "se": float(item[4].detach())}
                                            for item in candidates])
            else:
                p0, ph, ps, failures, calls = _oof(source, geometry, work_b, support, cached, config,
                                                  folds, include_head=needs_head)
                selected_oof = p0, ph, ps
                cv.update(n_candidate_fits=calls, candidate_count=1)
            cv.update(n_folds=len(folds), valid_folds=len(folds) - failures, fold_fallbacks=failures,
                      standard_error_scope="selection heuristic; temporally dependent folds are not independent inference units")
        else:
            cv.update(cv_scheme="none", selection_rule="fixed" if config.variant in ("fixed_half", "uniform_stack") else "insample_control")
        if config.variant.startswith("xfit") and cv["valid_folds"] == 0:
            cv.update(status="SOURCE_ONLY_NO_VALID_TRAINING_FOLD", fallback_reason="cv_no_valid_training_fold",
                      selected_candidate="source", selected_config={}, mix_source=1., mix_head=0., mix_spatial=0.)
            state = core.Fitted(core.zero_fast_weights(source, work_b.shape[1], device=work_b.device),
                                {}, "cv_no_valid_training_fold")
            return _wrap(state, cv=cv, create_graph=create_graph)
        state = _cached_fit(source, geometry, work_b, support, cached, selected_config,
                            protected_baseline=protected_baseline)
        cv["n_candidate_fits"] += 1
        head_state = _cached_fit(source, geometry, work_b, support, cached, selected_config, variant="head_newton") if needs_head else None
        cv["n_candidate_fits"] += int(needs_head)
        cv.update(selected_candidate=f"head{selected_config.head_l2}-radius{selected_config.radius}",
                  selected_config={"head_l2": selected_config.head_l2, "radius": selected_config.radius,
                                   "base_variant": selected_config.base_variant})
        if config.variant == "xfit_select":
            return _wrap(state, cv=cv, create_graph=create_graph)
        if selected_oof is not None:
            p0, ph, ps = selected_oof
        else:
            p0 = cached[0].sigmoid()
            ps = _probability(source, geometry, work_b, state, support.x)
            ph = _probability(source, geometry, work_b, head_state, support.x) if needs_head else p0
        if needs_head:
            if config.variant == "uniform_stack":
                mix = work_b.new_full((3,), 1 / 3)
            else:
                mix = simplex_brier_weights(torch.stack((p0, ph, ps), 1), support.y, config.stacking_l2)
            components = None, head_state, state
        else:
            alpha = work_b.new_tensor(.5) if config.variant == "fixed_half" else shrink_weight(ps, p0, support.y, config.stacking_l2)
            mix = torch.stack((1 - alpha, alpha))
            components = None, state
        cv.update(mix_source=float(mix[0].detach()), mix_head=float(mix[1].detach()) if needs_head else 0.,
                  mix_spatial=float(mix[-1].detach()), stacking_l2=config.stacking_l2,
                  mixture_constraint="simplex_probability_averaging", insample_selection=config.variant.startswith("insample"))
        answer = _wrap(state, components=components, mixture_weights=mix, cv=cv, create_graph=create_graph)
    core._assert_frozen(source)
    return answer


def predict(source, geometry, b, fitted, query_x):
    """Actual component predictions, convexly averaged in probability space."""
    components = getattr(fitted, "components", None)
    if components is None:
        return core.predict(source, geometry, b, fitted, query_x)
    probabilities = torch.stack([_probability(source, geometry, b, item, query_x)
                                 for item in components], 1)
    probability = (probabilities @ fitted.mixture_weights).clamp(1e-12, 1 - 1e-12)
    return torch.stack((torch.log1p(-probability), probability.log()), 1)
