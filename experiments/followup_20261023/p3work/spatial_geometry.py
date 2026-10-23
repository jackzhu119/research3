"""Paper3 v2 geometry, independent of legacy_baseline_v1.

Inputs are finite C x T trials after upstream CAR/temporal demeaning and
before joint RMS. All geometric matrices and the RMS reduction use float64.
No dataset download or training is performed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Literal

import torch
from torch import Tensor


def tensor_sha256(value: Tensor) -> str:
    """Hash actual canonical float64 array bytes, including shape."""
    value = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    digest = hashlib.sha256(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def helmert_rows(channels: int = 11, *, device=None) -> Tensor:
    """Deterministic positive-leading Helmert contrasts, (C-1) x C."""
    if channels < 2:
        raise ValueError("at least two channels required")
    q = torch.zeros(channels - 1, channels, dtype=torch.float64, device=device)
    for k in range(1, channels):
        scale = (k * (k + 1)) ** -0.5
        q[k - 1, :k] = scale
        q[k - 1, k] = -k * scale
    return q


def traceless_basis(d: int = 10, *, kind: Literal["full", "symmetric", "skew"] = "full", device=None) -> Tensor:
    """E in R^(d*d) x n with row-major vectorization and fixed ordering.

    Full: off-diagonal standard matrices, then Helmert diagonal contrasts.
    Symmetric/skew are registered geometry ablations, not the main method.
    """
    if d < 2 or kind not in {"full", "symmetric", "skew"}:
        raise ValueError("invalid dimension or basis kind")
    matrices = []
    if kind == "full":
        for i in range(d):
            for j in range(d):
                if i != j:
                    e = torch.zeros(d, d, dtype=torch.float64, device=device)
                    e[i, j] = 1
                    matrices.append(e)
    else:
        for i in range(d):
            for j in range(i + 1, d):
                e = torch.zeros(d, d, dtype=torch.float64, device=device)
                e[i, j] = 2 ** -0.5
                e[j, i] = (1 if kind == "symmetric" else -1) * 2 ** -0.5
                matrices.append(e)
    if kind != "skew":
        matrices.extend(torch.diag(row) for row in helmert_rows(d, device=device))
    return torch.stack(matrices, dim=0).reshape(-1, d * d).T.contiguous()


def positive_qr(value: Tensor) -> Tensor:
    """Thin QR with nonnegative R diagonal, including a deterministic zero sign."""
    q, r = torch.linalg.qr(value, mode="reduced")
    signs = torch.where(torch.diagonal(r) < 0, -1.0, 1.0)
    return q * signs.unsqueeze(0)


def initialize_dictionary(rank: int, seed: int, *, d: int = 10, kind="full", device=None) -> Tensor:
    n = traceless_basis(d, kind=kind).shape[1]
    if rank < 1 or rank > n:
        raise ValueError("rank exceeds the generator coordinate dimension")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    b = positive_qr(torch.randn(n, rank, generator=generator, dtype=torch.float64))
    return b.to(device=device)


def generators(e: Tensor, b: Tensor) -> Tensor:
    if e.dtype != torch.float64 or b.dtype != torch.float64:
        raise ValueError("E and B must use float64")
    d = int(e.shape[0] ** 0.5)
    if e.shape[0] != d * d or b.ndim != 2 or b.shape[0] != e.shape[1]:
        raise ValueError("incompatible E/B dimensions")
    return (e @ b).T.reshape(b.shape[1], d, d)


def project_ball(theta: Tensor, radius: float) -> Tensor:
    if radius <= 0:
        raise ValueError("radius must be positive")
    # clamp_min(1), rather than an epsilon norm, is also differentiable at zero.
    return theta / (torch.linalg.vector_norm(theta) / radius).clamp_min(1.0)


def joint_trial_rms(value: Tensor) -> Tensor:
    """N(Y) with an exact float64 RMS reduction; no amplitude floor."""
    if value.ndim < 2 or not torch.isfinite(value).all():
        raise ValueError("nonfinite or malformed trial; upstream QC required")
    y = value.to(torch.float64)
    rms = y.square().mean(dim=(-2, -1), keepdim=True).sqrt()
    if torch.any(rms == 0):
        raise ValueError("zero-energy trial; upstream QC required")
    return y / rms


@dataclass(frozen=True)
class SourceCovarianceTrial:
    dataset: str
    subject: str
    event_uid: str
    x: Tensor
    role: str = "source_pretrain_train"


@dataclass(frozen=True)
class ReferenceCovariance:
    matrix: Tensor
    sqrt: Tensor
    invsqrt: Tensor
    event_uids: tuple[str, ...]
    events_sha256: str
    weighting: str = "dataset_equal/subject_equal/trial_equal"
    shrinkage: float = 0.05


def spd_roots(matrix: Tensor) -> tuple[Tensor, Tensor]:
    value = matrix.to(torch.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not torch.isfinite(value).all():
        raise ValueError("finite square SPD matrix required")
    if not torch.allclose(value, value.T, atol=1e-12, rtol=1e-12):
        raise ValueError("reference covariance must be symmetric")
    eigenvalues, eigenvectors = torch.linalg.eigh(value)
    if torch.any(eigenvalues <= 0):
        raise ValueError("reference covariance must be strictly positive definite")
    return ((eigenvectors * eigenvalues.sqrt()) @ eigenvectors.T,
            (eigenvectors * eigenvalues.rsqrt()) @ eigenvectors.T)


def source_reference(trials: Iterable[SourceCovarianceTrial], q: Tensor | None = None) -> ReferenceCovariance:
    """Fixed 0.05 shrinkage, equally weighted dataset/subject/trial hierarchy.

    Only an explicitly authorized source pretraining training split is accepted.
    Dataset/subject provenance is mandatory and duplicate event UIDs fail closed.
    A caller with raw data must first apply frozen upstream QC; invalid trials
    fail here rather than silently altering the source training manifest.
    """
    q = helmert_rows() if q is None else q.to(torch.float64)
    groups: dict[str, dict[str, list[Tensor]]] = {}
    # Exact registry IDs; aliases would risk counting one source twice.
    source_datasets = {"BNCI2014_001", "Cho2017", "Lee2019_MI", "PhysionetMI"}
    seen = set()
    for trial in trials:
        if trial.role != "source_pretrain_train":
            raise PermissionError("Rref accepts source_pretrain_train only")
        if trial.dataset not in source_datasets:
            raise PermissionError("Rref is restricted to the four registered source datasets")
        if not all((trial.dataset, trial.subject, trial.event_uid)) or trial.event_uid in seen:
            raise ValueError("missing provenance or duplicated source event")
        x = trial.x.to(device=q.device, dtype=torch.float64)
        if x.ndim != 2 or x.shape[0] != q.shape[1] or not torch.isfinite(x).all():
            raise ValueError("invalid source trial")
        z = q @ x
        c = z @ z.T
        trace = torch.trace(c)
        if trace <= 0:
            raise ValueError("zero-energy source trial")
        groups.setdefault(trial.dataset, {}).setdefault(trial.subject, []).append(c / trace)
        seen.add(trial.event_uid)
    if set(groups) != source_datasets:
        raise ValueError("Rref requires training events from all four registered sources")
    dataset_means = []
    for dataset in sorted(groups):
        subjects = groups[dataset]
        dataset_means.append(torch.stack([
            torch.stack(subjects[subject]).mean(dim=0) for subject in sorted(subjects)
        ]).mean(dim=0))
    mean = torch.stack(dataset_means).mean(dim=0)
    d = q.shape[0]
    matrix = 0.95 * mean + 0.05 * torch.eye(d, dtype=torch.float64, device=q.device) / d
    sqrt, invsqrt = spd_roots(matrix)
    uids = tuple(sorted(seen))
    events_sha = hashlib.sha256("\n".join(uids).encode()).hexdigest()
    return ReferenceCovariance(matrix, sqrt, invsqrt, uids, events_sha)


@dataclass(frozen=True)
class SpatialGeometry:
    q: Tensor
    rref_sqrt: Tensor
    rref_invsqrt: Tensor
    e: Tensor

    def validate(self) -> None:
        d, channels = self.q.shape
        if channels != d + 1 or self.e.shape[0] != d * d:
            raise ValueError("invalid CAR/generator dimensions")
        if any(v.dtype != torch.float64 or not torch.isfinite(v).all()
               for v in (self.q, self.rref_sqrt, self.rref_invsqrt, self.e)):
            raise ValueError("geometry must contain finite float64 arrays")
        eye = torch.eye(d, dtype=torch.float64, device=self.q.device)
        h = torch.eye(channels, dtype=torch.float64, device=self.q.device) - torch.ones(channels, channels, dtype=torch.float64, device=self.q.device) / channels
        if not torch.allclose(self.q @ self.q.T, eye, atol=1e-10, rtol=1e-10) or not torch.allclose(self.q.T @ self.q, h, atol=1e-10, rtol=1e-10):
            raise ValueError("Q must be the row-orthogonal CAR coordinate map")
        if not torch.allclose(self.rref_sqrt @ self.rref_invsqrt, eye, atol=1e-10, rtol=1e-10):
            raise ValueError("reference square roots disagree")
        if not torch.allclose(self.e.T @ self.e, torch.eye(self.e.shape[1], dtype=torch.float64, device=self.e.device), atol=1e-10, rtol=1e-10):
            raise ValueError("E must be Frobenius-orthogonal")
        traces = self.e.T.reshape(-1, d, d).diagonal(dim1=1, dim2=2).sum(1)
        if not torch.allclose(traces, torch.zeros_like(traces), atol=1e-10, rtol=0):
            raise ValueError("E must be traceless")

    def sha256(self) -> str:
        self.validate()
        return hashlib.sha256("/".join(tensor_sha256(v) for v in
            (self.q, self.rref_sqrt, self.rref_invsqrt, self.e)).encode()).hexdigest()

    @classmethod
    def from_reference(cls, reference: Tensor, *, channels=11, kind="full"):
        root, inverse = spd_roots(reference)
        if reference.shape != (channels - 1, channels - 1):
            raise ValueError("reference and channel count disagree")
        return cls(helmert_rows(channels, device=root.device), root, inverse,
                   traceless_basis(channels - 1, kind=kind, device=root.device))

    def validate_dictionary(self, b: Tensor) -> None:
        self.validate()
        if b.ndim != 2 or b.shape[0] != self.e.shape[1] or b.dtype != torch.float64 or not torch.isfinite(b).all():
            raise ValueError("finite float64 B must match E's coordinates")
        if not torch.allclose(b.T @ b, torch.eye(b.shape[1], dtype=b.dtype, device=b.device), atol=1e-10, rtol=1e-10):
            raise ValueError("dictionary B must have orthonormal columns")

    def matrix(self, theta: Tensor, b: Tensor, *, family="exp") -> tuple[Tensor, Tensor, Tensor]:
        if theta.dtype != torch.float64:
            raise ValueError("theta must use float64")
        g = generators(self.e, b)
        if theta.shape != (g.shape[0],):
            raise ValueError("theta and dictionary rank disagree")
        m = torch.einsum("r,rij->ij", theta, g)
        if family == "exp":
            t = torch.matrix_exp(m)
        elif family == "additive":
            t = torch.eye(m.shape[0], dtype=m.dtype, device=m.device) + m
        else:
            raise ValueError("unknown transformation family")
        a = self.q.T @ self.rref_sqrt @ t @ self.rref_invsqrt @ self.q
        return a, t, m

    def transform(self, x: Tensor, theta: Tensor, b: Tensor, *, family="exp") -> Tensor:
        a, _, _ = self.matrix(theta, b, family=family)
        if x.ndim != 3 or x.shape[1] != self.q.shape[1]:
            raise ValueError("expected N x C x T input")
        return joint_trial_rms(a @ x.to(device=a.device, dtype=torch.float64))


def stiefel_step(b: Tensor, gradient: Tensor, *, learning_rate=0.01, max_norm=1.0) -> Tensor:
    """Tangent-project, clip its Frobenius norm globally, then QR retract."""
    if b.dtype != torch.float64 or gradient.dtype != torch.float64:
        raise ValueError("Stiefel coordinates/gradient must use float64")
    if not torch.isfinite(gradient).all() or learning_rate <= 0 or max_norm <= 0:
        raise ValueError("invalid meta-gradient or step configuration")
    btg = b.T @ gradient
    tangent = gradient - b @ (0.5 * (btg + btg.T))
    tangent = tangent / (torch.linalg.vector_norm(tangent) / max_norm).clamp_min(1.0)
    return positive_qr(b - learning_rate * tangent)
