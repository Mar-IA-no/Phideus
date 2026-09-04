"""Neural primitives for the proportional graph factorial smoke.

The learned block edits observed edge relations and predicts edge reliability.
It never receives private causal masks or latent node potentials as inputs, and
the graph solve remains an external operation shared by every arm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .proportional_graph_contract import PublicGraphObservation, incidence_matrix

EVIDENCE_LEVELS = frozenset({"raw", "closure"})
MIXER_TYPES = frozenset({"generic", "typed"})


@dataclass(frozen=True)
class NeuralGraphOutput:
    corrected_log_ratio: torch.Tensor
    reliability: torch.Tensor
    correction: torch.Tensor
    path_attention: torch.Tensor


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def observation_tensors(
    observation: PublicGraphObservation,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Convert one public observation without admitting private authority."""
    return {
        "edge_index": torch.as_tensor(
            observation.edge_index, dtype=torch.long, device=device
        ),
        "observed_log_ratio": torch.as_tensor(
            observation.observed_log_ratio, dtype=dtype, device=device
        ),
        "edge_valid": torch.as_tensor(
            observation.edge_valid, dtype=torch.bool, device=device
        ),
        "path_index": torch.as_tensor(
            observation.path_index, dtype=torch.long, device=device
        ),
        "path_sign": torch.as_tensor(observation.path_sign, dtype=dtype, device=device),
        "path_valid": torch.as_tensor(
            observation.path_valid, dtype=torch.bool, device=device
        ),
        "edge_variance": torch.as_tensor(
            observation.edge_variance, dtype=dtype, device=device
        ),
        "n_nodes": torch.as_tensor(
            observation.n_nodes, dtype=torch.long, device=device
        ),
    }


def shuffled_path_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Balance-preserving path-incidence shuffle used as a causal control.

    Operand pairs and their orientation signs are reassigned jointly. Target
    edges, path counts, operand marginals and sign marginals stay fixed. Every
    valid path must change and no reassigned operand may equal its target edge;
    otherwise the control is rejected instead of silently retaining identities.
    """
    path_index = tensors["path_index"]
    valid_members = (
        torch.nonzero(tensors["path_valid"], as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .numpy()
    )
    if len(valid_members) < 2:
        return {
            **tensors,
            "path_shuffle_eligible": torch.tensor(False),
        }
    path_sign = tensors["path_sign"]
    shuffled = path_index.clone()
    shuffled_sign = path_sign.clone()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    members = torch.as_tensor(valid_members, dtype=torch.long, device=path_index.device)
    local = path_index[members].detach().cpu().numpy()
    targets = local[:, 0]
    operand_pairs = local[:, 1:]
    allowed = np.ones((len(members), len(members)), dtype=bool)
    allowed &= np.arange(len(members))[:, None] != np.arange(len(members))[None, :]
    allowed &= targets[:, None] != operand_pairs[None, :, 0]
    allowed &= targets[:, None] != operand_pairs[None, :, 1]
    jitter = torch.rand((len(members), len(members)), generator=generator).numpy()
    rows, cols = linear_sum_assignment(np.where(allowed, jitter, 1e6))
    if len(rows) != len(members) or not np.all(allowed[rows, cols]):
        return {
            **tensors,
            "path_shuffle_eligible": torch.tensor(False),
        }
    order = torch.as_tensor(cols, dtype=torch.long, device=path_index.device)
    shuffled[members, 1:] = path_index[members[order], 1:]
    shuffled_sign[members, 1:] = path_sign[members[order], 1:]
    if torch.any(torch.all(shuffled[members] == path_index[members], dim=1)):
        raise AssertionError("path shuffle retained an identity assignment")
    return {
        **tensors,
        "path_index": shuffled,
        "path_sign": shuffled_sign,
        "path_shuffle_eligible": torch.tensor(True),
    }


def materialize_closure_evidence(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Materialize observed two-hop closure outside the learned module."""
    index = tensors["path_index"]
    values = tensors["observed_log_ratio"][index]
    signed = values * tensors["path_sign"]
    closure = signed[:, 0] - signed[:, 1] - signed[:, 2]
    return {**tensors, "path_closure": closure}


class ProportionalPathMixer(nn.Module):
    """Capacity-matched generic or composition-typed local path mixer."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        evidence: str = "raw",
        mixer: str = "generic",
        weight_floor: float = 1e-3,
        input_scale: float = 1.0,
        mix_paths: bool = True,
    ) -> None:
        super().__init__()
        if evidence not in EVIDENCE_LEVELS:
            raise ValueError(f"unknown evidence level: {evidence}")
        if mixer not in MIXER_TYPES:
            raise ValueError(f"unknown mixer type: {mixer}")
        if hidden_dim < 4 or not 0.0 < weight_floor < 1.0 or input_scale <= 0:
            raise ValueError("invalid model hyperparameters")
        self.hidden_dim = int(hidden_dim)
        self.evidence = evidence
        self.mixer = mixer
        self.weight_floor = float(weight_floor)
        self.mix_paths = bool(mix_paths)
        self.register_buffer("input_scale", torch.tensor(float(input_scale)))

        self.edge_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.path_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.path_score = nn.Linear(hidden_dim, 1)
        self.swap_logit = nn.Parameter(torch.zeros(()))
        self.update_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid()
        )
        self.update_value = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def _encode(self, values: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        scale = self.input_scale.to(dtype=values.dtype, device=values.device)
        features = torch.stack(
            (values / scale, torch.sqrt(variance.clamp_min(1e-12)) / scale), dim=-1
        )
        return self.edge_encoder(features)

    @staticmethod
    def _aggregate(
        messages: torch.Tensor,
        scores: torch.Tensor,
        targets: torch.Tensor,
        n_edges: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aggregate = messages.new_zeros((n_edges, messages.shape[-1]))
        attention = scores.new_zeros(scores.shape)
        for target in torch.unique(targets, sorted=True):
            mask = targets == target
            local = torch.softmax(scores[mask], dim=0)
            attention[mask] = local
            aggregate[target] = (local[:, None] * messages[mask]).sum(dim=0)
        return aggregate, attention

    def _core(
        self,
        values: torch.Tensor,
        variance: torch.Tensor,
        path_index: torch.Tensor,
        path_sign: torch.Tensor,
        path_valid: torch.Tensor,
        path_closure: torch.Tensor | None,
        *,
        typed: bool,
        mix_paths: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_state = self._encode(values, variance)
        valid = path_valid
        valid &= torch.all(path_index >= 0, dim=1)
        if not torch.any(valid):
            aggregate = edge_state.new_zeros(edge_state.shape)
            attention = values.new_zeros((len(path_index),))
        else:
            index = path_index[valid]
            signs = path_sign[valid]
            oriented = values[index] * signs
            encoded = self._encode(oriented, variance[index])
            if not mix_paths:
                encoded = encoded[:, :1].expand(-1, 3, -1)
            role_signs = torch.ones_like(signs)
            closure_feature = (
                path_closure[valid, None] / self.input_scale.to(values)
                if self.evidence == "closure"
                else values.new_zeros((len(index), 1))
            )

            def path_features(states: torch.Tensor) -> torch.Tensor:
                return torch.cat(
                    (
                        states[:, 0],
                        states[:, 1],
                        states[:, 2],
                        role_signs,
                        closure_feature,
                    ),
                    dim=-1,
                )

            forward_messages = self.path_mlp(path_features(encoded))
            swapped = encoded[:, [0, 2, 1]]
            swapped_messages = self.path_mlp(path_features(swapped))
            symmetric = 0.5 * (forward_messages + swapped_messages)
            if typed:
                symmetric_scale = 0.5 + torch.sigmoid(self.swap_logit)
                messages = symmetric_scale * symmetric
            else:
                # Both statistics are invariant to exchanging the two operand
                # roles. GENERIC remains flexible without acquiring a hidden
                # dependence on the arbitrary edge order induced by relabeling.
                mix = torch.sigmoid(self.swap_logit)
                messages = mix * symmetric + (1.0 - mix) * torch.abs(
                    forward_messages - swapped_messages
                )
            local_scores = self.path_score(messages).squeeze(-1)
            if not mix_paths:
                messages = messages * torch.sigmoid(local_scores)[:, None]
            aggregate, local_attention = self._aggregate(
                messages, local_scores, index[:, 0], len(values)
            )
            attention = values.new_zeros((len(path_index),))
            attention[valid] = local_attention
        joined = torch.cat((edge_state, aggregate), dim=-1)
        hidden = self.norm(
            edge_state + self.update_gate(joined) * self.update_value(joined)
        )
        correction = self.correction_head(hidden).squeeze(-1) * self.input_scale.to(
            values
        )
        reliability_logit = self.reliability_head(hidden).squeeze(-1)
        return correction, reliability_logit, attention

    def forward(self, tensors: dict[str, torch.Tensor]) -> NeuralGraphOutput:
        values = tensors["observed_log_ratio"]
        variance = tensors["edge_variance"]
        path_index = tensors["path_index"]
        path_sign = tensors["path_sign"]
        path_valid = tensors["path_valid"].clone()
        path_valid &= torch.all(tensors["edge_valid"][path_index], dim=1)
        if self.evidence == "closure" and "path_closure" not in tensors:
            raise ValueError("closure evidence must be materialized outside the model")
        supplied_closure = tensors.get("path_closure")

        edge_index = tensors["edge_index"]
        canonical_sign = torch.where(
            edge_index[:, 0] < edge_index[:, 1],
            torch.ones_like(values),
            -torch.ones_like(values),
        )
        canonical_values = values * canonical_sign
        effective_sign = path_sign * canonical_sign[path_index]
        plus = self._core(
            canonical_values,
            variance,
            path_index,
            effective_sign,
            path_valid,
            supplied_closure,
            typed=self.mixer == "typed",
            mix_paths=self.mix_paths,
        )
        minus = self._core(
            -canonical_values,
            variance,
            path_index,
            effective_sign,
            path_valid,
            None if supplied_closure is None else -supplied_closure,
            typed=self.mixer == "typed",
            mix_paths=self.mix_paths,
        )
        correction_canonical = 0.5 * (plus[0] - minus[0])
        reliability_logit = 0.5 * (plus[1] + minus[1])
        attention = 0.5 * (plus[2] + minus[2])
        correction = correction_canonical * canonical_sign
        corrected = values + correction

        reliability = self.weight_floor + (1.0 - self.weight_floor) * torch.sigmoid(
            reliability_logit
        )
        reliability = torch.where(
            tensors["edge_valid"], reliability, torch.zeros_like(reliability)
        )
        return NeuralGraphOutput(
            corrected_log_ratio=corrected,
            reliability=reliability,
            correction=correction,
            path_attention=attention,
        )


class GenericMessagePassing(nn.Module):
    """Permutation-equivariant global control without typed path composition.

    Its modules and tensor shapes mirror :class:`ProportionalPathMixer`, so the
    candidate and this control can share an exact initialization and parameter
    count. Context is built from unordered endpoint neighborhoods using only
    orientation-invariant magnitudes; the final correction is odd in the target
    edge value.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        weight_floor: float = 1e-3,
        input_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.evidence = "raw"
        self.mixer = "generic_message_passing"
        self.weight_floor = float(weight_floor)
        self.register_buffer("input_scale", torch.tensor(float(input_scale)))
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.path_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.path_score = nn.Linear(hidden_dim, 1)
        self.swap_logit = nn.Parameter(torch.zeros(()))
        self.update_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid()
        )
        self.update_value = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def _encode(self, values: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        scale = self.input_scale.to(dtype=values.dtype, device=values.device)
        return self.edge_encoder(
            torch.stack(
                (values / scale, torch.sqrt(variance.clamp_min(1e-12)) / scale),
                dim=-1,
            )
        )

    def forward(self, tensors: dict[str, torch.Tensor]) -> NeuralGraphOutput:
        values = tensors["observed_log_ratio"]
        variance = tensors["edge_variance"]
        edges = tensors["edge_index"]
        valid = tensors["edge_valid"]
        edge_state = self._encode(values.abs(), variance)
        n_nodes = int(tensors["n_nodes"])
        node_sum = edge_state.new_zeros((n_nodes, self.hidden_dim))
        node_count = edge_state.new_zeros((n_nodes, 1))
        valid_state = edge_state * valid[:, None].to(edge_state.dtype)
        valid_count = valid[:, None].to(edge_state.dtype)
        for endpoint in (0, 1):
            node_sum.index_add_(0, edges[:, endpoint], valid_state)
            node_count.index_add_(
                0,
                edges[:, endpoint],
                valid_count,
            )
        node_state = node_sum / node_count.clamp_min(1.0)
        endpoint_state = node_state[edges]
        scale = self.input_scale.to(values)
        degrees = node_count[edges].squeeze(-1) / max(float(n_nodes), 1.0)
        scalars = torch.stack(
            (
                degrees[:, 0],
                degrees[:, 1],
                values.abs() / scale,
                torch.sqrt(variance.clamp_min(1e-12)) / scale,
            ),
            dim=-1,
        )

        def features(
            endpoints: torch.Tensor, local_scalars: torch.Tensor
        ) -> torch.Tensor:
            return torch.cat(
                (edge_state, endpoints[:, 0], endpoints[:, 1], local_scalars), dim=-1
            )

        forward_message = self.path_mlp(features(endpoint_state, scalars))
        swapped_message = self.path_mlp(
            features(endpoint_state[:, [1, 0]], scalars[:, [1, 0, 2, 3]])
        )
        symmetric = 0.5 * (forward_message + swapped_message)
        difference = torch.abs(forward_message - swapped_message)
        mix = torch.sigmoid(self.swap_logit)
        message = mix * symmetric + (1.0 - mix) * difference
        score = self.path_score(message).squeeze(-1)
        message = message * torch.sigmoid(score)[:, None]
        joined = torch.cat((edge_state, message), dim=-1)
        hidden = self.norm(
            edge_state + self.update_gate(joined) * self.update_value(joined)
        )
        relative_correction = torch.tanh(self.correction_head(hidden).squeeze(-1))
        correction = values * relative_correction
        corrected = values + correction
        reliability = self.weight_floor + (1.0 - self.weight_floor) * torch.sigmoid(
            self.reliability_head(hidden).squeeze(-1)
        )
        reliability = torch.where(valid, reliability, torch.zeros_like(reliability))
        return NeuralGraphOutput(
            corrected_log_ratio=corrected,
            reliability=reliability,
            correction=correction,
            path_attention=torch.sigmoid(score),
        )


class EdgewiseMLP(nn.Module):
    """Local edge-only baseline with no path or neighborhood access."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        weight_floor: float = 1e-3,
        input_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.evidence = "raw"
        self.mixer = "edge_mlp"
        self.weight_floor = float(weight_floor)
        self.register_buffer("input_scale", torch.tensor(float(input_scale)))
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, tensors: dict[str, torch.Tensor]) -> NeuralGraphOutput:
        values = tensors["observed_log_ratio"]
        variance = tensors["edge_variance"]
        scale = self.input_scale.to(values)
        hidden = self.edge_encoder(
            torch.stack(
                (values.abs() / scale, torch.sqrt(variance.clamp_min(1e-12)) / scale),
                dim=-1,
            )
        )
        correction = values * torch.tanh(self.correction_head(hidden).squeeze(-1))
        reliability = self.weight_floor + (1.0 - self.weight_floor) * torch.sigmoid(
            self.reliability_head(hidden).squeeze(-1)
        )
        reliability = torch.where(
            tensors["edge_valid"], reliability, torch.zeros_like(reliability)
        )
        return NeuralGraphOutput(
            corrected_log_ratio=values + correction,
            reliability=reliability,
            correction=correction,
            path_attention=values.new_empty((0,)),
        )


def differentiable_wls(
    observation: PublicGraphObservation,
    values: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Solve weighted graph potentials with a mean-zero gauge in Torch."""
    device, dtype = values.device, values.dtype
    valid = torch.as_tensor(observation.edge_valid, dtype=torch.bool, device=device)
    incidence = torch.as_tensor(
        incidence_matrix(observation.n_nodes, observation.edge_index),
        dtype=dtype,
        device=device,
    )[valid]
    valid_values = values[valid]
    valid_weights = weights[valid].clamp_min(1e-6)
    valid_weights = valid_weights / valid_weights.mean()
    laplacian = incidence.T @ (valid_weights[:, None] * incidence)
    rhs = incidence.T @ (valid_weights * valid_values)
    ones = torch.ones((observation.n_nodes, 1), dtype=dtype, device=device)
    zero = torch.zeros((1, 1), dtype=dtype, device=device)
    kkt = torch.cat(
        (torch.cat((laplacian, ones), dim=1), torch.cat((ones.T, zero), dim=1)), dim=0
    )
    solution = torch.linalg.solve(
        kkt, torch.cat((rhs, torch.zeros(1, dtype=dtype, device=device)))
    )
    return solution[:-1]


def direct_centered_decoder(
    observation: PublicGraphObservation,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    weight_floor: float = 1e-3,
) -> np.ndarray:
    """Decode node potentials by one weighted incidence aggregation, without a solve.

    This deliberately weak readout is the direct-decoder control: every edge
    votes once for its two endpoints, incident weight normalizes each node, and
    the resulting node vector is centered to fix the additive gauge.
    """
    relation = np.asarray(values, dtype=np.float64)
    reliability = np.asarray(weights, dtype=np.float64)
    if relation.shape != (len(observation.edge_index),):
        raise ValueError("values must have one entry per edge")
    if reliability.shape != relation.shape:
        raise ValueError("weights must have one entry per edge")
    valid = np.asarray(observation.edge_valid, dtype=bool)
    incidence = incidence_matrix(observation.n_nodes, observation.edge_index)[valid]
    valid_weights = np.maximum(reliability[valid], float(weight_floor))
    numerator = incidence.T @ (valid_weights * relation[valid])
    denominator = np.abs(incidence).T @ valid_weights
    decoded = numerator / np.maximum(denominator, float(weight_floor))
    return decoded - decoded.mean()


def local_closure_loss(
    observation: PublicGraphObservation,
    values: torch.Tensor,
) -> torch.Tensor:
    if len(observation.path_index) == 0:
        return values.new_zeros(())
    index = torch.as_tensor(
        observation.path_index, dtype=torch.long, device=values.device
    )
    signs = torch.as_tensor(
        observation.path_sign, dtype=values.dtype, device=values.device
    )
    valid = torch.as_tensor(
        observation.path_valid, dtype=torch.bool, device=values.device
    )
    edge_valid = torch.as_tensor(
        observation.edge_valid, dtype=torch.bool, device=values.device
    )
    valid &= torch.all(edge_valid[index], dim=1)
    signed = values[index[valid]] * signs[valid]
    residual = signed[:, 0] - signed[:, 1] - signed[:, 2]
    return residual.abs().mean() if len(residual) else values.new_zeros(())


def exact_closure_only(
    observation: PublicGraphObservation,
    *,
    weight_floor: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Parameter-free local path median and closure-derived reliability."""
    values = np.asarray(observation.observed_log_ratio, dtype=np.float64)
    corrected = values.copy()
    reliability = np.ones_like(values)
    if len(observation.path_index) == 0:
        return corrected, reliability
    valid = observation.path_valid & np.all(
        observation.edge_valid[observation.path_index], axis=1
    )
    index = observation.path_index[valid]
    signs = observation.path_sign[valid]
    signed = values[index] * signs
    path_estimate = (signed[:, 1] + signed[:, 2]) / signs[:, 0]
    noise_scale = max(float(np.sqrt(np.median(observation.edge_variance))), 1e-8)
    for target in np.unique(index[:, 0]):
        candidates = path_estimate[index[:, 0] == target]
        combined = np.concatenate(([values[target]], candidates))
        corrected[target] = float(np.median(combined))
        discrepancy = float(np.median(np.abs(candidates - values[target])))
        reliability[target] = max(
            weight_floor, float(np.exp(-discrepancy / noise_scale))
        )
    return corrected, reliability
