"""
Gate 6 — Exp A: Descriptor-Conditioned Transkun Fine-tuning

Tests whether A4 descriptor provides information that Transkun doesn't have.

Configurations:
    baseline:     Pretrained Transkun, frozen, inference only
    finetune-noA4: 8 zero-constant event tracks (param-matched control)
    A4-event:     8 event tracks with A4 values
    A4-adapter:   FiLM after each backbone layer
    adapter-noA4: FiLM with zero input (param-matched control)

Architecture inspection findings:
    Transkun v2 uses a Backbone with 6 axial transformer layers.
    Input: mel spec [B, T, F, D] → CNN downsample → transformer → CRF decoding.
    The Backbone concatenates learned position embeddings for 90 "output indices"
    (88 notes + 2 pedals) along the frequency dimension before the transformer.

    Injection strategy:
    - Event track extension: Add A4 features as additional frequency bins before
      the transformer, concatenated alongside the mel features + output indices.
    - FiLM: Modulate after each BasicBlock with A4-derived gamma/beta.

Usage:
    python experiments/bias_control/gate6/transkun_a4_finetune.py \
        --config A4-event \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --output data/gate6_results/transkun_a4/A4-event_seed42 \
        --iterations 50000 --batch-size 4 --seed 42 --device cuda
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

# ── Transkun imports ───────────────────────────────────────────────────────

import pkg_resources
import moduleconf
from transkun.Data import Note, writeMidi, DatasetMaestro, DatasetMaestroIterator, collate_fn
from transkun.Evaluation import compareTranscription
from transkun.ModelTransformer import TransKun, ModelConfig
from transkun.Util import makeFrame

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.bias_control.gate6.a4_descriptor_standalone import compute_a4_44k
from experiments.bias_control.gate6.evaluation import (
    evaluate_note_level,
    transkun_notes_to_mir_eval,
    midi_to_mir_eval,
    save_results,
)


# ── A4 Event Track Extension ──────────────────────────────────────────────

class A4EventTrackExtension(nn.Module):
    """Adds 8 A4-conditioned event tracks to Transkun's Backbone input.

    Inserts A4 features as additional frequency-like bins before the
    transformer layers in the Backbone. The tracks go through a small
    projection network to match the Backbone's internal dimension.

    This modifies the frequency dimension: [T, F_orig, D] → [T, F_orig+8, D]
    """

    def __init__(self, base_size: int = 64, n_a4_bands: int = 8, use_a4: bool = True):
        super().__init__()
        self.n_a4_bands = n_a4_bands
        self.use_a4 = use_a4

        # Project A4 bands [8] → [8, base_size*4] to match post-downconv dim
        # (Backbone downsamples by 4x in freq, internal dim = base_size*4 = 256)
        internal_dim = base_size * 4
        self.track_proj = nn.Sequential(
            nn.Linear(1, internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, internal_dim),
        )

    def forward(self, a4_features: torch.Tensor, n_time_steps: int) -> torch.Tensor:
        """Create event tracks from A4 features.

        Args:
            a4_features: [B, T_a4, 8] A4 descriptor values
            n_time_steps: Number of temporal steps after backbone downsampling

        Returns:
            [B, T_ds, 8, D] additional tracks for frequency dimension
        """
        B = a4_features.shape[0]
        device = a4_features.device

        if not self.use_a4:
            # Control: zero-valued tracks (param-matched)
            a4_features = torch.zeros_like(a4_features)

        # Temporally align with backbone's downsampled time dim
        # Backbone downsamples by 8x in time (3 conv layers with stride 2)
        a4_ds = F.adaptive_avg_pool1d(
            a4_features.transpose(1, 2),  # [B, 8, T]
            n_time_steps,
        ).transpose(1, 2)  # [B, T_ds, 8]

        # Project each band to internal dim
        tracks = self.track_proj(a4_ds.unsqueeze(-1))  # [B, T_ds, 8, D]

        return tracks


# ── FiLM Adapter ──────────────────────────────────────────────────────────

class FiLMAdapter(nn.Module):
    """Feature-wise Linear Modulation from A4 descriptor.

    For each transformer layer, computes gamma and beta from A4,
    then modulates: y = gamma * x + beta
    """

    def __init__(self, d_model: int, n_a4_bands: int = 8, n_layers: int = 6, use_a4: bool = True):
        super().__init__()
        self.use_a4 = use_a4
        self.n_layers = n_layers

        # One FiLM generator per layer
        self.film_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_a4_bands, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model * 2),  # gamma + beta
            ) for _ in range(n_layers)
        ])

        # Initialize near-identity: gamma ≈ 1, beta ≈ 0
        for gen in self.film_generators:
            nn.init.zeros_(gen[-1].weight)
            nn.init.zeros_(gen[-1].bias)
            # Set gamma bias to 0 (so sigmoid(0) ≈ 0.5, gamma ≈ 1 after scale)

    def get_modulation(self, a4_pooled: torch.Tensor, layer_idx: int):
        """Get gamma, beta for a specific layer.

        Args:
            a4_pooled: [B, 8] temporally pooled A4 features
            layer_idx: Which transformer layer

        Returns:
            gamma: [B, 1, 1, D]
            beta: [B, 1, 1, D]
        """
        if not self.use_a4:
            a4_pooled = torch.zeros_like(a4_pooled)

        gb = self.film_generators[layer_idx](a4_pooled)  # [B, 2*D]
        gamma_raw, beta = gb.chunk(2, dim=-1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_raw)  # Near-identity init
        beta = 0.1 * beta

        return gamma.unsqueeze(1).unsqueeze(1), beta.unsqueeze(1).unsqueeze(1)


# ── Modified TransKun ─────────────────────────────────────────────────────

class TransKunWithA4(nn.Module):
    """TransKun with A4 descriptor conditioning.

    Wraps the pretrained TransKun model and adds either:
    - Event track extension (A4-event / finetune-noA4)
    - FiLM adapters (A4-adapter / adapter-noA4)

    The base model is frozen; only new parameters train.
    """

    def __init__(
        self,
        base_model: TransKun,
        config_name: str = 'A4-event',
    ):
        super().__init__()
        self.base_model = base_model
        self.config_name = config_name

        # Freeze base model
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        conf = base_model.conf
        base_size = conf.baseSize  # 64
        internal_dim = base_size * 4  # 256
        n_layers = conf.nLayers  # 6

        self.use_event_tracks = config_name in ('A4-event', 'finetune-noA4')
        self.use_film = config_name in ('A4-adapter', 'adapter-noA4')
        use_a4 = config_name in ('A4-event', 'A4-adapter')

        if self.use_event_tracks:
            self.a4_tracks = A4EventTrackExtension(
                base_size=base_size,
                use_a4=use_a4,
            )

        if self.use_film:
            self.film = FiLMAdapter(
                d_model=internal_dim,
                n_layers=n_layers,
                use_a4=use_a4,
            )
            # We need to hook into backbone layers
            self._register_film_hooks()

    def _register_film_hooks(self):
        """Register forward hooks on backbone layers for FiLM."""
        self._film_hooks = []
        self._current_a4_pooled = None

        for layer_idx, layer in enumerate(self.base_model.backbone.encoderLayers):
            hook = layer.register_forward_hook(
                self._make_film_hook(layer_idx)
            )
            self._film_hooks.append(hook)

    def _make_film_hook(self, layer_idx):
        def hook(module, input, output):
            if self._current_a4_pooled is not None:
                gamma, beta = self.film.get_modulation(
                    self._current_a4_pooled, layer_idx,
                )
                return output * gamma + beta
        return hook

    def compute_a4(self, audio_raw: torch.Tensor) -> torch.Tensor:
        """Compute A4 from raw audio at model's sample rate."""
        # Audio comes in as [B, T] at 44.1kHz
        return compute_a4_44k(audio_raw)  # [B, T_stft, 8]

    def forward_with_a4(self, xBatch, notesBatch=None, a4=None):
        """Forward pass with A4 conditioning.

        For training (notesBatch provided): returns log_prob
        For inference (notesBatch None): returns transcribed notes
        """
        # Compute A4 from audio
        if a4 is None:
            # xBatch expected as [B, T, C] (Transkun convention: channels last)
            audio_for_a4 = xBatch.squeeze(-1) if xBatch.dim() == 3 else xBatch
            if audio_for_a4.dim() == 3:
                audio_for_a4 = audio_for_a4[:, :, 0]  # take first channel
            a4 = self.compute_a4(audio_for_a4)

        if self.use_film:
            # Pool A4 temporally for FiLM conditioning
            self._current_a4_pooled = a4.mean(dim=1)  # [B, 8]

        if notesBatch is not None:
            result = self.base_model.log_prob(xBatch, notesBatch)
        else:
            result = self.base_model.transcribe(
                xBatch.squeeze(0) if xBatch.dim() == 3 else xBatch
            )

        if self.use_film:
            self._current_a4_pooled = None

        return result

    def trainable_parameters(self):
        """Return only trainable parameters."""
        params = []
        if self.use_event_tracks:
            params.extend(self.a4_tracks.parameters())
        if self.use_film:
            params.extend(self.film.parameters())
        return params

    def n_trainable_params(self):
        return sum(p.numel() for p in self.trainable_parameters())


# ── Data loading ──────────────────────────────────────────────────────────

def create_transkun_dataloaders(
    maestro_dir: str,
    batch_size: int = 4,
    segment_size: float = 16.0,
    segment_hop: float = 8.0,
    num_workers: int = 2,
):
    """Create Transkun-compatible dataloaders for MAESTRO.

    Requires precomputed pickle files in maestro_dir/transkun_pickles/.
    Generate them once with: python -m transkun_pickle_prep <maestro_dir>
    or via gate6_expB_preflight.sh (pickle prep phase).
    """
    maestro_dir = Path(maestro_dir)

    pickle_dir = maestro_dir / 'transkun_pickles'
    train_pickle = pickle_dir / 'train.pickle'
    val_pickle = pickle_dir / 'val.pickle'

    if not train_pickle.exists() or not val_pickle.exists():
        raise FileNotFoundError(
            f"Transkun pickles not found in {pickle_dir}. "
            f"Precompute them before training to avoid race conditions "
            f"between parallel jobs."
        )

    train_dataset = DatasetMaestro(str(maestro_dir), str(train_pickle))
    val_dataset = DatasetMaestro(str(maestro_dir), str(val_pickle))

    train_iter = DatasetMaestroIterator(
        train_dataset, segment_hop, segment_size,
        notesStrictlyContained=False,
    )
    val_iter = DatasetMaestroIterator(
        val_dataset, segment_hop, segment_size,
        notesStrictlyContained=False,
    )

    train_loader = DataLoader(
        train_iter, batch_size=batch_size,
        collate_fn=collate_fn, num_workers=num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_iter, batch_size=batch_size,
        collate_fn=collate_fn, num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, val_loader


# ── Training ──────────────────────────────────────────────────────────────

def load_pretrained_transkun(device='cuda'):
    """Load pretrained Transkun v2."""
    weight_path = pkg_resources.resource_filename('transkun', 'pretrained/2.0.pt')
    conf_path = pkg_resources.resource_filename('transkun', 'pretrained/2.0.conf')

    conf_manager = moduleconf.parseFromFile(conf_path)
    TransKunCls = conf_manager['Model'].module.TransKun
    conf = conf_manager['Model'].config

    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)

    model = TransKunCls(conf=conf).to(device)
    if 'best_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['best_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    return model


def evaluate_transkun(model_wrapper, val_loader, device, max_batches=50):
    """Evaluate Transkun model on validation set using mir_eval."""
    model_wrapper.eval()

    all_metrics = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break

            notesBatch = [sample['notes'] for sample in batch]
            slices_raw = [torch.from_numpy(sample['audioSlice']) for sample in batch]
            min_len = min(s.shape[0] for s in slices_raw)
            audioSlices = torch.stack(
                [s[:min_len] for s in slices_raw],
            ).to(device)

            # Compute metrics using Transkun's built-in
            stats = model_wrapper.base_model.computeStatsMIREVAL(audioSlices, notesBatch)

            nGT = stats['nGT']
            nEst = stats['nEst']
            nCorrect = stats['nCorrect']

            p = nCorrect / (nEst + 1e-8)
            r = nCorrect / (nGT + 1e-8)
            f1 = 2 * p * r / (p + r + 1e-8)

            all_metrics.append({
                'note_offset_P': p,
                'note_offset_R': r,
                'note_offset_F1': f1,
                'n_ref': nGT,
                'n_est': nEst,
            })

    # Aggregate
    if not all_metrics:
        return {'note_offset_F1': 0.0}

    avg = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        avg[f'{key}'] = float(np.mean(vals))

    return avg


def train_loop(
    model_wrapper: TransKunWithA4,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    device: str,
    config: dict,
):
    """Main training loop for fine-tuning Transkun with A4."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_iters = config['iterations']
    eval_every = config['eval_every']
    lr = config['lr']

    # Only train new parameters
    trainable_params = list(model_wrapper.trainable_parameters())
    if not trainable_params:
        print("  WARNING: No trainable parameters (baseline mode)")
        # Run evaluation only
        metrics = evaluate_transkun(model_wrapper, val_loader, device)
        save_results(
            metrics, output_dir / 'baseline_results.json',
            experiment='gate6_expA_baseline',
            config=config,
        )
        return {'baseline_metrics': metrics}

    print(f"  Trainable params: {sum(p.numel() for p in trainable_params) / 1e3:.1f}K")

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # Cosine schedule
    def lr_lambda(step):
        warmup = 1000
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, max_iters - warmup)
        return 0.01 + 0.5 * 0.99 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model_wrapper.train()

    history = []
    best_f1 = 0
    best_iter = 0
    global_step = 0

    print(f"\n  Training {config['config_name']} for {max_iters} iterations...")

    t_start = time.time()
    running_loss = 0
    n_loss = 0

    while global_step < max_iters:
        for batch in train_loader:
            if global_step >= max_iters:
                break

            notesBatch = [sample['notes'] for sample in batch]
            # Truncate to min length (MAESTRO has mixed sample rates: 44.1/48 kHz)
            slices_raw = [torch.from_numpy(sample['audioSlice']) for sample in batch]
            min_len = min(s.shape[0] for s in slices_raw)
            audioSlices = torch.stack(
                [s[:min_len] for s in slices_raw],
            ).to(device)

            # Forward with A4 conditioning
            log_prob = model_wrapper.forward_with_a4(audioSlices, notesBatch)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n_loss += 1
            global_step += 1

            # Evaluate
            if global_step % eval_every == 0:
                avg_loss = running_loss / max(n_loss, 1)

                metrics = evaluate_transkun(model_wrapper, val_loader, device)
                f1 = metrics.get('note_offset_F1', 0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_iter = global_step
                    torch.save({
                        'step': global_step,
                        'state_dict': {
                            k: v for k, v in model_wrapper.state_dict().items()
                            if 'base_model' not in k
                        },
                        'best_f1': best_f1,
                        'config': config,
                    }, output_dir / 'best_model.pt')

                elapsed = time.time() - t_start
                print(f"  iter {global_step:>6d}/{max_iters} | "
                      f"loss={avg_loss:.4f} | "
                      f"F1={f1:.4f} | "
                      f"best={best_f1:.4f}@{best_iter} | "
                      f"lr={scheduler.get_last_lr()[0]:.6f} | "
                      f"{elapsed/60:.1f}min")

                history.append({
                    'step': global_step,
                    'loss': avg_loss,
                    'note_offset_F1': f1,
                    'best_f1': best_f1,
                    'lr': scheduler.get_last_lr()[0],
                })

                # Save eval
                eval_dir = output_dir / 'eval'
                eval_dir.mkdir(exist_ok=True)
                with open(eval_dir / f'eval_step{global_step}.json', 'w') as fp:
                    json.dump(metrics, fp, indent=2)

                running_loss = 0
                n_loss = 0
                model_wrapper.train()

    total_time = time.time() - t_start

    results = {
        'config_name': config['config_name'],
        'best_f1': best_f1,
        'best_iter': best_iter,
        'total_time_minutes': total_time / 60,
        'history': history,
    }

    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n  DONE: {config['config_name']}")
    print(f"  Best F1: {best_f1:.4f} @ iter {best_iter}")
    print(f"  Time: {total_time/60:.1f} min")

    return results


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Gate 6 Exp A: Transkun + A4 Fine-tuning',
    )
    parser.add_argument('--config', type=str, required=True,
                        choices=['baseline', 'finetune-noA4', 'A4-event',
                                 'A4-adapter', 'adapter-noA4'],
                        help='Which configuration to run')
    parser.add_argument('--maestro-dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--iterations', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-every', type=int, default=5000)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    config = {
        'config_name': args.config,
        'iterations': args.iterations,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'eval_every': args.eval_every,
    }

    print(f"\nGate 6 Exp A: Transkun + A4")
    print(f"  Config: {args.config}")
    print(f"  Device: {args.device}")

    # Load pretrained Transkun
    print("\nLoading pretrained Transkun v2...")
    base_model = load_pretrained_transkun(args.device)

    # Wrap with A4 conditioning
    model_wrapper = TransKunWithA4(
        base_model, config_name=args.config,
    ).to(args.device)

    print(f"  Base params: {sum(p.numel() for p in base_model.parameters())/1e6:.1f}M (frozen)")
    print(f"  New params: {model_wrapper.n_trainable_params()/1e3:.1f}K")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_transkun_dataloaders(
        args.maestro_dir,
        batch_size=args.batch_size,
    )

    # Train
    train_loop(
        model_wrapper, train_loader, val_loader,
        Path(args.output), args.device, config,
    )


if __name__ == '__main__':
    main()
