#!/usr/bin/env python3
"""Audit: Test bias_control components individually."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch


def test_midi_encoder():
    """Test MIDIEncoder without HuggingFace."""
    from src.bias_control.encoders.midi_encoder import MIDIEncoder, duration_to_bucket

    # Test duration_to_bucket
    durations = torch.tensor([0.1, 0.5, 1.0, 2.0])
    buckets = duration_to_bucket(durations)
    assert buckets.shape == (4,), f"Expected (4,), got {buckets.shape}"
    assert buckets.min() >= 0, "Buckets should be non-negative"
    assert buckets.max() < 32, "Buckets should be < 32"
    print("  duration_to_bucket: ✅")

    # Test encoder
    encoder = MIDIEncoder(embed_dim=512, n_layers=4, n_heads=8)
    pitch = torch.randint(0, 128, (2, 50))
    velocity = torch.randint(0, 128, (2, 50))
    duration = torch.randint(0, 32, (2, 50))

    output = encoder(pitch, velocity, duration)
    assert output.shape == (2, 512), f"Expected (2, 512), got {output.shape}"
    print("  MIDIEncoder forward: ✅")

    return True


def test_mert_lite():
    """Test MERTEncoderLite without HuggingFace."""
    from src.bias_control.encoders.mert_encoder import MERTEncoderLite

    # Use smaller output dim and shorter audio for test
    encoder = MERTEncoderLite(output_dim=256, n_layers=2, n_heads=4)
    waveform = torch.randn(2, 24000 * 2)  # 2 seconds at 24kHz (smaller for test)

    output = encoder(waveform)
    assert output.shape == (2, 256), f"Expected (2, 256), got {output.shape}"
    print("  MERTEncoderLite forward: ✅")

    return True


def test_projection():
    """Test ProjectionHead."""
    from src.bias_control.encoders.projection import ProjectionHead

    proj = ProjectionHead(input_dim=1024, hidden_dim=512, output_dim=256)
    x = torch.randn(4, 1024)

    output = proj(x)
    assert output.shape == (4, 256), f"Expected (4, 256), got {output.shape}"
    print("  ProjectionHead forward: ✅")

    return True


def test_dann():
    """Test DANNLoss and GradientReversalLayer."""
    from src.bias_control.losses.dann import DANNLoss, GradientReversalLayer

    # GRL test
    grl = GradientReversalLayer(lambda_=1.0)
    x = torch.randn(4, 256, requires_grad=True)
    y = grl(x)
    assert y.shape == x.shape, f"GRL shape mismatch"
    print("  GradientReversalLayer: ✅")

    # DANN loss test
    dann = DANNLoss(input_dim=256, hidden_dim=64)
    embeddings = torch.randn(8, 256)
    labels = torch.cat([torch.zeros(4), torch.ones(4)]).long()

    loss, metrics = dann(embeddings, labels)
    assert loss.item() > 0, "Loss should be positive"
    assert 'domain_accuracy' in metrics, "Missing domain_accuracy in metrics"
    print("  DANNLoss forward: ✅")

    return True


def test_cross_modal_model():
    """Test CrossModalModel with MERTEncoderLite."""
    from src.bias_control.architectures.cross_modal_model import CrossModalModel

    # Use smaller config for test
    model = CrossModalModel(
        audio_encoder='lite',  # NO HuggingFace
        audio_hidden_dim=256,  # Smaller for test
        midi_embed_dim=256,
        midi_n_layers=2,
        proj_hidden_dim=128,
        proj_output_dim=64,
        use_dann=False,
    )

    # Dummy data - shorter audio for memory
    audio = torch.randn(2, 24000 * 2)  # 2 seconds
    midi_pitch = torch.randint(0, 128, (2, 50))
    midi_velocity = torch.randint(0, 128, (2, 50))
    midi_duration = torch.randint(0, 32, (2, 50))
    midi_mask = torch.zeros(2, 50, dtype=torch.bool)

    audio_emb, midi_emb = model(
        audio=audio,
        midi_pitch=midi_pitch,
        midi_velocity=midi_velocity,
        midi_duration=midi_duration,
        midi_mask=midi_mask,
    )

    assert audio_emb.shape == (2, 64), f"Expected (2, 64), got {audio_emb.shape}"
    assert midi_emb.shape == (2, 64), f"Expected (2, 64), got {midi_emb.shape}"
    print("  CrossModalModel forward: ✅")

    # Test VICReg loss
    loss, metrics = model.compute_vicreg_loss(audio_emb, midi_emb)
    assert loss.item() > 0, "VICReg loss should be positive"
    assert 'inv_loss' in metrics, "Missing inv_loss"
    print("  CrossModalModel VICReg loss: ✅")

    return True


def main():
    print("=" * 60)
    print("BIAS_CONTROL COMPONENT AUDIT")
    print("=" * 60)

    tests = [
        ("MIDIEncoder", test_midi_encoder),
        ("MERTEncoderLite", test_mert_lite),
        ("ProjectionHead", test_projection),
        ("DANNLoss", test_dann),
        ("CrossModalModel", test_cross_modal_model),
    ]

    all_passed = True
    for name, test_fn in tests:
        print(f"\nTesting {name}...")
        try:
            test_fn()
            print(f"  {name}: ✅ PASSED")
        except Exception as e:
            print(f"  {name}: ❌ FAILED - {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("STATUS: ✅ ALL COMPONENT TESTS PASSED")
        return 0
    else:
        print("STATUS: ❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
