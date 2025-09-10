"""
Integration Test for DTJ-VIB System

This script tests the complete DTJ-VIB pipeline to ensure all components
work together correctly.
"""

import os
import sys
import torch
import logging

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import DTJ-VIB components
from model.DT_JSCC_VIB import DTJSCC_VIB, DTJSCC_VIB_Small, create_dtj_vib_model
from model.losses import RIBLoss
from datasets.tactile_dataloader import get_data, VibrationDataset, TactileSignalProcessor
from utils.modulation import QAM, PSK
from utils.accuracy import accuracy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_signal_processor():
    """Test the signal processor for converting 1D signals to 2D pseudo-images."""
    print("\n=== Testing Signal Processor ===")
    
    processor = TactileSignalProcessor(normalize=True, augment=False)
    
    # Create test signal
    test_signal = torch.randn(512)
    
    try:
        # Convert to pseudo-image
        pseudo_image = processor.signal_to_pseudo_image(test_signal)
        
        print(f"✓ Signal shape: {test_signal.shape}")
        print(f"✓ Pseudo-image shape: {pseudo_image.shape}")
        print(f"✓ Value range: [{pseudo_image.min().item():.3f}, {pseudo_image.max().item():.3f}]")
        
        # Test batch processing
        batch_signals = torch.randn(4, 512)
        batch_pseudo_images = torch.stack([
            processor.signal_to_pseudo_image(signal) 
            for signal in batch_signals
        ])
        
        print(f"✓ Batch pseudo-images shape: {batch_pseudo_images.shape}")
        print("✓ Signal processor test passed!")
        
        return True
        
    except Exception as e:
        print(f"✗ Signal processor test failed: {e}")
        return False


def test_model_architectures():
    """Test DTJ-VIB model architectures."""
    print("\n=== Testing Model Architectures ===")
    
    # Test input
    batch_size = 4
    test_input = torch.randn(batch_size, 2, 16, 16)
    
    try:
        # Test standard model
        print("Testing DTJSCC_VIB (standard)...")
        model_std = DTJSCC_VIB(
            in_channels=2,
            latent_channels=256,
            out_classes=10,
            num_embeddings=16
        )
        
        # Create a simple modulation for testing
        from utils.modulation import PSK
        test_mod = PSK(M=16, PSNR=10.0)
        
        model_std.train()
        output_std, dist_std = model_std(test_input, mod=test_mod)
        
        print(f"✓ Standard model output shape: {output_std.shape}")
        print(f"✓ Standard model dist shape: {dist_std.shape}")
        print(f"✓ Standard model parameters: {sum(p.numel() for p in model_std.parameters()):,}")
        
        # Test small model
        print("Testing DTJSCC_VIB_Small...")
        model_small = DTJSCC_VIB_Small(
            in_channels=2,
            latent_channels=128,
            out_classes=10,
            num_embeddings=16
        )
        
        model_small.train()
        output_small, dist_small = model_small(test_input, mod=test_mod)
        
        print(f"✓ Small model output shape: {output_small.shape}")
        print(f"✓ Small model dist shape: {dist_small.shape}")
        print(f"✓ Small model parameters: {sum(p.numel() for p in model_small.parameters()):,}")
        
        # Test factory function
        print("Testing model factory...")
        model_factory = create_dtj_vib_model(
            model_type='standard',
            in_channels=2,
            latent_channels=256,
            out_classes=10
        )
        
        print("✓ Model architectures test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model architectures test failed: {e}")
        return False


def test_modulation():
    """Test modulation and demodulation."""
    print("\n=== Testing Modulation ===")
    
    try:
        # Test PSK modulation
        print("Testing PSK modulation...")
        psk_mod = PSK(M=16, PSNR=10.0)
        
        # Test symbols
        test_symbols = torch.randint(0, 16, (10,))
        modulated = psk_mod.modulate(test_symbols)
        noisy = psk_mod.awgn(modulated)
        demodulated = torch.tensor(psk_mod.demodulate(noisy.numpy()))
        
        print(f"✓ PSK symbols: {test_symbols[:5].tolist()}...")
        print(f"✓ PSK modulated shape: {modulated.shape}")
        print(f"✓ PSK demodulated: {demodulated[:5].tolist()}...")
        
        # Test QAM modulation
        print("Testing QAM modulation...")
        qam_mod = QAM(M=16, PSNR=10.0)
        
        modulated_qam = qam_mod.modulate(test_symbols)
        noisy_qam = qam_mod.awgn(modulated_qam)
        demodulated_qam = qam_mod.demodulate(noisy_qam)
        
        print(f"✓ QAM modulated shape: {modulated_qam.shape}")
        print(f"✓ QAM demodulated: {demodulated_qam[:5].tolist()}...")
        
        print("✓ Modulation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Modulation test failed: {e}")
        return False


def test_loss_function():
    """Test the RIB loss function."""
    print("\n=== Testing Loss Function ===")
    
    try:
        # Create test data
        batch_size = 4
        num_classes = 10
        num_embeddings = 16
        
        # Fake model outputs (with gradients)
        outputs = torch.randn(batch_size, num_classes, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # Fake distribution (from quantizer, with gradients)
        dist_logits = torch.randn(batch_size * 4, num_embeddings, requires_grad=True)
        dist = torch.softmax(dist_logits, dim=-1)
        
        # Test loss
        criterion = RIBLoss(lam=0.001)
        loss, loss_post, loss_entr = criterion(dist, outputs, targets)
        
        print(f"✓ Total loss: {loss.item():.4f}")
        print(f"✓ Classification loss: {loss_post.item():.4f}")
        print(f"✓ Entropy loss: {loss_entr.item():.4f}")
        
        # Test backward pass
        loss.backward()
        
        print("✓ Loss function test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        return False


def test_accuracy_computation():
    """Test accuracy computation."""
    print("\n=== Testing Accuracy Computation ===")
    
    try:
        # Create test predictions and targets
        batch_size = 10
        num_classes = 10
        
        # Perfect predictions
        outputs_perfect = torch.eye(num_classes)[:batch_size] * 10  # High confidence
        targets = torch.arange(batch_size) % num_classes
        
        acc_perfect = accuracy(outputs_perfect, targets, (1, 3))
        
        print(f"✓ Perfect predictions - Top1: {acc_perfect[0].item():.1f}%, Top3: {acc_perfect[1].item():.1f}%")
        
        # Random predictions
        outputs_random = torch.randn(batch_size, num_classes)
        acc_random = accuracy(outputs_random, targets, (1, 3))
        
        print(f"✓ Random predictions - Top1: {acc_random[0].item():.1f}%, Top3: {acc_random[1].item():.1f}%")
        
        print("✓ Accuracy computation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Accuracy computation test failed: {e}")
        return False


def test_end_to_end_pipeline():
    """Test the complete end-to-end pipeline."""
    print("\n=== Testing End-to-End Pipeline ===")
    
    try:
        # Create model
        model = DTJSCC_VIB_Small(
            in_channels=2,
            latent_channels=128,
            out_classes=10,
            num_embeddings=16
        )
        
        # Create test data (simulating pseudo-images from vibration signals)
        batch_size = 4
        test_pseudo_images = torch.randn(batch_size, 2, 16, 16)
        test_labels = torch.randint(0, 10, (batch_size,))
        
        # Create modulation
        mod = PSK(M=16, PSNR=8.0)
        
        # Forward pass
        model.train()
        outputs, dist = model(test_pseudo_images, mod=mod)
        
        # Compute loss
        criterion = RIBLoss(lam=0.001)
        loss, loss_post, loss_entr = criterion(dist, outputs, test_labels)
        
        # Compute accuracy
        acc = accuracy(outputs, test_labels, (1, 3))
        
        print(f"✓ Pipeline input shape: {test_pseudo_images.shape}")
        print(f"✓ Pipeline output shape: {outputs.shape}")
        print(f"✓ Pipeline loss: {loss.item():.4f}")
        print(f"✓ Pipeline accuracy: {acc[0].item():.1f}%")
        
        # Test backward pass
        loss.backward()
        
        # Check gradients
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"✓ Gradient norm: {grad_norm:.4f}")
        
        print("✓ End-to-end pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"✗ End-to-end pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("=== DTJ-VIB Integration Test Suite ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    tests = [
        ("Signal Processor", test_signal_processor),
        ("Model Architectures", test_model_architectures),
        ("Modulation", test_modulation),
        ("Loss Function", test_loss_function),
        ("Accuracy Computation", test_accuracy_computation),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*50)
    print("INTEGRATION TEST RESULTS")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("-"*50)
    print(f"PASSED: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! DTJ-VIB system is ready for use.")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
