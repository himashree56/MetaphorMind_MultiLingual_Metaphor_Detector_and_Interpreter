"""
Quick test script to check if models can be loaded
Run this before starting the main server to diagnose issues
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_BASE_PATH = Path(__file__).parent.parent / "models"

def test_model_loading():
    """Test if models can be loaded"""
    languages = ['hindi', 'tamil', 'telugu', 'kannada']
    
    print("\n" + "="*60)
    print("Testing Model Loading")
    print("="*60 + "\n")
    
    for lang in languages:
        model_path = MODEL_BASE_PATH / f"{lang}_model"
        
        print(f"\n[{lang.upper()}]")
        print(f"Path: {model_path}")
        print(f"Exists: {model_path.exists()}")
        
        if not model_path.exists():
            print(f"❌ Model directory not found!")
            continue
        
        # Check for required files
        required_files = ['config.json', 'tokenizer_config.json']
        print(f"\nChecking required files:")
        for file in required_files:
            file_path = model_path / file
            exists = file_path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {file}")
        
        # Try loading tokenizer
        try:
            print(f"\nLoading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            print(f"✓ Tokenizer loaded successfully")
            print(f"  Vocab size: {len(tokenizer)}")
        except Exception as e:
            print(f"❌ Tokenizer failed: {str(e)}")
            continue
        
        # Try loading model
        try:
            print(f"Loading model...")
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path),
                num_labels=2
            )
            model.eval()
            print(f"✓ Model loaded successfully")
            
            # Get model size
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {total_params:,}")
            print(f"  Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
            
        except Exception as e:
            print(f"❌ Model failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        test_model_loading()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
