# Download Fake and Real News Dataset from Kaggle
# This script will download and extract the dataset automatically.
# Prerequisites:
# 1. Install kaggle: pip install kaggle
# 2. Go to https://www.kaggle.com/settings/account
# 3. Click "Create New API Token" (downloads kaggle.json)
# 4. Place kaggle.json in user home .kaggle folder
# 5. Run this script

import os
import subprocess
import pandas as pd
from pathlib import Path

def setup_kaggle():
    """Verify Kaggle API is installed and configured."""
    try:
        import kaggle
        print("✓ Kaggle library is installed")
    except ImportError:
        print("Installing Kaggle CLI...")
        subprocess.check_call(['pip', 'install', 'kaggle'])
        print("✓ Kaggle installed")
    
    # Check if credentials exist
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        print("ERROR: Kaggle API credentials not found!")
        print("To fix this:")
        print("1. Go to: https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token'")
        print("3. This downloads 'kaggle.json'")
        cred_path = str(Path.home() / '.kaggle')
        print(f"4. Move it to: {cred_path}")
        print("5. Run this script again")
        return False
    
    print("✓ Kaggle credentials found")
    return True


def download_dataset():
    """Download the Fake and Real News Dataset from Kaggle."""
    dataset_dir = 'dataset'
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("\n📥 Downloading dataset from Kaggle...")
    print("   Dataset: Fake and Real News Dataset")
    print("   URL: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset")
    
    try:
        # Download dataset
        subprocess.check_call([
            'kaggle', 'datasets', 'download',
            '-d', 'clmentbisaillon/fake-and-real-news-dataset',
            '-p', dataset_dir,
            '--unzip'
        ])
        print("✓ Dataset downloaded and extracted!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR downloading dataset: {e}")
        return False


def merge_datasets():
    """Merge Fake.csv and True.csv into news.csv."""
    dataset_dir = 'dataset'
    fake_path = os.path.join(dataset_dir, 'Fake.csv')
    true_path = os.path.join(dataset_dir, 'True.csv')
    output_path = os.path.join(dataset_dir, 'news.csv')
    
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print("ERROR: Dataset files not found!")
        print(f"   Expected: {fake_path}")
        print(f"   Expected: {true_path}")
        return False
    
    print("\n🔄 Merging dataset files...")
    
    # Load datasets
    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(true_path)
    
    print(f"   Fake articles: {len(fake_df)}")
    print(f"   Real articles: {len(real_df)}")
    
    # Add label column
    fake_df['label'] = 'FAKE'
    real_df['label'] = 'REAL'
    
    # Combine and shuffle
    combined_df = pd.concat([fake_df, real_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Select relevant columns
    if 'title' in combined_df.columns and 'text' in combined_df.columns:
        combined_df = combined_df[['title', 'text', 'label']]
    
    # Save
    combined_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Dataset merged successfully!")
    print(f"   Location: {output_path}")
    print(f"   Total articles: {len(combined_df)}")
    print(f"   Columns: {combined_df.columns.tolist()}")
    
    return True


def main():
    """Main function."""
    print("="*60)
    print("FAKE NEWS DATASET DOWNLOADER")
    print("="*60)
    
    # Setup Kaggle
    if not setup_kaggle():
        print("Please setup Kaggle API credentials first.")
        return False
    
    # Download dataset
    if not download_dataset():
        print("Download failed. Check your internet connection.")
        return False
    
    # Merge datasets
    if not merge_datasets():
        print("Failed to merge datasets.")
        return False
    
    print("\nSUCCESS! Dataset is ready.")
    print("="*60)
    print("Next step: python model/train_model.py")
    print("This will train on the full dataset (40,000 articles)")
    print("Expected accuracy: 92-94%")
    print("Training time: 15-25 minutes")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
