"""
Merge Fake.csv and True.csv into news.csv
Use this if you manually downloaded the dataset files.
"""

import os
import pandas as pd

def merge_datasets():
    """Merge Fake.csv and True.csv into news.csv"""
    
    dataset_dir = 'dataset'
    fake_path = os.path.join(dataset_dir, 'Fake.csv')
    true_path = os.path.join(dataset_dir, 'True.csv')
    output_path = os.path.join(dataset_dir, 'news.csv')
    
    # Check if files exist
    if not os.path.exists(fake_path):
        print(f"❌ Error: {fake_path} not found!")
        print(f"   Please download Fake.csv from Kaggle and place it in dataset/ folder")
        return False
    
    if not os.path.exists(true_path):
        print(f"❌ Error: {true_path} not found!")
        print(f"   Please download True.csv from Kaggle and place it in dataset/ folder")
        return False
    
    print("📂 Loading datasets...")
    print(f"   Reading: {fake_path}")
    print(f"   Reading: {true_path}")
    
    # Load datasets
    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(true_path)
    
    print(f"\n📊 Dataset sizes:")
    print(f"   Fake articles: {len(fake_df):,}")
    print(f"   Real articles: {len(real_df):,}")
    print(f"   Total: {len(fake_df) + len(real_df):,}")
    
    # Add label column
    print(f"\n🏷️  Adding labels...")
    fake_df['label'] = 'FAKE'
    real_df['label'] = 'REAL'
    
    # Combine datasets
    print(f"🔀 Merging and shuffling...")
    combined_df = pd.concat([fake_df, real_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Select relevant columns (title, text, label)
    if 'title' in combined_df.columns and 'text' in combined_df.columns:
        combined_df = combined_df[['title', 'text', 'label']]
        print(f"✓ Selected columns: {combined_df.columns.tolist()}")
    else:
        print(f"Available columns: {combined_df.columns.tolist()}")
    
    # Save merged dataset
    print(f"\n💾 Saving merged dataset...")
    combined_df.to_csv(output_path, index=False)
    
    # Verify
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\n✅ SUCCESS!")
    print(f"   File: {output_path}")
    print(f"   Size: {file_size_mb:.1f} MB")
    print(f"   Rows: {len(combined_df):,}")
    print(f"   Columns: {combined_df.columns.tolist()}")
    
    print(f"\n📈 Label distribution:")
    print(combined_df['label'].value_counts())
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("MERGE KAGGLE DATASET FILES")
    print("="*60)
    print()
    
    success = merge_datasets()
    
    if success:
        print("\n" + "="*60)
        print("Next step: Train the model")
        print("="*60)
        print("\nRun: python model/train_model.py")
        print("\nExpected results:")
        print("  • Accuracy: 92-94%")
        print("  • Training time: 15-25 minutes")
        print("  • Models saved in: model/")
    else:
        print("\n❌ Failed to merge datasets")
        print("Please ensure Fake.csv and True.csv are in dataset/ folder")
