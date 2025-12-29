"""
Demo script for MalwareTextTokenizer
Shows how to use the tokenizer module
"""

from tokenizer import MalwareTextTokenizer
import numpy as np

def main():
    print("="*70)
    print("🔤 MALWARE TEXT TOKENIZER - DEMO")
    print("="*70)
    
    # Sample malware and benign texts
    sample_texts = [
        "<script>alert('XSS Attack!')</script>",
        "SELECT * FROM users WHERE id=1 OR 1=1",
        "' OR '1'='1' --",
        "<img src=x onerror=alert(1)>",
        "This is a normal benign text sample",
        "Hello world, how are you today?",
    ]
    
    labels = [1, 1, 1, 1, 0, 0]  # 1=malware, 0=benign
    
    # 1. Create and build tokenizer
    print("\n" + "="*70)
    print("1️⃣ BUILDING TOKENIZER")
    print("="*70)
    
    tokenizer = MalwareTextTokenizer(max_tokens=1000, sequence_length=50)
    tokenizer.build_vocabulary(sample_texts)
    
    print(f"\n📊 Tokenizer: {tokenizer}")
    print(f"   Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"   First 20 tokens: {tokenizer.vocabulary[:20]}")
    
    # 2. Convert text to token IDs
    print("\n" + "="*70)
    print("2️⃣ TEXT TO TOKEN IDS")
    print("="*70)
    
    test_text = sample_texts[0]
    print(f"\n📝 Original Text:")
    print(f"   \"{test_text}\"")
    
    token_ids = tokenizer.text_to_token_ids(test_text)
    print(f"\n🔢 Token IDs:")
    print(f"   {token_ids}")
    print(f"   Shape: {token_ids.shape}")
    print(f"   Non-zero tokens: {np.count_nonzero(token_ids)}")
    
    # 3. Convert token IDs back to text
    print("\n" + "="*70)
    print("3️⃣ TOKEN IDS TO TEXT")
    print("="*70)
    
    reconstructed = tokenizer.token_ids_to_text(token_ids)
    print(f"\n🔄 Reconstructed Text:")
    print(f"   \"{reconstructed}\"")
    
    # 4. Batch conversion
    print("\n" + "="*70)
    print("4️⃣ BATCH CONVERSION")
    print("="*70)
    
    batch_texts = sample_texts[:3]
    print(f"\n📝 Original Texts ({len(batch_texts)} samples):")
    for i, text in enumerate(batch_texts, 1):
        print(f"   {i}. \"{text}\"")
    
    batch_token_ids = tokenizer.text_to_token_ids(batch_texts)
    print(f"\n🔢 Batch Token IDs:")
    print(f"   Shape: {batch_token_ids.shape}")
    print(f"   Sample: {batch_token_ids[0][:20]}...")
    
    # 5. Token lookup
    print("\n" + "="*70)
    print("5️⃣ TOKEN LOOKUP")
    print("="*70)
    
    test_tokens = ["script", "alert", "select", "unknown_token_xyz"]
    print(f"\n🔍 Token to ID Mapping:")
    for token in test_tokens:
        token_id = tokenizer.get_token_id(token)
        print(f"   '{token}' -> ID: {token_id}")
    
    test_ids = [2, 5, 10, 15, 100, 999]
    print(f"\n🔍 ID to Token Mapping:")
    for tid in test_ids:
        token = tokenizer.get_token(tid)
        print(f"   ID {tid} -> '{token}'")
    
    # 6. Detailed text analysis
    print("\n" + "="*70)
    print("6️⃣ DETAILED TEXT ANALYSIS")
    print("="*70)
    
    tokenizer.analyze_text(sample_texts[1])
    
    # 7. Save and load tokenizer
    print("\n" + "="*70)
    print("7️⃣ SAVE & LOAD TOKENIZER")
    print("="*70)
    
    save_path = "output/demo_tokenizer.pkl"
    tokenizer.save(save_path)
    
    print("\n📥 Loading tokenizer...")
    loaded_tokenizer = MalwareTextTokenizer.load(save_path)
    print(f"   Loaded: {loaded_tokenizer}")
    
    # Verify loaded tokenizer works
    test_ids = loaded_tokenizer.text_to_token_ids(sample_texts[0])
    print(f"\n✅ Verification:")
    print(f"   Original token IDs match: {np.array_equal(token_ids, test_ids)}")
    
    # 8. Statistics
    print("\n" + "="*70)
    print("8️⃣ TOKENIZATION STATISTICS")
    print("="*70)
    
    all_token_ids = tokenizer.text_to_token_ids(sample_texts)
    
    print(f"\n📊 Statistics for {len(sample_texts)} texts:")
    print(f"   Total sequences: {len(all_token_ids)}")
    print(f"   Max sequence length: {tokenizer.sequence_length}")
    print(f"   Avg non-zero tokens: {np.count_nonzero(all_token_ids, axis=1).mean():.1f}")
    print(f"   Min non-zero tokens: {np.count_nonzero(all_token_ids, axis=1).min()}")
    print(f"   Max non-zero tokens: {np.count_nonzero(all_token_ids, axis=1).max()}")
    
    # 9. Malware vs Benign comparison
    print("\n" + "="*70)
    print("9️⃣ MALWARE VS BENIGN COMPARISON")
    print("="*70)
    
    malware_ids = all_token_ids[:4]  # First 4 are malware
    benign_ids = all_token_ids[4:]   # Last 2 are benign
    
    print(f"\n🦠 Malware samples:")
    print(f"   Avg tokens: {np.count_nonzero(malware_ids, axis=1).mean():.1f}")
    print(f"   Sample token IDs: {malware_ids[0][:15]}")
    
    print(f"\n✅ Benign samples:")
    print(f"   Avg tokens: {np.count_nonzero(benign_ids, axis=1).mean():.1f}")
    print(f"   Sample token IDs: {benign_ids[0][:15]}")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main()
