"""
Text Tokenizer Module
Handles text vectorization and token ID conversion for malware detection
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import os


class MalwareTextTokenizer:
    """Text tokenizer that converts text to token IDs"""
    
    def __init__(self, max_tokens=10000, sequence_length=200):
        """
        Initialize tokenizer
        
        Args:
            max_tokens: Maximum vocabulary size
            sequence_length: Maximum sequence length for padding
        """
        self.max_tokens = max_tokens
        self.sequence_length = sequence_length
        self.vectorize_layer = None
        self.vocabulary = None
        self.token_to_id = {}
        self.id_to_token = {}
        
    def build_vocabulary(self, texts):
        """
        Build vocabulary from training texts
        
        Args:
            texts: List or array of text samples
        """
        print(f"\n🔤 Building vocabulary...")
        print(f"   - Max tokens: {self.max_tokens}")
        print(f"   - Sequence length: {self.sequence_length}")
        
        # Create TextVectorization layer
        self.vectorize_layer = keras.layers.TextVectorization(
            max_tokens=self.max_tokens,
            output_mode='int',
            output_sequence_length=self.sequence_length
        )
        
        # Adapt to training data
        self.vectorize_layer.adapt(texts)
        
        # Get vocabulary
        self.vocabulary = self.vectorize_layer.get_vocabulary()
        
        # Create token-ID mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocabulary)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocabulary)}
        
        print(f"✅ Vocabulary built: {len(self.vocabulary)} tokens")
        print(f"\n📋 Special Tokens:")
        print(f"   - ID 0: '{self.vocabulary[0]}' → [PAD] Padding token")
        print(f"   - ID 1: '{self.vocabulary[1]}' → [UNK] Unknown token")
        if len(self.vocabulary) > 2:
            print(f"   - ID 2: '{self.vocabulary[2]}' → Most frequent token")
            print(f"   - ID 3: '{self.vocabulary[3]}' → 2nd most frequent")
            print(f"   - ID 4: '{self.vocabulary[4]}' → 3rd most frequent")
        print(f"\n   First 15 tokens: {self.vocabulary[:15]}")
        
    def text_to_token_ids(self, texts):
        """
        Convert text to token IDs
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            numpy array of token IDs (2D for batch, 1D for single text)
        """
        if self.vectorize_layer is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
            
        # Convert to token IDs
        token_ids = self.vectorize_layer(texts).numpy()
        
        # Return single sequence if input was single text
        if single_input:
            return token_ids[0]
        return token_ids
    
    def token_ids_to_text(self, token_ids):
        """
        Convert token IDs back to text
        
        Args:
            token_ids: Array of token IDs (1D or 2D)
            
        Returns:
            String or list of strings
        """
        if self.id_to_token is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        # Handle single sequence
        if len(token_ids.shape) == 1:
            token_ids = np.expand_dims(token_ids, 0)
            single_input = True
        else:
            single_input = False
            
        # Convert IDs to tokens
        texts = []
        for sequence in token_ids:
            tokens = [self.id_to_token.get(int(idx), '[UNK]') for idx in sequence if idx != 0]
            text = ' '.join(tokens)
            texts.append(text)
        
        if single_input:
            return texts[0]
        return texts
    
    def get_token_id(self, token):
        """Get ID for a specific token"""
        return self.token_to_id.get(token, 1)  # 1 is typically [UNK]
    
    def get_token(self, token_id):
        """Get token for a specific ID"""
        return self.id_to_token.get(token_id, '[UNK]')
    
    def save(self, filepath):
        """
        Save tokenizer to file
        
        Args:
            filepath: Path to save tokenizer
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        config = {
            'max_tokens': self.max_tokens,
            'sequence_length': self.sequence_length,
            'vocabulary': self.vocabulary,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"✅ Tokenizer saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load tokenizer from file
        
        Args:
            filepath: Path to tokenizer file
            
        Returns:
            MalwareTextTokenizer instance
        """
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        tokenizer = cls(
            max_tokens=config['max_tokens'],
            sequence_length=config['sequence_length']
        )
        
        tokenizer.vocabulary = config['vocabulary']
        tokenizer.token_to_id = config['token_to_id']
        tokenizer.id_to_token = config['id_to_token']
        
        # Rebuild vectorize layer
        tokenizer.vectorize_layer = keras.layers.TextVectorization(
            max_tokens=tokenizer.max_tokens,
            output_mode='int',
            output_sequence_length=tokenizer.sequence_length
        )
        tokenizer.vectorize_layer.set_vocabulary(tokenizer.vocabulary)
        
        print(f"✅ Tokenizer loaded from: {filepath}")
        return tokenizer
    
    def analyze_text(self, text):
        """
        Analyze and display tokenization details for a text
        
        Args:
            text: Input text string
        """
        print("\n" + "="*60)
        print("📝 TEXT TOKENIZATION ANALYSIS")
        print("="*60)
        
        print(f"\n📄 Original Text:")
        print(f"   {text[:200]}...")
        
        # Get token IDs
        token_ids = self.text_to_token_ids(text)
        
        print(f"\n🔢 Token IDs:")
        print(f"   Length: {len(token_ids)}")
        print(f"   Non-zero tokens: {np.count_nonzero(token_ids)}")
        print(f"   Padding tokens (0): {np.sum(token_ids == 0)}")
        print(f"   Token IDs: {token_ids[:20]}...")
        
        # Get tokens with their IDs
        print(f"\n🔤 Token ID → Token Mapping:")
        non_zero_ids = token_ids[token_ids != 0][:20]  # First 20 non-padding tokens
        for tid in non_zero_ids:
            token = self.get_token(int(tid))
            token_type = ""
            if tid == 0:
                token_type = " [PAD]"
            elif tid == 1:
                token_type = " [UNK]"
            print(f"   ID {tid:3d} → '{token}'{token_type}")
        
        # Reconstruct text
        reconstructed = self.token_ids_to_text(token_ids)
        print(f"\n🔄 Reconstructed Text:")
        print(f"   {reconstructed[:200]}...")
        
        # Statistics
        print(f"\n📊 Statistics:")
        print(f"   - Unique token IDs: {len(np.unique(token_ids[token_ids != 0]))}")
        print(f"   - Unknown tokens ([UNK]): {np.sum(token_ids == 1)}")
        print(f"   - Known tokens: {np.sum((token_ids != 0) & (token_ids != 1))}")
        
        print("\n" + "="*60)
    
    def get_vocab_size(self):
        """Get actual vocabulary size"""
        if self.vocabulary is None:
            return 0
        return len(self.vocabulary)
    
    def __repr__(self):
        return f"MalwareTextTokenizer(max_tokens={self.max_tokens}, sequence_length={self.sequence_length}, vocab_size={self.get_vocab_size()})"


def create_tokenized_datasets(tokenizer, train_texts, train_labels, 
                               val_texts, val_labels, test_texts, test_labels,
                               batch_size=128):
    """
    Create TensorFlow datasets with tokenized data
    
    Args:
        tokenizer: MalwareTextTokenizer instance
        train_texts, val_texts, test_texts: Text arrays
        train_labels, val_labels, test_labels: Label arrays
        batch_size: Batch size for datasets
        
    Returns:
        train_ds, val_ds, test_ds: TensorFlow datasets
    """
    print("\n📦 Creating tokenized datasets...")
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    train_ds = train_ds.map(
        lambda text, label: (tokenizer.vectorize_layer(text), label),
        num_parallel_calls=AUTOTUNE
    )
    train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_texts, val_labels))
    val_ds = val_ds.map(
        lambda text, label: (tokenizer.vectorize_layer(text), label),
        num_parallel_calls=AUTOTUNE
    )
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((test_texts, test_labels))
    test_ds = test_ds.map(
        lambda text, label: (tokenizer.vectorize_layer(text), label),
        num_parallel_calls=AUTOTUNE
    )
    test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)
    
    print(f"✅ Datasets created with batch_size={batch_size}")
    
    return train_ds, val_ds, test_ds


# Example usage
if __name__ == "__main__":
    # Demo
    print("🔤 MalwareTextTokenizer Demo")
    
    # Sample texts
    sample_texts = [
        "<script>alert('XSS')</script>",
        "SELECT * FROM users WHERE id=1 OR 1=1",
        "Normal benign text sample",
    ]
    
    # Create and build tokenizer
    tokenizer = MalwareTextTokenizer(max_tokens=100, sequence_length=20)
    tokenizer.build_vocabulary(sample_texts)
    
    # Analyze a text
    tokenizer.analyze_text(sample_texts[0])
    
    # Convert text to token IDs
    token_ids = tokenizer.text_to_token_ids(sample_texts[0])
    print(f"\nToken IDs: {token_ids}")
    
    # Convert back to text
    reconstructed = tokenizer.token_ids_to_text(token_ids)
    print(f"Reconstructed: {reconstructed}")
    
    # Save tokenizer
    tokenizer.save("output/tokenizer.pkl")
    
    # Load tokenizer
    loaded_tokenizer = MalwareTextTokenizer.load("output/tokenizer.pkl")
    print(f"\nLoaded: {loaded_tokenizer}")
