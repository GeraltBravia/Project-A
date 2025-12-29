"""
Giải thích cách đánh dấu Token trong MalwareTextTokenizer
"""

print("="*70)
print("📚 CÁCH ĐÁNH DẤU TOKEN TRONG TOKENIZER")
print("="*70)

print("""
🔤 TensorFlow TextVectorization tự động tạo các SPECIAL TOKENS:

┌─────────────────────────────────────────────────────────┐
│  ID  │  Token  │  Ý nghĩa                               │
├─────────────────────────────────────────────────────────┤
│  0   │  ""     │  [PAD] - Padding token (để căn độ dài) │
│  1   │  [UNK]  │  Unknown token (từ không có trong vocab)│
│  2-N │  words  │  Các token thật theo tần suất xuất hiện │
└─────────────────────────────────────────────────────────┘

📊 VÍ DỤ TOKENIZATION:

Input text: "<script>alert('XSS')</script>"

1️⃣ BƯỚC 1: Tách thành tokens (words)
   ['<script>', 'alert', 'xss', '</script>']

2️⃣ BƯỚC 2: Convert tokens → IDs (theo vocabulary)
   [45, 123, 789, 46]
   
   Trong đó:
   - '<script>' → ID 45
   - 'alert'    → ID 123
   - 'xss'      → ID 789
   - '</script>'→ ID 46

3️⃣ BƯỚC 3: Padding (nếu sequence_length = 200)
   [45, 123, 789, 46, 0, 0, 0, ..., 0]
   
   Thêm token ID=0 ([PAD]) để đủ 200 elements

4️⃣ BƯỚC 4: Xử lý Unknown tokens
   Nếu gặp token chưa có trong vocab:
   'unknown_word' → ID 1 ([UNK])

📋 CÁC CÁCH ĐÁNH DẤU TOKEN:

1. **By Position (ID)**:
   - ID càng nhỏ → token càng phổ biến
   - ID 0: Padding
   - ID 1: Unknown
   - ID 2+: Theo frequency (cao → thấp)

2. **By Type**:
   - Normal tokens: ID >= 2
   - Special tokens: ID 0, 1

3. **By Content**:
   Vocabulary được sắp xếp theo tần suất:
   ['', '[UNK]', 'script', 'select', 'alert', 'from', ...]
    ↑    ↑        ↑         ↑         ↑        ↑
    0    1        2         3         4        5

🎯 TRONG CODE:

tokenizer.token_to_id = {
    '': 0,           # Padding token
    '[UNK]': 1,      # Unknown token
    'script': 2,     # Most frequent word
    'select': 3,     # 2nd most frequent
    'alert': 4,      # 3rd most frequent
    ...
}

tokenizer.id_to_token = {
    0: '',           # ID 0 → Padding
    1: '[UNK]',      # ID 1 → Unknown
    2: 'script',     # ID 2 → Most frequent
    3: 'select',     # ID 3 → 2nd frequent
    4: 'alert',      # ID 4 → 3rd frequent
    ...
}

💡 THAM KHẢO:

Sequence: [45, 123, 0, 0, 0]
          ↓    ↓    ↓  ↓  ↓
Token:    word1 word2 PAD PAD PAD

Trong model:
- Token ID = 0 → Không học (masked/ignored)
- Token ID = 1 → Học như token đặc biệt
- Token ID >= 2 → Học bình thường

🔍 XEM THÊM:
- Chạy: python demo_tokenizer.py
- Xem function: tokenizer.analyze_text()
""")

print("="*70)
print("✅ CÁCH KIỂM TRA:")
print("="*70)

print("""
1. Build vocabulary:
   tokenizer.build_vocabulary(texts)
   → Tự động hiển thị special tokens

2. Analyze text:
   tokenizer.analyze_text(text)
   → Hiển thị chi tiết ID → Token mapping

3. Get token info:
   tokenizer.get_token(0)   → ''  (Padding)
   tokenizer.get_token(1)   → '[UNK]' (Unknown)
   tokenizer.get_token_id('script') → 2
""")

print("="*70)
