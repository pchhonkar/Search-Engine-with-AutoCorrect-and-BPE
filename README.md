#  Search Engine with Auto-Correct and BPE

This project implements a **functional search engine** on a text corpus using **tokenization, subword modeling (BPE), edit distance-based auto-correction**, and **inverted indexing**.  
It retrieves relevant paragraphs for user queries, handles misspellings intelligently, and ranks results by similarity.

---

## 📂 Project Overview

The project is divided into **four main phases**:

### **Phase 1: Indexing the Corpus**
- **Corpus:** [Gutenberg Dataset](https://shibamoulilahiri.github.io/gutenberg_dataset.html)  
- Reads ~100 books (can scale up to 3000)
- Splits each book into **paragraphs** and assigns unique paragraph IDs
- Applies **Byte Pair Encoding (BPE)** for tokenization
- Builds an **inverted index** mapping each token to the paragraphs it appears in
- Maintains a **vocabulary** of unique tokens for use in auto-correction

### **Phase 2: User Interaction**
- Prompts the user for a **search query**
- Tokenizes the query and checks tokens against the vocabulary

### **Phase 3: Auto-Correct and OOV Handling**
- If a token is **out-of-vocabulary (OOV)**:
  - Calculates **Levenshtein Edit Distance** between the token and vocabulary words
  - Suggests corrections if distance ≤ threshold (default = 2)
  - If no correction is close enough, applies **BPE decomposition** to break the word into known subword units

### **Phase 4: Search and Ranking**
- Performs **vectorized similarity comparison** between query and paragraph tokens  
  (e.g., cosine similarity on TF-IDF representations)
- Displays **Top 10 matching paragraphs** with:
  - Paragraph ID  
  - Book title  
  - Short snippet preview
- Continues in a loop until user types `exit` or `quit`

---

## 🧠 Key Features

| Feature | Description |
|----------|-------------|
| **Tokenization** | Uses Byte Pair Encoding for subword segmentation |
| **Inverted Index** | Efficient paragraph-level search |
| **Auto-Correct** | Suggests close matches using edit distance |
| **OOV Handling** | Decomposes unknown words into known subwords |
| **Ranking** | Uses similarity metrics for top results |
| **Scalability** | Works efficiently on 100–3000 books |

ine-with-AutoCorrect-and-BPE.git
cd Search-Engine-with-AutoCorrect-and-BPE
