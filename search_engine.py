#!/usr/bin/env python3
"""
Phase 1 + Query:
- Sampling, paragraphization, BPE tokenizer, inverted index, IDF, paragraph norms
- Interactive search with auto-correct:
    * Word-level autocorrect (Levenshtein) BEFORE BPE
    * BPE-level subword fallback
"""

import os
import re
import sys
import math
import random
import zipfile
import pickle
from pathlib import Path
from typing import List
from collections import defaultdict, Counter

from tqdm import tqdm
from rich import print as rprint
from rich.panel import Panel

# Hugging Face tokenizers for BPE
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# Utilities / filesystem

def ensure_dir(p: str | Path) -> None:
    p = Path(p)
    if p.exists() and p.is_file():
        raise RuntimeError(f"'{p}' exists as a file. Please remove or rename it so a directory can be created.")
    p.mkdir(parents=True, exist_ok=True)

# ZIP reading + cleaning

def list_txt_members(zip_path: str) -> List[str]:
    """Return .txt members in the zip (case-insensitive)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
    return names


def read_text_from_zip(zip_path: str, member: str) -> str:
    """Read a text file member from the zip with robust decoding."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        raw = zf.read(member)
    # try utf-8, then latin-1 as fallback (ignore undecodable chars)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Optional: remove Gutenberg headers/footers if markers exist.
    If not found, returns original text.
    """
    start_pat = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\n",
                           re.IGNORECASE | re.DOTALL)
    end_pat = re.compile(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*",
                         re.IGNORECASE | re.DOTALL)

    start_match = start_pat.search(text)
    end_match = end_pat.search(text)

    if start_match and end_match:
        return text[start_match.end(): end_match.start()]
    return text


def split_paragraphs(text: str) -> List[str]:
    """Split on blank lines; collapse excess whitespace; drop tiny paras."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = re.split(r"\n\s*\n+", text)
    cleaned = []
    for p in paras:
        p2 = re.sub(r"[ \t]+", " ", p.strip())
        if len(p2) >= 25:
            cleaned.append(p2)
    return cleaned

# Sampling

def sample_books(all_members: List[str], k: int, seed: int = 42) -> List[str]:
    random.seed(seed)
    if len(all_members) <= k:
        return all_members
    return random.sample(all_members, k)


def preview_sample(zip_path: str, sampled: List[str], max_preview: int = 1) -> None:
    """Print a quick preview: book name, paragraph count, and first 2 paragraphs."""
    for i, member in enumerate(sampled[:max_preview]):
        txt = read_text_from_zip(zip_path, member)
        txt = strip_gutenberg_boilerplate(txt)
        paras = split_paragraphs(txt)

        rprint(Panel.fit(f"[bold]Sample #{i}[/bold]\nFile: {member}\nParagraphs: {len(paras)}"))
        if paras:
            rprint("[cyan]First paragraph:[/cyan]")
            rprint(paras[0][:800] + ("..." if len(paras[0]) > 800 else ""))
        if len(paras) > 1:
            rprint("\n[cyan]Second paragraph:[/cyan]")
            rprint(paras[1][:800] + ("..." if len(paras[1]) > 800 else ""))


def save_sample_list(sampled: List[str], out_dir: str) -> None:
    ensure_dir(out_dir)
    out_path = Path(out_dir) / "sampled_books.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for name in sampled:
            f.write(name + "\n")


def load_sample_list(index_dir: str) -> list[str]:
    path = Path(index_dir) / "sampled_books.txt"
    if not path.exists():
        raise RuntimeError(f"sampled_books.txt not found at {path}. Run the sampling step first.")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def stream_sample_texts(zip_path: str, sampled_members: list[str]) -> list[str]:
    """Read all sampled files, strip boilerplate, return list of texts."""
    texts = []
    for member in tqdm(sampled_members, desc="Reading sampled books"):
        txt = read_text_from_zip(zip_path, member)
        txt = strip_gutenberg_boilerplate(txt)
        texts.append(txt)
    return texts

# BPE training + usage

def train_bpe_tokenizer(texts: list[str], vocab_size: int = 8000, min_freq: int = 2) -> Tokenizer:
    """Train a simple whitespace-pretokenized BPE on provided texts."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_freq, special_tokens=["[UNK]"])
    tokenizer.post_processor = TemplateProcessing(single="$A", pair="$A $B", special_tokens=[])
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, out_dir: str):
    ensure_dir(out_dir)
    (Path(out_dir) / "tokenizer.json").write_text(tokenizer.to_str(), encoding="utf-8")


def load_tokenizer(index_dir: str) -> Tokenizer:
    tok_path = Path(index_dir) / "tokenizer.json"
    if not tok_path.exists():
        raise RuntimeError(f"tokenizer.json not found at {tok_path}. Train it first.")
    tok = Tokenizer.from_file(str(tok_path))
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = TemplateProcessing(single="$A", pair="$A $B", special_tokens=[])
    return tok

# Paragraphization + tokenization + persistence

def paragraphs_from_text(text: str) -> list[str]:
    return split_paragraphs(text)


def tokenize_paragraphs(tokenizer: Tokenizer, paras: list[str]) -> list[list[str]]:
    tokenized = []
    for p in paras:
        enc = tokenizer.encode(p)
        tokenized.append(enc.tokens)
    return tokenized


def make_paragraphs_and_bpe(zip_path: str, index_dir: str, vocab_size: int = 8000, min_freq: int = 2):
    """
    Build artifacts:
        index/tokenizer.json
        index/para_text.pkl
        index/para_meta.pkl
        index/tokens_by_para.pkl
        index/vocab.pkl
        index/word_vocab.pkl   <-- word-level vocabulary for autocorrect
    """
    sampled = load_sample_list(index_dir)
    texts = stream_sample_texts(zip_path, sampled)

    # Train and save tokenizer
    rprint(Panel.fit(f"Training BPE (vocab_size={vocab_size}, min_freq={min_freq})"))
    tokenizer = train_bpe_tokenizer(texts, vocab_size=vocab_size, min_freq=min_freq)
    save_tokenizer(tokenizer, index_dir)
    rprint("[green]Saved tokenizer to index/tokenizer.json[/green]")

    para_text: dict[str, str] = {}
    para_meta: dict[str, dict] = {}
    tokens_by_para: dict[str, list[str]] = {}

    total_paras = 0
    for bidx, member in enumerate(tqdm(sampled, desc="Processing books → paragraphs + BPE")):
        raw = read_text_from_zip(zip_path, member)
        raw = strip_gutenberg_boilerplate(raw)
        paras = paragraphs_from_text(raw)
        tok_paras = tokenize_paragraphs(tokenizer, paras)

        for pidx, (p_text, p_toks) in enumerate(zip(paras, tok_paras)):
            pid = f"{bidx}:{pidx}"
            para_text[pid] = p_text
            para_meta[pid] = {"book_id": bidx, "book_title": member}
            tokens_by_para[pid] = p_toks
            total_paras += 1

    rprint(f"Total paragraphs processed: [bold]{total_paras}[/bold]")

    # Persist artifacts
    ensure_dir(index_dir)
    with open(Path(index_dir) / "para_text.pkl", "wb") as f:
        pickle.dump(para_text, f)
    with open(Path(index_dir) / "para_meta.pkl", "wb") as f:
        pickle.dump(para_meta, f)
    with open(Path(index_dir) / "tokens_by_para.pkl", "wb") as f:
        pickle.dump(tokens_by_para, f)

    # Explicit BPE vocab for auto-correct at subword level
    vocab = set()
    for toks in tokens_by_para.values():
        vocab.update(toks)
    with open(Path(index_dir) / "vocab.pkl", "wb") as f:
        pickle.dump(sorted(vocab), f)

    # Word-level vocabulary from raw paragraphs (include singletons)
    word_counts = Counter()
    word_re = re.compile(r"[A-Za-z]+")
    for p in para_text.values():
        for w in word_re.findall(p.lower()):
            if len(w) >= 3:
                word_counts[w] += 1
    word_vocab = [w for w, c in word_counts.items() if c >= 1]
    word_vocab.sort()
    with open(Path(index_dir) / "word_vocab.pkl", "wb") as f:
        pickle.dump(word_vocab, f)

    rprint("[bold green]Saved:[/bold green] para_text.pkl, para_meta.pkl, tokens_by_para.pkl, vocab.pkl, word_vocab.pkl")


# Inverted index + IDF + paragraph norms

def build_inverted_index_and_stats(index_dir: str):
    """
    Builds:
      postings: dict[token] -> dict[para_id] -> first_position (int)
      idf: dict[token] -> float (smooth IDF)
      para_norm: dict[para_id] -> float (L2 norm of TF-IDF vector with log-tf)

    Reads:
      tokens_by_para.pkl

    Writes:
      postings.pkl, idf.pkl, para_norm.pkl
    """
    d = Path(index_dir)
    tbp_path = d / "tokens_by_para.pkl"
    if not tbp_path.exists():
        raise RuntimeError(f"{tbp_path} not found. Run --make_paras_and_bpe first.")

    rprint(Panel.fit("Loading tokenized paragraphs (tokens_by_para.pkl)"))
    with open(tbp_path, "rb") as f:
        tokens_by_para: dict[str, list[str]] = pickle.load(f)

    N_docs = len(tokens_by_para)
    rprint(f"Total paragraphs to index: [bold]{N_docs}[/bold]")

    postings: dict[str, dict[str, int]] = defaultdict(dict)   # token -> {para_id: first_pos}
    tf: dict[str, dict[str, int]] = {}

    for pid, toks in tqdm(tokens_by_para.items(), desc="Building TF & postings"):
        counts = Counter(toks)
        tf[pid] = dict(counts)

        seen = set()
        for pos, tok in enumerate(toks):
            if tok not in seen:
                postings[tok][pid] = pos
                seen.add(tok)

    rprint("Computing DF & IDF …")
    df = {tok: len(pmap) for tok, pmap in postings.items()}
    idf = {tok: math.log(1.0 + N_docs / (1.0 + df_val)) for tok, df_val in df.items()}

    rprint("Computing paragraph norms …")
    para_norm: dict[str, float] = {}
    for pid, counts in tqdm(tf.items(), desc="Norms"):
        s = 0.0
        for tok, c in counts.items():
            tf_w = 1.0 + math.log(c)
            s += (tf_w * idf.get(tok, 0.0)) ** 2
        para_norm[pid] = math.sqrt(s) if s > 0 else 1.0

    rprint("Saving postings, idf, para_norm …")
    with open(d / "postings.pkl", "wb") as f:
        pickle.dump(postings, f)
    with open(d / "idf.pkl", "wb") as f:
        pickle.dump(idf, f)
    with open(d / "para_norm.pkl", "wb") as f:
        pickle.dump(para_norm, f)

    rprint("[bold green]Phase 1 indexing complete:[/bold green] postings.pkl, idf.pkl, para_norm.pkl saved.")


# Auto-correct helpers

def levenshtein(a: str, b: str, cutoff: int = 2) -> int:
    """Two-row DP Levenshtein with early cutoff."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > cutoff:
        return cutoff + 1
    if la == 0:
        return lb if lb <= cutoff else cutoff + 1
    if lb == 0:
        return la if la <= cutoff else cutoff + 1

    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        min_row = cur[0]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            del_ = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            v = min(ins, del_, sub)
            cur.append(v)
            if v < min_row:
                min_row = v
        if min_row > cutoff:
            return cutoff + 1
        prev = cur
    return prev[-1]


def build_vocab_indexes(vocab: List[str]):
    by_len = defaultdict(list)
    by_first = defaultdict(list)
    for w in vocab:
        if not w:
            continue
        by_len[len(w)].append(w)
        by_first[w[0]].append(w)
    return by_len, by_first


def prune_candidates(token: str, by_len, by_first, max_len_diff: int = 2) -> List[str]:
    L = len(token)
    pool = set()
    for l in range(max(1, L - max_len_diff), L + max_len_diff + 1):
        for w in by_len.get(l, []):
            pool.add(w)
    if token:
        first = token[0]
        pool = [w for w in pool if w[0] == first]
    else:
        pool = list(pool)
    return list(pool)[:5000]


def autocorrect_token(token: str, vocab_set: set, by_len, by_first, cutoff: int = 2):
    """
    Improved tie-breaking:
      - sort candidates for determinism
      - prefer LONGER candidate on equal distance
      - then alphabetical for stability
    """
    if token in vocab_set or not token or token == "[UNK]":
        return None
    cands = sorted(prune_candidates(token, by_len, by_first, max_len_diff=cutoff))
    best = None
    best_d = cutoff + 1
    for w in cands:
        d = levenshtein(token, w, cutoff=cutoff)
        if d < best_d:
            best_d, best = d, w
        elif d == best_d and best is not None:
            if len(w) > len(best):
                best = w
            elif len(w) == len(best) and w < best:
                best = w
    return best if best_d <= cutoff else None


def longest_known_subwords(token: str, vocab_set: set) -> List[str]:
    """Greedy longest-match segmentation of a string into known tokens."""
    if not token:
        return []
    n = len(token)
    parts = []
    i = 0
    while i < n:
        match = None
        for j in range(n, i, -1):
            sub = token[i:j]
            if sub in vocab_set:
                match = sub
                break
        if match:
            parts.append(match)
            i += len(match)
        else:
            if token[i] in vocab_set:
                parts.append(token[i])
            i += 1
    return parts or [token]


def autocorrect_plain_words_in_query(query: str, word_vocab_list: List[str], cutoff: int = 2):
    """Autocorrect plain alphabetic words BEFORE BPE tokenization."""
    word_vocab_set = set(word_vocab_list)
    by_len, by_first = build_vocab_indexes(word_vocab_list)

    # find alphabetic words
    spans = []
    for m in re.finditer(r"[A-Za-z]+", query):
        spans.append((m.start(), m.end(), m.group(0)))

    corrections = {}
    for _, _, w in spans:
        lw = w.lower()
        if lw not in word_vocab_set and len(lw) >= 3:
            cand = autocorrect_token(lw, word_vocab_set, by_len, by_first, cutoff=cutoff)
            if cand:
                corrections[w] = cand  # store lowercase correction

    if not corrections:
        return query, {}

    # replace while preserving first-letter capitalization
    def repl(m):
        orig = m.group(0)
        lw = orig.lower()
        corr = corrections.get(orig) or corrections.get(lw)
        if not corr:
            return orig
        return corr.capitalize() if orig[0].isupper() else corr

    new_q = re.sub(r"[A-Za-z]+", repl, query)
    return new_q, corrections


# Search helpers

def make_snippet(text: str, terms: List[str], radius: int = 80) -> str:
    low = text.lower()
    hit = None
    for t in terms:
        if not t or t == "[UNK]":
            continue
        pos = low.find(t.lower())
        if pos != -1:
            if hit is None or pos < hit:
                hit = pos
    if hit is None:
        start = 0
    else:
        start = max(0, hit - radius)
    end = min(len(text), start + 2 * radius + 60)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


def interactive_search(index_dir: str,
                       edit_cutoff: int = 2,
                       allow_subword_fallback: bool = True,
                       allow_word_autocorrect: bool = True):
    d = Path(index_dir)

    # Load artifacts
    tokenizer = load_tokenizer(index_dir)
    with open(d / "vocab.pkl", "rb") as f:
        vocab_list = pickle.load(f)
    vocab_set = set(vocab_list)
    by_len, by_first = build_vocab_indexes(vocab_list)

    with open(d / "postings.pkl", "rb") as f:
        postings: dict[str, dict[str, int]] = pickle.load(f)
    with open(d / "idf.pkl", "rb") as f:
        idf: dict[str, float] = pickle.load(f)
    with open(d / "para_norm.pkl", "rb") as f:
        para_norm: dict[str, float] = pickle.load(f)
    with open(d / "para_text.pkl", "rb") as f:
        para_text: dict[str, str] = pickle.load(f)
    with open(d / "para_meta.pkl", "rb") as f:
        para_meta: dict[str, dict] = pickle.load(f)

    # word-level vocabulary for pre-BPE autocorrect
    with open(d / "word_vocab.pkl", "rb") as f:
        word_vocab_list: List[str] = pickle.load(f)

    rprint(Panel.fit("[bold]Search ready.[/bold] Type your query. Type [cyan]exit[/cyan] or [cyan]quit[/cyan] to stop."))

    while True:
        try:
            q = input("Enter search query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        #  word-level autocorrect BEFORE BPE 
        if allow_word_autocorrect:
            q_corrected, plain_corrections = autocorrect_plain_words_in_query(q, word_vocab_list, cutoff=edit_cutoff)
        else:
            q_corrected, plain_corrections = (q, {})

        if plain_corrections:
            for orig, corr in plain_corrections.items():
                rprint(f"Did you mean: [bold]{corr}[/bold] (for '{orig}')?")

        # Tokenize corrected query via BPE
        q_tokens = tokenizer.encode(q_corrected).tokens

        # For snippets, also keep rough plain terms (split on words)
        plain_terms = re.findall(r"[A-Za-z0-9]+", q_corrected.lower())

        final_tokens: List[str] = []
        messages: List[str] = []
        for t in q_tokens:
            if t == "[UNK]":
                continue  # skip unknown marker tokens
            if not re.search(r"[A-Za-z0-9]", t):  # skip pure punctuation-like tokens
                continue

            if t in vocab_set:
                final_tokens.append(t)
            else:
                # BPE-level correction only for alphabetic-ish pieces
                corr = None
                if re.search(r"[A-Za-z]", t) and len(t) >= 3:
                    corr = autocorrect_token(t, vocab_set, by_len, by_first, cutoff=edit_cutoff)

                if corr:
                    messages.append(f"Did you mean: [bold]{corr}[/bold] (for '{t}')?")
                    final_tokens.append(corr)
                else:
                    if allow_subword_fallback:
                        parts = longest_known_subwords(t, vocab_set)
                        if parts != [t]:
                            messages.append(f"Using subwords for '{t}': " + " ".join(parts))
                        final_tokens.extend(parts)
                    # else: drop token

        if messages:
            for m in messages:
                rprint(m)

        q_counts = Counter([t for t in final_tokens if t in idf])
        if not q_counts:
            rprint("[yellow]No known tokens in query after processing.[/yellow]")
            continue

        q_weights = {t: (1.0 + math.log(c)) * idf[t] for t, c in q_counts.items()}
        q_norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0

        # Candidate paragraphs: union of postings per token
        candidate_pids = set()
        for t in q_weights:
            candidate_pids.update(postings.get(t, {}).keys())

        if not candidate_pids:
            rprint("[red]No paragraph found for this query.[/red]")
            continue

        # Scoring: IDF-weighted presence normalized by paragraph norm
        scores = []
        for pid in candidate_pids:
            dot = 0.0
            for t, q_w in q_weights.items():
                if pid in postings.get(t, ()):
                    dot += q_w * idf[t]
            denom = q_norm * (para_norm.get(pid, 1.0) or 1.0)
            s = dot / denom if denom > 0 else 0.0
            if s > 0:
                scores.append((pid, s))

        if not scores:
            rprint("[red]No paragraph found for this query.[/red]")
            continue

        scores.sort(key=lambda x: x[1], reverse=True)
        topk = scores[:10]

        # Print results
        rprint(Panel.fit(f"[bold]Top {len(topk)} results[/bold]"))
        for rank, (pid, sc) in enumerate(topk, 1):
            meta = para_meta[pid]
            text = para_text[pid]
            snippet = make_snippet(text, plain_terms or final_tokens, radius=80)
            rprint(f"[bold][{rank}][/bold] Score: {sc:.3f}")
            rprint(f"Book: {meta['book_title']}  |  Paragraph ID: {pid}")
            rprint(snippet)
            rprint("-" * 80)



# CLI

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1: Build artifacts; Phase 2: interactive search.")
    parser.add_argument("--zip", required=True, help="Path to Gutenberg_original.zip")
    parser.add_argument("--sample", type=int, default=100, help="Number of books to sample (default: 100)")
    parser.add_argument("--index_dir", default="index", help="Directory to store artifacts (default: index)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--preview", type=int, default=1, help="How many sampled books to preview (default: 1)")

    # Build artifacts
    parser.add_argument("--make_paras_and_bpe", action="store_true",
                        help="Paragraphize sampled books, train BPE, tokenize, and persist artifacts.")
    parser.add_argument("--vocab_size", type=int, default=8000, help="BPE vocab size (default: 8000)")
    parser.add_argument("--min_freq", type=int, default=2, help="BPE min token frequency (default: 2)")
    parser.add_argument("--build_index", action="store_true",
                        help="Build inverted index + IDF + paragraph norms from tokens_by_para.pkl")

    # Query mode + experiment switches
    parser.add_argument("--query", action="store_true", help="Enter interactive search loop")
    parser.add_argument("--edit_cutoff", type=int, default=2, help="Levenshtein cutoff for autocorrect (default: 2)")
    parser.add_argument("--no_subword_fallback", action="store_true", help="Disable BPE subword fallback for OOV")
    parser.add_argument("--no_word_autocorrect", action="store_true", help="Disable pre-BPE word autocorrect")

    args = parser.parse_args()

    zip_path = args.zip
    if not Path(zip_path).exists():
        rprint(f"[red]ZIP not found:[/red] {zip_path}")
        sys.exit(1)

    rprint(Panel.fit(f"Opening ZIP: [bold]{zip_path}[/bold]"))

    all_txt = list_txt_members(zip_path)
    if not all_txt:
        rprint("[red]No .txt files found in the ZIP.[/red]")
        sys.exit(1)

    rprint(f"Total .txt files in zip: [bold]{len(all_txt)}[/bold]")

    sampled = sample_books(all_txt, k=args.sample, seed=args.seed)
    rprint(f"Sampled [bold]{len(sampled)}[/bold] books (seed={args.seed}).")

    # Save the sample list for reproducibility
    save_sample_list(sampled, args.index_dir)
    rprint(f"Saved sampled list to [green]{Path(args.index_dir) / 'sampled_books.txt'}[/green]")

    # Preview the first sampled book(s)
    preview_sample(zip_path, sampled, max_preview=args.preview)
    rprint("[bold green]Phase 1 (sampling) complete:[/bold green] ZIP loaded, books listed and sampled, preview printed.")

    if args.make_paras_and_bpe:
        make_paragraphs_and_bpe(zip_path=args.zip,
                                index_dir=args.index_dir,
                                vocab_size=args.vocab_size,
                                min_freq=args.min_freq)
        rprint("[bold green]Paragraphization + BPE step complete.[/bold green]")

    if args.build_index:
        build_inverted_index_and_stats(index_dir=args.index_dir)

    if args.query:
        interactive_search(index_dir=args.index_dir,
                           edit_cutoff=args.edit_cutoff,
                           allow_subword_fallback=not args.no_subword_fallback,
                           allow_word_autocorrect=not args.no_word_autocorrect)


if __name__ == "__main__":
    main()
