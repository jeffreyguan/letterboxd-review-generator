import unicodedata
import regex as re

def get_counts(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def _merge_chunk_tracked(chunk_ids, pair, idx, counts, where, chunk_idx):
    """Same substitution as merge(), but also updates counts/where in place
    for only the pairs that are actually created/destroyed by this merge,
    instead of requiring a full corpus rescan to know what changed."""
    newids = []
    i = 0
    n = len(chunk_ids)
    while i < n:
        if i < n - 1 and chunk_ids[i] == pair[0] and chunk_ids[i + 1] == pair[1]:
            if newids:
                left = newids[-1]
                old_left = (left, pair[0])
                if old_left != pair:
                    counts[old_left] -= 1
                    if counts[old_left] <= 0:
                        del counts[old_left]
                        where[old_left].discard(chunk_idx)
                        if not where[old_left]:
                            del where[old_left]
                new_left = (left, idx)
                counts[new_left] = counts.get(new_left, 0) + 1
                where.setdefault(new_left, set()).add(chunk_idx)
            if i + 2 < n:
                right = chunk_ids[i + 2]
                old_right = (pair[1], right)
                if old_right != pair:
                    counts[old_right] -= 1
                    if counts[old_right] <= 0:
                        del counts[old_right]
                        where[old_right].discard(chunk_idx)
                        if not where[old_right]:
                            del where[old_right]
                new_right = (idx, right)
                counts[new_right] = counts.get(new_right, 0) + 1
                where.setdefault(new_right, set()).add(chunk_idx)
            newids.append(idx)
            i += 2
        else:
            newids.append(chunk_ids[i])
            i += 1
    return newids

# helper functions copied (see line 128)
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""" #gpt4
        self.special_token_strings = [
            "<|rating0.5|>",
            "<|rating1.0|>",
            "<|rating1.5|>",
            "<|rating2.0|>",
            "<|rating2.5|>",
            "<|rating3.0|>",
            "<|rating3.5|>",
            "<|rating4.0|>",
            "<|rating4.5|>",
            "<|rating5.0|>",
            "<|startoftitle|>",
            "<|endoftitle|>",
            "<|endoftext|>"
        ]
        self.special_tokens = {}
        self.vocab = self._build_vocab()
    
    def train(self, text, vocab_size):
        num_merges = vocab_size - len(self.vocab)
        chunks = re.findall(self.pattern, text)
        ids = [list(chunk.encode("utf-8")) for chunk in chunks]

        counts = {}
        where = {}
        for chunk_idx, chunk_ids in enumerate(ids):
            for pair in zip(chunk_ids, chunk_ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
                where.setdefault(pair, set()).add(chunk_idx)

        for i in range(num_merges):
            if not counts:
                break
            pair = max(counts, key=lambda x: (counts[x], x))
            idx = len(self.vocab)

            for chunk_idx in list(where.get(pair, ())):
                ids[chunk_idx] = _merge_chunk_tracked(ids[chunk_idx], pair, idx, counts, where, chunk_idx)

            del counts[pair]
            if pair in where:
                del where[pair]

            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        for token in self.special_token_strings:
            idx = len(self.vocab)
            self.special_tokens[token] = idx
            self.vocab[idx] = token.encode("utf-8")

        return ids      

    def _encode_chunk(self, text):
        ids = list(text)
        while len(ids) > 1:
            pairs = get_counts(ids)
            pair = min(pairs, key=lambda x: self.merges.get(x, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids
    
    def _encode_non_special(self, text):
        chunks = re.findall(self.pattern, text)
        ids = []
        for chunk in chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text):
        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self._encode_non_special(part))
        return ids

    def decode(self, ids):
        text = bytes()
        for id in ids:
            text += self.vocab[id]
        return text.decode("utf-8", errors="replace")

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    # save and load functions copied from karpathy's minibpe repo
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
        