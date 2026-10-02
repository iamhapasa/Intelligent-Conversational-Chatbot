import re
import math
import random
from dataclasses import dataclass
from collections import Counter, deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


@dataclass
class CFG:
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    max_pairs = 25000
    max_vocab = 30000

    src_max_len = 64
    tgt_max_len = 60
    history_turns = 2

    d_model = 256
    nhead = 8
    num_encoder_layers = 4
    num_decoder_layers = 4
    dim_feedforward = 1024
    dropout = 0.1

    batch_size = 32
    epochs = 8
    lr = 2e-4
    val_ratio = 0.1

    max_gen_len = 60
    top_k = 30
    top_p = 0.92
    temperature = 0.75
    repetition_penalty = 1.4
    retrieval_threshold = 0.24

    checkpoint = "chatbot_transformer_seq2seq.pt"


cfg = CFG()
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


def clean_text(s: str):
    s = str(s)
    s = re.sub(r"\{\{[^}]+\}\}", lambda m: m.group().strip("{}").replace("_", " "), s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(text: str):
    return re.findall(r"[a-zA-Z0-9']+", str(text).lower())


def pick_col(cols, names):
    for n in names:
        if n in cols:
            return n
    return None


def load_pairs(max_pairs):
    ds = load_dataset(cfg.dataset_name)
    tr = ds["train"]
    q_col = pick_col(tr.column_names, ["instruction", "prompt", "question", "input", "user_message", "user"])
    a_col = pick_col(tr.column_names, ["response", "output", "answer", "assistant_message", "bot", "assistant"])
    if q_col is None or a_col is None:
        raise ValueError(f"Could not detect Q/A columns from: {tr.column_names}")

    pairs = []
    for row in tr:
        q, a = clean_text(row[q_col]), clean_text(row[a_col])
        if q and a:
            pairs.append((q, a))
        if len(pairs) >= max_pairs:
            break

    print(f"Loaded pairs: {len(pairs)}")
    print(f"Columns used: {q_col} -> {a_col}")
    return pairs


def split_pairs(pairs, val_ratio=0.1):
    idx = list(range(len(pairs)))
    random.Random(cfg.seed).shuffle(idx)
    cut = int(len(idx) * (1 - val_ratio))
    tr_idx, va_idx = idx[:cut], idx[cut:]
    train_pairs = [pairs[i] for i in tr_idx]
    val_pairs = [pairs[i] for i in va_idx]
    return train_pairs, val_pairs


def build_vocab(pairs, max_vocab):
    c = Counter()
    for q, a in pairs:
        c.update(tokenize(q))
        c.update(tokenize(a))
    itos = [PAD, BOS, EOS, UNK] + [w for w, _ in c.most_common(max(0, max_vocab - 4))]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def encode(text, stoi, max_len, add_bos=False, add_eos=True):
    ids = [stoi.get(t, stoi[UNK]) for t in tokenize(text)]
    if add_bos:
        ids = [stoi[BOS]] + ids
    if add_eos:
        ids = ids + [stoi[EOS]]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


def decode(ids, itos, pad_id, bos_id, eos_id):
    out = []
    for i in ids:
        if i == eos_id:
            break
        if i in (pad_id, bos_id):
            continue
        if 0 <= i < len(itos):
            out.append(itos[i])
    return clean_text(" ".join(out))


def normalize_tokens(text: str):
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "to", "for", "of", "and", "or",
        "in", "on", "my", "i", "me", "you", "your", "it", "this", "that", "with", "please"
    }
    return [t for t in tokenize(text) if t not in stop]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def retrieve_best_response(user_text, kb_pairs, threshold):
    q_tok = normalize_tokens(user_text)
    best_score = 0.0
    best_resp = None
    for q, a in kb_pairs:
        s = jaccard(q_tok, normalize_tokens(q))
        if s > best_score:
            best_score = s
            best_resp = a
    if best_score >= threshold:
        return clean_text(best_resp)
    return None


def is_low_quality_response(txt: str):
    t = txt.lower()
    bad = [
        "i've grasped that",
        "i'm delighted that you're",
        "let's dive right step by step",
        "this can be frustrating and i'm here to guide you",
        "to proceed i would need some information",
    ]
    if any(p in t for p in bad):
        return True
    words = txt.split()
    if len(words) < 4:
        return True
    if len(set(words)) / max(1, len(words)) < 0.45:
        return True
    return False


class ChatDataset(Dataset):
    def __init__(self, pairs, stoi):
        self.pairs = pairs
        self.stoi = stoi

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        src = torch.tensor(encode(src_text, self.stoi, cfg.src_max_len, add_bos=False, add_eos=True), dtype=torch.long)
        tgt_full = encode(tgt_text, self.stoi, cfg.tgt_max_len, add_bos=True, add_eos=True)
        tgt_in = torch.tensor(tgt_full[:-1], dtype=torch.long)
        tgt_out = torch.tensor(tgt_full[1:], dtype=torch.long)
        return src, tgt_in, tgt_out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = cfg.d_model
        self.src_emb = nn.Embedding(vocab_size, cfg.d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(vocab_size, cfg.d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(cfg.d_model, max_len=max(cfg.src_max_len, cfg.tgt_max_len) + 64)
        self.tf = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(cfg.d_model, vocab_size)

    def make_tgt_mask(self, t, device):
        return torch.triu(torch.ones(t, t, device=device), diagonal=1).bool()

    def forward(self, src, tgt_in):
        src_pad = (src == self.pad_id)
        tgt_pad = (tgt_in == self.pad_id)
        tgt_mask = self.make_tgt_mask(tgt_in.size(1), tgt_in.device)
        src_e = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_e = self.pos(self.tgt_emb(tgt_in) * math.sqrt(self.d_model))
        out = self.tf(
            src=src_e,
            tgt=tgt_e,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.fc(out)


def evaluate_model(model, loader, vocab_size, pad_id):
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = src.to(cfg.device), tgt_in.to(cfg.device), tgt_out.to(cfg.device)
            logits = model(src, tgt_in)
            loss = loss_fn(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            total_loss += float(loss.item())

            pred = logits.argmax(dim=-1)
            mask = (tgt_out != pad_id)
            correct_tokens += int((pred[mask] == tgt_out[mask]).sum().item())
            total_tokens += int(mask.sum().item())

    avg_loss = total_loss / max(1, len(loader))
    ppl = math.exp(min(avg_loss, 20))
    tok_acc = correct_tokens / max(1, total_tokens)
    return avg_loss, ppl, tok_acc


def train_model(epochs=None, max_pairs=None):
    if epochs is not None:
        cfg.epochs = int(epochs)
    if max_pairs is not None:
        cfg.max_pairs = int(max_pairs)

    pairs = load_pairs(cfg.max_pairs)
    train_pairs, val_pairs = split_pairs(pairs, cfg.val_ratio)
    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    stoi, itos = build_vocab(train_pairs, cfg.max_vocab)
    pad_id = stoi[PAD]
    vocab_size = len(itos)

    train_loader = DataLoader(ChatDataset(train_pairs, stoi), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ChatDataset(val_pairs, stoi), batch_size=cfg.batch_size, shuffle=False)

    model = TransformerSeq2Seq(vocab_size, pad_id).to(cfg.device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=2
    )

    best_val = float('inf')
    for ep in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0

        for src, tgt_in, tgt_out in train_loader:
            src, tgt_in, tgt_out = src.to(cfg.device), tgt_in.to(cfg.device), tgt_out.to(cfg.device)
            logits = model(src, tgt_in)
            loss = loss_fn(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.item())

        tr_loss = tr_loss / max(1, len(train_loader))
        val_loss, val_ppl, val_acc = evaluate_model(model, val_loader, vocab_size, pad_id)
        scheduler.step(val_loss)

        print(f"Epoch {ep}/{cfg.epochs} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f} | val_tok_acc={val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "stoi": stoi,
                    "itos": itos,
                    "kb_pairs": train_pairs[:8000],
                },
                cfg.checkpoint,
            )
            print(f"Saved best checkpoint: {cfg.checkpoint}")


def top_k_top_p_sample(logits, k=6, p=0.85, temperature=0.7):
    logits = logits / max(temperature, 1e-6)
    k = max(1, min(k, logits.size(-1)))
    v, i = torch.topk(logits, k)
    probs = torch.softmax(v, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    mask = cdf > p
    mask[1:] = mask[:-1].clone()
    mask[0] = False
    v = v.masked_fill(mask, -1e9)
    probs = torch.softmax(v, dim=-1)
    pick = torch.multinomial(probs, 1)
    return int(i[pick].item())


def load_model():
    ckpt = torch.load(cfg.checkpoint, map_location=cfg.device)
    stoi, itos = ckpt["stoi"], ckpt["itos"]
    model = TransformerSeq2Seq(len(itos), stoi[PAD]).to(cfg.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    kb_pairs = ckpt.get("kb_pairs", [])
    return model, stoi, itos, kb_pairs


def generate(model, stoi, itos, src_text):
    pad_id, bos_id, eos_id = stoi[PAD], stoi[BOS], stoi[EOS]
    src = torch.tensor([encode(src_text, stoi, cfg.src_max_len, add_bos=False, add_eos=True)], dtype=torch.long, device=cfg.device)

    out_ids = [bos_id]
    with torch.no_grad():
        for _ in range(cfg.max_gen_len):
            tgt = torch.tensor([out_ids], dtype=torch.long, device=cfg.device)
            logits = model(src, tgt)[:, -1, :].squeeze(0)
            for prev in set(out_ids[1:]):
                logits[prev] /= cfg.repetition_penalty
            nxt = top_k_top_p_sample(logits, k=cfg.top_k, p=cfg.top_p, temperature=cfg.temperature)
            if nxt == eos_id:
                break
            out_ids.append(nxt)

    txt = decode(out_ids[1:], itos, pad_id, bos_id, eos_id)
    if not txt:
        return "please rephrase your query in one line."
    w = txt.split()
    if len(w) > 40:
        joined = " ".join(w[:40])
        cut_pos = -1
        for punct in [".", "!", "?"]:
            pos = joined.rfind(punct)
            if pos > 20:
                cut_pos = max(cut_pos, pos)
        if cut_pos > 0:
            txt = joined[:cut_pos + 1]
        else:
            txt = joined + "."
    return txt


def chat_mode():
    model, stoi, itos, kb_pairs = load_model()
    history = deque(maxlen=2 * cfg.history_turns * 2)
    print("\nChatbot ready. Type 'exit' to stop.\n")

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            print("Bot: goodbye")
            break

        context = list(history)[-2 * cfg.history_turns:]
        context.append("user: " + user)
        src_text = " [SEP] ".join(context)

        bot = generate(model, stoi, itos, src_text)
        retrieved = retrieve_best_response(user, kb_pairs, cfg.retrieval_threshold) if kb_pairs else None
        if is_low_quality_response(bot) and retrieved:
            bot = retrieved
        elif retrieved and len(bot.split()) > 45:
            bot = retrieved

        print("Bot:", bot)

        history.append("user: " + user)
        history.append("bot: " + bot)


def run_pipeline(train_epochs=2, max_pairs=5000, chat_after_train=True):
    train_model(epochs=train_epochs, max_pairs=max_pairs)
    if chat_after_train:
        chat_mode()


if __name__ == "__main__":
    run_pipeline(train_epochs=8, max_pairs=20000, chat_after_train=False)

