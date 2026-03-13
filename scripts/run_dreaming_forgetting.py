"""Sequential retention experiment: naive vs dreaming with held-out topic eval.

This version is designed for larger A100 runs:
1. Freeze topic corpora once and store them in the checkpoint.
2. Build held-out eval splits that are never used for training.
3. Balance train-set size across topics.
4. Run multiple seeds.
5. Compare naive sequential fine-tuning vs dreaming regularization.

Example:
    python scripts/run_dreaming_forgetting.py --model meta-llama/Llama-3.2-1B
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time

import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wm.cfg import ChunkCfg
from wm.chunk import Chunker
from wm.eval.anchor import AnchorEval
from wm.ingest.exa import ExaSrc
from wm.recipe import EATRDRunner
from wm.search.content_filter import clean_text
from wm.search.dedup import dedup_chunks
from wm.store import EpisodeStore
from wm.types import Episode


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("retention")


MODEL_NAME = None
OUT_PATH = None
CHECKPOINT_PATH = None
STATE_DIR = None
EXPERIMENT_VERSION = 2

LORA_R = 8
LORA_ALPHA = 16
STEPS = int(os.environ.get("RETENTION_STEPS", "200"))
BS = int(os.environ.get("RETENTION_BS", "2"))
LR = 2e-4
MAX_LEN = 128
DREAM_N = 2
DREAM_LEN = 32
TRAIN_FRAC = float(os.environ.get("RETENTION_TRAIN_FRAC", "0.8"))
MAX_EP_PER_QUERY = int(os.environ.get("RETENTION_EP_PER_QUERY", "5"))
MAX_DREAM_PROMPTS_PER_TOPIC = int(os.environ.get("RETENTION_DREAM_PER_TOPIC", "12"))
DATA_SEED = int(os.environ.get("RETENTION_DATA_SEED", "42"))
TARGET_TRAIN_ROWS = int(os.environ.get("RETENTION_TRAIN_ROWS", "0"))
SEED_TEXT = os.environ.get("RETENTION_SEEDS", "42,43,44")
EXA_KEY = os.environ.get("EXA_API_KEY", "e337f35a-e56c-4ae7-8596-f44959053342")
HF_LLAMA_TOKEN = "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"

_DEFAULT_RESULTS_ROOT = os.environ.get("WM_RESULTS_DIR", os.path.join(os.getcwd(), "results"))
_RETENTION_OUT = os.environ.get("RETENTION_DIR", os.path.join(_DEFAULT_RESULTS_ROOT, "retention"))

TOPICS = [
    ("forensics", [
        "unsolved cold case forensic evidence DNA 2024",
        "DNA forensic breakthroughs criminal investigation",
        "forensic science new techniques crime solving",
    ]),
    ("chemistry", [
        "organic reaction mechanisms catalysis synthesis",
        "CRISPR gene editing molecular biology",
        "computational chemistry molecular dynamics simulation",
    ]),
    ("finance", [
        "quantitative trading strategies risk modeling",
        "credit risk assessment Basel IV regulations",
        "earnings analysis SEC filings equity valuation",
    ]),
    ("legal", [
        "Fourth Amendment digital privacy warrant cell phone search",
        "constitutional law Supreme Court precedent free speech",
        "intellectual property AI generated content copyright law",
    ]),
    ("medicine", [
        "clinical pharmacology drug interactions adverse effects",
        "oncology targeted therapy biomarkers immunotherapy",
        "pathophysiology cardiology heart failure mechanisms",
    ]),
]

ANCHORS = [
    "The Earth revolves around the Sun.",
    "Water freezes at zero degrees Celsius.",
    "DNA carries genetic information.",
    "Gravity pulls objects toward Earth.",
    "The chemical formula for water is H2O.",
]

CONFIGS = [
    ("naive", False),
    ("dreaming", True),
]

SYNTHETIC_TEXTS = {
    "forensics": [
        "Forensic DNA analysis has revolutionized criminal investigations. Modern techniques like touch DNA and familial searching allow investigators to identify suspects from trace biological material left at crime scenes.",
        "Cold case investigations have seen renewed interest with advances in genetic genealogy. Investigators can now use public DNA databases to identify suspects in decades-old unsolved cases through distant relative matching.",
        "Rapid DNA technology enables crime labs to process DNA samples in under two hours, compared with the traditional timeline of weeks or months, with significant implications for active investigations.",
        "Forensic toxicology has improved detection of novel psychoactive substances, allowing crime labs to identify low-concentration compounds in both living and post-mortem samples.",
        "Digital forensics increasingly relies on smartphone extraction, cloud artifact recovery, and metadata correlation to reconstruct timelines for criminal investigations.",
    ],
    "chemistry": [
        "Catalytic reactions form the foundation of modern organic synthesis. Transition metal catalysts enable selective bond formation with stereochemical control that was previously difficult to achieve.",
        "CRISPR-Cas9 gene editing uses guide RNA to direct the Cas9 protein to specific genomic locations, enabling precise modifications to DNA sequences with broad applications in medicine and agriculture.",
        "Molecular dynamics simulations model atomic interactions over time, revealing conformational changes in proteins and reaction pathways that are difficult to observe experimentally.",
        "Polymer chemistry studies how monomers react to form macromolecules with tunable mechanical, thermal, and electrical properties for industrial applications.",
        "Green chemistry emphasizes solvent reduction, atom economy, and safer reaction conditions to reduce environmental impact without sacrificing yield.",
    ],
    "finance": [
        "Quantitative trading strategies use statistical models and algorithmic execution to identify and exploit market inefficiencies. Common approaches include mean reversion, momentum, and pairs trading.",
        "Credit risk assessment under Basel IV requires banks to use standardized approaches for many asset classes, with reduced reliance on internal models for capital calculations.",
        "Discounted cash flow analysis and comparable company analysis are primary valuation methods used in equity research and review of SEC filings.",
        "Earnings quality analysis focuses on revenue recognition, margin sustainability, free cash flow conversion, and balance-sheet risk signals.",
        "Portfolio risk modeling estimates variance, factor exposure, and drawdown sensitivity under different market scenarios to support allocation decisions.",
    ],
    "legal": [
        "The Fourth Amendment constrains unreasonable searches and seizures, and modern digital-privacy cases often focus on how warrant requirements apply to phones, cloud data, and location records.",
        "Constitutional law relies heavily on precedent, where courts distinguish, extend, or limit earlier rulings when deciding disputes involving speech, privacy, and due process.",
        "Intellectual property law for AI-generated content turns on questions of authorship, ownership, copyrightability, and whether training or generation infringes protected works.",
        "Criminal procedure evaluates when the exclusionary rule applies, especially when digital evidence is obtained through invalid warrants or overly broad search methods.",
        "International humanitarian law regulates conduct in armed conflict through principles such as distinction, proportionality, and military necessity.",
    ],
    "medicine": [
        "Clinical pharmacology studies how drugs are absorbed, distributed, metabolized, and excreted, and it focuses on dosing, adverse effects, and important drug-drug interactions.",
        "Targeted cancer therapies act on specific molecular pathways, while biomarker testing helps identify which patients are most likely to benefit from a given treatment.",
        "Heart failure pathophysiology involves impaired cardiac output, neurohormonal compensation, fluid retention, and progressive ventricular remodeling.",
        "Immunotherapy activates the immune system against tumors, but its benefits and toxicities depend on tumor biology, checkpoint signaling, and patient-specific factors.",
        "Antimicrobial stewardship aims to optimize antibiotic choice and duration while reducing resistance, toxicity, and unnecessary broad-spectrum exposure.",
    ],
}


def pr(msg: str) -> None:
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}", flush=True)


def parse_seeds(text: str) -> list[int]:
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("RETENTION_SEEDS must contain at least one integer")
    return seeds


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "model"


def _login_for_model(model_name: str) -> None:
    _ = model_name
    login(HF_LLAMA_TOKEN)


def _configure(args) -> None:
    global MODEL_NAME, OUT_PATH, CHECKPOINT_PATH, STATE_DIR
    MODEL_NAME = args.model
    _login_for_model(MODEL_NAME)
    if args.out:
        OUT_PATH = args.out
    else:
        fname = f"retention_{_sanitize(MODEL_NAME)}.json"
        OUT_PATH = os.path.join(_RETENTION_OUT, fname)
    stem, ext = os.path.splitext(OUT_PATH)
    if not ext:
        ext = ".json"
        OUT_PATH = stem + ext
    CHECKPOINT_PATH = stem + "_checkpoint" + ext
    STATE_DIR = stem + "_state"


def _save_checkpoint(state: dict) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH) or ".", exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)
    log.info("checkpoint saved -> %s", CHECKPOINT_PATH)


def _load_checkpoint() -> dict:
    if not os.path.exists(CHECKPOINT_PATH):
        return {}
    try:
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    except Exception as e:
        log.warning("failed loading checkpoint: %s", e)
        return {}


def _adapter_path(seed: int, config_name: str, topic_idx: int) -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, f"s{seed}_{config_name}_t{topic_idx}_adapter.pt")


def _adapter_state(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}


def _save_adapter(model, seed: int, config_name: str, topic_idx: int) -> str:
    path = _adapter_path(seed, config_name, topic_idx)
    torch.save(_adapter_state(model), path)
    return path


def _load_adapter_state(model, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=False)


def _load_adapter(model, path: str) -> None:
    state = torch.load(path, map_location="cpu")
    _load_adapter_state(model, state)


def _row_hash(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(row.get("text", "").encode("utf-8", errors="ignore"))
        h.update(str(row.get("authority", 1.0)).encode("ascii"))
    return h.hexdigest()[:16]


def _rows_to_dataset(rows: list[dict]) -> Dataset:
    if not rows:
        return Dataset.from_list([{"text": "", "authority": 1.0}]).select([])
    return Dataset.from_list(rows)


def _device_name() -> str:
    if not torch.cuda.is_available():
        return "CPU only"
    return torch.cuda.get_device_name(0)


def _anchor_reference(model, tok, anchors: list[str], max_len: int = 128) -> list[dict]:
    model.eval()
    dev = next(model.parameters()).device
    refs = []
    with torch.no_grad():
        for anchor in anchors:
            enc = tok(anchor, return_tensors="pt", truncation=True, max_length=max_len)
            enc_dev = {k: v.to(dev) for k, v in enc.items()}
            logits = model(**enc_dev).logits.detach().cpu().float()
            refs.append({"anchor": anchor, "inputs": enc, "logits": logits})
    return refs


def _drift_kl_to_reference(model, refs: list[dict]) -> float:
    model.eval()
    dev = next(model.parameters()).device
    total = 0.0
    n = 0
    with torch.no_grad():
        for ref in refs:
            enc = {k: v.to(dev) for k, v in ref["inputs"].items()}
            logits_new = model(**enc).logits
            p = F.softmax(ref["logits"].to(logits_new.device, dtype=logits_new.dtype), dim=-1)
            q = F.log_softmax(logits_new, dim=-1)
            total += F.kl_div(q, p, reduction="batchmean").clamp_min(0.0).item()
            n += 1
    return total / max(n, 1)


def _serialize_episode(ep: Episode) -> dict:
    return {
        "url": ep.url,
        "title": ep.title,
        "body": ep.body,
        "authority": ep.authority,
        "topic": ep.topic,
    }


def _data_signature() -> dict:
    return {
        "version": EXPERIMENT_VERSION,
        "topic_names": [name for name, _ in TOPICS],
        "train_frac": TRAIN_FRAC,
        "data_seed": DATA_SEED,
        "max_ep_per_query": MAX_EP_PER_QUERY,
        "target_train_rows": TARGET_TRAIN_ROWS,
    }


def fetch_topic(topic_name: str, queries: list[str]) -> list[Episode]:
    if not EXA_KEY:
        log.warning("%s: no EXA_API_KEY set, using synthetic fallback", topic_name)
        return _synthetic_episodes(topic_name)
    os.environ["EXA_API_KEY"] = EXA_KEY
    src = ExaSrc()
    episodes = []
    for query in queries:
        try:
            fetched = src.fetch(query, n=MAX_EP_PER_QUERY)
            for ep in fetched:
                ep.topic = topic_name
            episodes.extend(fetched)
            log.info("%s | '%s': %d episodes", topic_name, query, len(fetched))
        except Exception as e:
            log.warning("%s | '%s' failed: %s", topic_name, query, e)
    return episodes or _synthetic_episodes(topic_name)


def _synthetic_episodes(topic_name: str) -> list[Episode]:
    texts = SYNTHETIC_TEXTS.get(topic_name, [])
    return [
        Episode(
            url=f"https://synthetic.example.com/{topic_name}/{i}",
            title=f"{topic_name.title()} Article {i}",
            body=text,
            authority=0.8,
            topic=topic_name,
        )
        for i, text in enumerate(texts)
    ]


def _prepare_rows(topic_name: str, episodes: list[Episode]) -> tuple[list[dict], list[dict], dict]:
    filtered = []
    for ep in episodes:
        cleaned = clean_text(ep.body)
        if not cleaned:
            continue
        ep_copy = copy.copy(ep)
        ep_copy.body = cleaned
        filtered.append(ep_copy)
    episodes = filtered

    db_path = f"/tmp/wm_retention_{topic_name}.db"
    store = EpisodeStore(db_path)
    store.put_many(episodes)
    chunks = Chunker(ChunkCfg(max_tok=256, overlap=32)).chunk_many(store.all())
    chunks = dedup_chunks(chunks)
    store.close()
    if os.path.exists(db_path):
        os.remove(db_path)

    rows = [{"text": c.text, "authority": c.authority, "eid": c.eid, "idx": c.idx} for c in chunks]
    if len(rows) < 2:
        raise RuntimeError(f"{topic_name}: not enough rows after preprocessing ({len(rows)})")

    rng = random.Random(f"{DATA_SEED}:{topic_name}:split")
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    cut = max(1, min(len(rows) - 1, int(round(len(rows) * TRAIN_FRAC))))
    train_rows = [rows[i] for i in idx[:cut]]
    eval_rows = [rows[i] for i in idx[cut:]]
    stats = {
        "n_episodes": len(episodes),
        "n_chunks": len(chunks),
        "n_train_raw": len(train_rows),
        "n_eval": len(eval_rows),
        "train_hash_raw": _row_hash(train_rows),
        "eval_hash": _row_hash(eval_rows),
    }
    return train_rows, eval_rows, stats


def load_or_prepare_data(state: dict) -> list[dict]:
    expected = _data_signature()
    if state.get("data") and state.get("data_meta") == expected:
        log.info("resuming frozen topic data from checkpoint")
        return state["data"]

    if state.get("data"):
        log.warning("data signature mismatch; rebuilding frozen topic snapshot")
    state.pop("runs", None)

    data = []
    for topic_name, queries in TOPICS:
        episodes = fetch_topic(topic_name, queries)
        train_rows, eval_rows, stats = _prepare_rows(topic_name, episodes)
        data.append({
            "name": topic_name,
            "queries": queries,
            "episodes": [_serialize_episode(ep) for ep in episodes],
            "train_rows": train_rows,
            "eval_rows": eval_rows,
            "stats": stats,
        })

    min_train = min(len(topic["train_rows"]) for topic in data)
    balance_n = min_train if TARGET_TRAIN_ROWS <= 0 else min(min_train, TARGET_TRAIN_ROWS)
    for topic in data:
        rng = random.Random(f"{DATA_SEED}:{topic['name']}:balance")
        idx = list(range(len(topic["train_rows"])))
        rng.shuffle(idx)
        topic["train_rows"] = [topic["train_rows"][i] for i in idx[:balance_n]]
        topic["stats"]["n_train_balanced"] = len(topic["train_rows"])
        topic["stats"]["balance_n"] = balance_n
        topic["stats"]["train_hash_balanced"] = _row_hash(topic["train_rows"])
        print(
            f"  {topic['name']}: episodes={len(topic['episodes'])} chunks={topic['stats']['n_chunks']} "
            f"train={topic['stats']['n_train_balanced']} eval={topic['stats']['n_eval']}"
        )

    state["data"] = data
    state["data_meta"] = expected
    _save_checkpoint(state)
    return data


def make_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    load_kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    if torch.cuda.is_available():
        model = model.cuda()
    lc = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lc)
    return model, tok


def eval_rows(model, tok, rows: list[dict], max_len: int = 128) -> dict:
    if not rows:
        return {"ppl": float("inf"), "acc": 0.0, "n_rows": 0}
    dev = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_rows = 0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for row in rows:
            text = row.get("text", "")
            if not text:
                continue
            enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
            enc = {k: v.to(dev) for k, v in enc.items()}
            out = model(**enc, labels=enc["input_ids"])
            total_loss += out.loss.item()
            total_rows += 1
            ids = enc["input_ids"]
            if ids.shape[1] < 2:
                continue
            preds = out.logits[:, :-1].argmax(dim=-1)
            tgts = ids[:, 1:]
            total_correct += (preds == tgts).sum().item()
            total_tokens += tgts.numel()
    if total_rows == 0:
        return {"ppl": float("inf"), "acc": 0.0, "n_rows": 0}
    return {
        "ppl": math.exp(total_loss / total_rows),
        "acc": total_correct / max(total_tokens, 1),
        "n_rows": total_rows,
    }


def eval_all_topics(model, tok, data: list[dict]) -> dict:
    return {topic["name"]: eval_rows(model, tok, topic["eval_rows"], max_len=MAX_LEN) for topic in data}


def build_dream_prompts(data: list[dict], current_topic_idx: int) -> list[str]:
    prompts = list(ANCHORS)
    for j in range(current_topic_idx):
        rows = data[j]["train_rows"][:MAX_DREAM_PROMPTS_PER_TOPIC]
        prompts.extend(row.get("text", "")[: MAX_LEN * 4] for row in rows if row.get("text"))
    seen = set()
    unique = []
    for prompt in prompts:
        if prompt and prompt not in seen:
            seen.add(prompt)
            unique.append(prompt)
    return unique or list(ANCHORS)


def forgetting_summary(updates: list[dict], topic_names: list[str]) -> dict:
    out = {}
    mean_vals = []
    for j, topic_name in enumerate(topic_names):
        vals = [u["evals"][topic_name]["acc"] for u in updates if u["topic_idx"] >= j]
        if not vals:
            continue
        best = max(vals)
        first = vals[0]
        final = vals[-1]
        forgetting = best - final
        out[topic_name] = {
            "first_after_learning": first,
            "best_seen": best,
            "final": final,
            "forgetting": forgetting,
            "retention_delta": final - first,
            "n_measurements": len(vals),
        }
        if len(vals) >= 2:
            mean_vals.append(forgetting)
    if mean_vals:
        out["_mean_forgetting"] = sum(mean_vals) / len(mean_vals)
    return out


def run_config(
    seed: int,
    config_name: str,
    dream_on: bool,
    model,
    base_adapter_state: dict[str, torch.Tensor],
    base_anchor_ref: list[dict],
    tok,
    data: list[dict],
    run_state: dict,
    state: dict,
    topic_names: list[str],
) -> dict:
    pr(f"CONFIG: {config_name}  (dreaming={'ON' if dream_on else 'OFF'})")
    cfg_state = run_state.setdefault("configs", {}).setdefault(config_name, {"dream_on": dream_on, "updates": []})
    updates = list(cfg_state.get("updates", []))
    done_count = sum(1 for u in updates if u.get("_status") == "done")

    _load_adapter_state(model, base_adapter_state)
    if done_count > 0:
        ckpt = _adapter_path(seed, config_name, done_count - 1)
        if os.path.exists(ckpt):
            _load_adapter(model, ckpt)
            print(f"  resumed from t{done_count - 1} adapter")
        else:
            log.warning("%s: adapter missing, restarting from scratch", config_name)
            updates = []
            done_count = 0
            _load_adapter_state(model, base_adapter_state)

    for i, topic in enumerate(data):
        if i < done_count:
            print(f"  [t{i}] {topic['name']}: RESUMED")
            continue

        pr(f"{config_name} | t{i}: {topic['name']}  ({i + 1}/{len(data)})")
        set_seed(seed * 1000 + i)
        t0 = time.time()

        prev_anchor_ref = _anchor_reference(model, tok, ANCHORS[:3], max_len=64)
        teacher = copy.deepcopy(model).eval() if dream_on else None
        ds_train = _rows_to_dataset(topic["train_rows"])
        dream_prompts = build_dream_prompts(data, i)
        lam_init = 1.0 if dream_on else 0.0
        runner = EATRDRunner(
            lr=LR,
            max_steps=STEPS,
            bs=BS,
            temp=2.0,
            eps_min=0.01,
            alpha=0.5,
            rho=0.0,
            lam_init=lam_init,
            max_len=MAX_LEN,
            dream_n=DREAM_N,
            dream_len=DREAM_LEN,
            d_targ=0.005,
            use_pi=False,
        )
        tr = runner.run(model, teacher, ds_train, dream_prompts, tok)
        print(
            f"  train: loss={tr.loss:.4f} dream_loss={tr.dream_loss:.4f} "
            f"lambda_final={tr.extras.get('lambda', lam_init):.4f} prompts={len(dream_prompts)}"
        )

        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        evals = eval_all_topics(model, tok, data)
        anchor_nll = AnchorEval(model, tok, ANCHORS).nll()
        drift_base = _drift_kl_to_reference(model, base_anchor_ref)
        drift_prev = _drift_kl_to_reference(model, prev_anchor_ref)
        print(f"  anchor_nll={anchor_nll:.4f} drift_base={drift_base:.6f} drift_prev={drift_prev:.6f}")

        ckpt_path = _save_adapter(model, seed, config_name, i)
        entry = {
            "topic": topic["name"],
            "topic_idx": i,
            "train_rows": len(topic["train_rows"]),
            "dream_prompts_n": len(dream_prompts),
            "train_loss": tr.loss,
            "dream_loss": tr.dream_loss,
            "lambda_final": tr.extras.get("lambda", lam_init),
            "steps": tr.steps,
            "evals": evals,
            "anchor_nll": anchor_nll,
            "drift_kl_base": drift_base,
            "drift_kl_prev": drift_prev,
            "time": time.time() - t0,
            "_adapter_ckpt": ckpt_path,
            "_status": "done",
        }
        updates.append(entry)
        cfg_state["updates"] = updates
        cfg_state["summary"] = forgetting_summary(updates, topic_names)
        _save_checkpoint(state)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "dream_on": dream_on,
        "updates": updates,
        "summary": forgetting_summary(updates, topic_names),
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def aggregate_runs(all_runs: dict, topic_names: list[str]) -> dict:
    agg = {}
    for config_name, _ in CONFIGS:
        forgetting_vals = []
        anchor_vals = []
        drift_vals = []
        final_acc_vals = []
        for seed_payload in all_runs.values():
            cfg = seed_payload["configs"].get(config_name)
            if not cfg or not cfg["updates"]:
                continue
            final_update = cfg["updates"][-1]
            forgetting_vals.append(cfg.get("summary", {}).get("_mean_forgetting", 0.0))
            anchor_vals.append(final_update["anchor_nll"])
            drift_vals.append(final_update["drift_kl_base"])
            final_acc_vals.append(_mean([final_update["evals"][topic]["acc"] for topic in topic_names]))
        agg[config_name] = {
            "n_seeds": len(forgetting_vals),
            "mean_forgetting_mean": _mean(forgetting_vals),
            "mean_forgetting_std": _std(forgetting_vals),
            "final_anchor_nll_mean": _mean(anchor_vals),
            "final_anchor_nll_std": _std(anchor_vals),
            "final_drift_base_mean": _mean(drift_vals),
            "final_drift_base_std": _std(drift_vals),
            "final_avg_acc_mean": _mean(final_acc_vals),
            "final_avg_acc_std": _std(final_acc_vals),
        }
    return agg


def print_analysis(all_runs: dict, data: list[dict]) -> None:
    topic_names = [topic["name"] for topic in data]
    for seed_key, seed_payload in all_runs.items():
        baseline = seed_payload["baseline"]
        pr(f"SEED {seed_key}: HELD-OUT TOPIC ACCURACY")
        header = f"{'config/update':<18}" + "".join(f"{name:>14}" for name in topic_names)
        print(header)
        print("-" * len(header))
        base_row = f"{'baseline':<18}"
        for topic_name in topic_names:
            base_row += f"{baseline['topic_evals'][topic_name]['acc']:>14.4f}"
        print(base_row)
        for config_name, _ in CONFIGS:
            payload = seed_payload["configs"].get(config_name, {"updates": []})
            for update in payload["updates"]:
                row = f"{config_name}/t{update['topic_idx']:<11}"
                for topic_name in topic_names:
                    row += f"{update['evals'][topic_name]['acc']:>14.4f}"
                print(row)

        pr(f"SEED {seed_key}: FORGETTING SUMMARY")
        for config_name, _ in CONFIGS:
            payload = seed_payload["configs"].get(config_name, {"summary": {}, "updates": []})
            summary = payload.get("summary", {})
            print(f"\n{config_name}:")
            for topic_name in topic_names:
                s = summary.get(topic_name)
                if not s:
                    continue
                print(
                    f"  {topic_name:<12} first={s['first_after_learning']:.4f} "
                    f"best={s['best_seen']:.4f} final={s['final']:.4f} "
                    f"forgetting={s['forgetting']:.4f}"
                )
            print(f"  mean_forgetting={summary.get('_mean_forgetting', 0.0):.4f}")

        pr(f"SEED {seed_key}: ANCHOR / DRIFT TRAJECTORY")
        for config_name, _ in CONFIGS:
            payload = seed_payload["configs"].get(config_name, {"updates": []})
            if not payload["updates"]:
                continue
            anchor_vals = " -> ".join(f"{u['anchor_nll']:.3f}" for u in payload["updates"])
            drift_vals = " -> ".join(f"{u['drift_kl_base']:.4f}" for u in payload["updates"])
            print(f"{config_name}: anchor {baseline['anchor_nll']:.3f} -> {anchor_vals}")
            print(f"{config_name}: drift_base {drift_vals}")

    agg = aggregate_runs(all_runs, topic_names)
    pr("AGGREGATE SUMMARY ACROSS SEEDS")
    for config_name, stats in agg.items():
        print(
            f"{config_name}: "
            f"mean_forgetting={stats['mean_forgetting_mean']:.4f}±{stats['mean_forgetting_std']:.4f}  "
            f"final_anchor={stats['final_anchor_nll_mean']:.4f}±{stats['final_anchor_nll_std']:.4f}  "
            f"final_drift={stats['final_drift_base_mean']:.4f}±{stats['final_drift_base_std']:.4f}  "
            f"final_avg_acc={stats['final_avg_acc_mean']:.4f}±{stats['final_avg_acc_std']:.4f}  "
            f"n={stats['n_seeds']}"
        )


def strip_private(obj):
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description="Sequential retention experiment with naive vs dreaming.")
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--out", default=None, help="optional output JSON path")
    args = ap.parse_args()

    _configure(args)
    run_seeds = parse_seeds(SEED_TEXT)
    set_seed(DATA_SEED)
    state = _load_checkpoint()
    state.setdefault("meta", {})
    state["meta"].update({
        "version": EXPERIMENT_VERSION,
        "model": args.model,
        "data_seed": DATA_SEED,
        "run_seeds": run_seeds,
        "steps_per_topic": STEPS,
        "batch_size": BS,
        "lr": LR,
        "train_frac": TRAIN_FRAC,
        "topics": [name for name, _ in TOPICS],
        "configs": [name for name, _ in CONFIGS],
        "target_train_rows": TARGET_TRAIN_ROWS,
    })

    pr("SEQUENTIAL RETENTION EXPERIMENT")
    print(f"Model:   {args.model}")
    print(f"Topics:  {' -> '.join(name for name, _ in TOPICS)}")
    print(f"Configs: {[name for name, _ in CONFIGS]}")
    print(f"Seeds:   {run_seeds}")
    print(f"Steps:   {STEPS} per topic | BS={BS} | LR={LR}")
    print(f"GPU:     {_device_name()}")

    pr("PHASE 1: PREPARE FROZEN TOPIC DATA")
    data = load_or_prepare_data(state)
    topic_names = [topic["name"] for topic in data]

    state.setdefault("runs", {})
    all_runs = {}
    for seed in run_seeds:
        pr(f"PHASE 2/3: SEED {seed} LOAD MODEL + RUN CONFIGS")
        run_key = str(seed)
        run_state = state["runs"].setdefault(run_key, {"seed": seed, "baseline": None, "configs": {}})

        set_seed(seed)
        model, tok = make_model()
        if seed == run_seeds[0]:
            model.print_trainable_parameters()
        base_adapter_state = _adapter_state(model)
        base_anchor_ref = _anchor_reference(model, tok, ANCHORS[:3], max_len=64)

        if run_state.get("baseline"):
            print(f"Seed {seed}: baseline RESUMED from checkpoint")
        else:
            print(f"Seed {seed}: computing baseline on all held-out topic splits...")
            run_state["baseline"] = {
                "topic_evals": eval_all_topics(model, tok, data),
                "anchor_nll": AnchorEval(model, tok, ANCHORS).nll(),
            }
            _save_checkpoint(state)
            print(f"  baseline anchor_nll={run_state['baseline']['anchor_nll']:.4f}")

        for config_name, dream_on in CONFIGS:
            run_state["configs"][config_name] = run_config(
                seed,
                config_name,
                dream_on,
                model,
                base_adapter_state,
                base_anchor_ref,
                tok,
                data,
                run_state,
                state,
                topic_names,
            )
            _save_checkpoint(state)

        all_runs[run_key] = run_state
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_analysis(all_runs, data)
    output = {
        "meta": state["meta"],
        "data_stats": {
            topic["name"]: {
                "queries": topic["queries"],
                "stats": topic["stats"],
            }
            for topic in data
        },
        "runs": strip_private(all_runs),
        "aggregate": aggregate_runs(all_runs, topic_names),
    }
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {OUT_PATH}")
    pr("DONE")


if __name__ == "__main__":
    main()
