"""Sequential retention experiment: naive vs dreaming with held-out topic eval.

1. Fetch fixed corpora for several topics once and freeze them in the checkpoint.
2. Split each topic into train/eval rows once; eval rows are never trained on.
3. Train sequentially across topics.
4. After each update, evaluate on every topic's held-out split.
5. Compare:
   - naive:    no dreaming regularization
   - dreaming: KL regularization to the pre-update teacher using prior-topic prompts

python websleuths-229/scripts/run_dreaming_forgetting.py --model meta-llama/Llama-3.2-1B
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
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wm.cfg import ChunkCfg
from wm.chunk import Chunker
from wm.guard.drift import drift_kl
from wm.ingest.exa import ExaSrc
from wm.recipe import EATRDRunner
from wm.search.content_filter import clean_text
from wm.search.dedup import dedup_chunks
from wm.store import EpisodeStore
from wm.types import Episode
from wm.eval.anchor import AnchorEval


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("retention")


MODEL_NAME = None
OUT_PATH = None
CHECKPOINT_PATH = None
STATE_DIR = None

LORA_R = 8
LORA_ALPHA = 16
STEPS = int(os.environ.get("RETENTION_STEPS", "50"))
BS = int(os.environ.get("RETENTION_BS", "2"))
LR = 2e-4
MAX_LEN = 128
DREAM_N = 2
DREAM_LEN = 32
TRAIN_FRAC = float(os.environ.get("RETENTION_TRAIN_FRAC", "0.8"))
MAX_EP_PER_QUERY = int(os.environ.get("RETENTION_EP_PER_QUERY", "5"))
MAX_DREAM_PROMPTS_PER_TOPIC = int(os.environ.get("RETENTION_DREAM_PER_TOPIC", "12"))
SEED = int(os.environ.get("RETENTION_SEED", "42"))
EXA_KEY = os.environ.get("EXA_API_KEY", "")

_GDRIVE = "/content/drive/MyDrive"
_GDRIVE_OUT = os.path.join(_GDRIVE, "retention")

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
}


def pr(msg: str) -> None:
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "model"


def _configure(args) -> None:
    global MODEL_NAME, OUT_PATH, CHECKPOINT_PATH, STATE_DIR
    MODEL_NAME = args.model
    if args.out:
        OUT_PATH = args.out
    else:
        fname = f"retention_{_sanitize(MODEL_NAME)}.json"
        out_dir = _GDRIVE_OUT if os.path.isdir(_GDRIVE) else "/tmp"
        OUT_PATH = os.path.join(out_dir, fname)
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


def _adapter_path(config_name: str, topic_idx: int) -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, f"{config_name}_t{topic_idx}_adapter.pt")


def _save_adapter(model, config_name: str, topic_idx: int) -> str:
    path = _adapter_path(config_name, topic_idx)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(state, path)
    return path


def _load_adapter(model, path: str) -> None:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)


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
    return f"{torch.cuda.get_device_name(0)}"


def _topic_spec() -> list[dict]:
    return [{"name": name, "queries": queries} for name, queries in TOPICS]


def _serialize_episode(ep: Episode) -> dict:
    return {
        "url": ep.url,
        "title": ep.title,
        "body": ep.body,
        "authority": ep.authority,
        "topic": ep.topic,
    }


def _make_episode(data: dict) -> Episode:
    return Episode(
        url=data["url"],
        title=data.get("title", ""),
        body=data.get("body", ""),
        authority=float(data.get("authority", 0.5)),
        topic=data.get("topic", ""),
    )


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

    rng = random.Random(SEED)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    cut = max(1, min(len(rows) - 1, int(round(len(rows) * TRAIN_FRAC))))
    train_rows = [rows[i] for i in idx[:cut]]
    eval_rows = [rows[i] for i in idx[cut:]]

    stats = {
        "n_episodes": len(episodes),
        "n_chunks": len(chunks),
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "train_hash": _row_hash(train_rows),
        "eval_hash": _row_hash(eval_rows),
    }
    return train_rows, eval_rows, stats


def load_or_prepare_data(state: dict) -> list[dict]:
    data = state.get("data")
    expected = _topic_spec()
    if data:
        names = [t["name"] for t in data]
        if names == [t["name"] for t in expected]:
            log.info("resuming frozen topic data from checkpoint")
            return data
        log.warning("checkpoint topic spec mismatch; rebuilding data snapshot")

    data = []
    for spec in expected:
        topic_name = spec["name"]
        queries = spec["queries"]
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
        print(
            f"  {topic_name}: episodes={len(episodes)} chunks={stats['n_chunks']} "
            f"train={stats['n_train']} eval={stats['n_eval']}"
        )

    state["data"] = data
    _save_checkpoint(state)
    return data


def make_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, trust_remote_code=True)
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
            logits = out.logits
            preds = logits[:, :-1].argmax(dim=-1)
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
    results = {}
    for topic in data:
        results[topic["name"]] = eval_rows(model, tok, topic["eval_rows"], max_len=MAX_LEN)
    return results


def build_dream_prompts(data: list[dict], current_topic_idx: int) -> list[str]:
    prompts = list(ANCHORS)
    for j in range(current_topic_idx):
        rows = data[j]["train_rows"][:MAX_DREAM_PROMPTS_PER_TOPIC]
        prompts.extend(row.get("text", "")[: MAX_LEN * 4] for row in rows if row.get("text"))
    # Deduplicate while keeping order stable.
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


def run_config(config_name: str, dream_on: bool, base_model, base_snap: dict, tok, data: list[dict], state: dict) -> list[dict]:
    pr(f"CONFIG: {config_name}  (dreaming={'ON' if dream_on else 'OFF'})")
    cfg_state = state.setdefault("configs", {}).setdefault(config_name, {"dream_on": dream_on, "updates": []})
    updates = list(cfg_state.get("updates", []))
    done_count = sum(1 for u in updates if u.get("_status") == "done")

    model = copy.deepcopy(base_model)
    model.load_state_dict(base_snap)

    if done_count > 0:
        ckpt = _adapter_path(config_name, done_count - 1)
        if os.path.exists(ckpt):
            _load_adapter(model, ckpt)
            print(f"  resumed from t{done_count - 1} adapter")
        else:
            log.warning("%s: adapter missing, restarting from scratch", config_name)
            updates = []
            done_count = 0
            model.load_state_dict(base_snap)

    for i, topic in enumerate(data):
        if i < done_count:
            print(f"  [t{i}] {topic['name']}: RESUMED")
            continue

        pr(f"{config_name} | t{i}: {topic['name']}  ({i + 1}/{len(data)})")
        set_seed(SEED + i)
        t0 = time.time()

        teacher = copy.deepcopy(model).eval()
        dream_prompts = build_dream_prompts(data, i)
        ds_train = _rows_to_dataset(topic["train_rows"])

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

        evals = eval_all_topics(model, tok, data)
        anchor_nll = AnchorEval(model, tok, ANCHORS).nll()
        drift_base = drift_kl(model, base_model, tok, ANCHORS[:3], max_len=64)
        drift_prev = drift_kl(model, teacher, tok, ANCHORS[:3], max_len=64)
        print(f"  anchor_nll={anchor_nll:.4f} drift_base={drift_base:.6f} drift_prev={drift_prev:.6f}")

        ckpt_path = _save_adapter(model, config_name, i)
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
        _save_checkpoint(state)

        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return updates


def print_analysis(all_results: dict, baseline: dict, data: list[dict]) -> None:
    topic_names = [t["name"] for t in data]

    pr("HELD-OUT TOPIC ACCURACY")
    header = f"{'config/update':<18}" + "".join(f"{name:>14}" for name in topic_names)
    print(header)
    print("-" * len(header))
    base_row = f"{'baseline':<18}"
    for topic_name in topic_names:
        base_row += f"{baseline['topic_evals'][topic_name]['acc']:>14.4f}"
    print(base_row)

    for config_name, payload in all_results.items():
        for update in payload["updates"]:
            row = f"{config_name}/t{update['topic_idx']:<11}"
            for topic_name in topic_names:
                row += f"{update['evals'][topic_name]['acc']:>14.4f}"
            print(row)

    pr("FORGETTING SUMMARY")
    for config_name, payload in all_results.items():
        summary = payload["summary"]
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

    pr("ANCHOR / DRIFT TRAJECTORY")
    for config_name, payload in all_results.items():
        anchor_vals = " -> ".join(f"{u['anchor_nll']:.3f}" for u in payload["updates"])
        drift_vals = " -> ".join(f"{u['drift_kl_base']:.4f}" for u in payload["updates"])
        print(f"{config_name}: anchor {baseline['anchor_nll']:.3f} -> {anchor_vals}")
        print(f"{config_name}: drift_base {drift_vals}")


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
    set_seed(SEED)
    state = _load_checkpoint()
    state.setdefault("meta", {})
    state["meta"].update({
        "model": args.model,
        "seed": SEED,
        "steps_per_topic": STEPS,
        "batch_size": BS,
        "lr": LR,
        "train_frac": TRAIN_FRAC,
        "topics": [name for name, _ in TOPICS],
        "configs": [name for name, _ in CONFIGS],
    })

    pr("SEQUENTIAL RETENTION EXPERIMENT")
    print(f"Model:   {args.model}")
    print(f"Topics:  {' -> '.join(name for name, _ in TOPICS)}")
    print(f"Configs: {[name for name, _ in CONFIGS]}")
    print(f"Seed:    {SEED}")
    print(f"Steps:   {STEPS} per topic | BS={BS} | LR={LR}")
    print(f"GPU:     {_device_name()}")

    pr("PHASE 1: PREPARE FROZEN TOPIC DATA")
    data = load_or_prepare_data(state)

    pr("PHASE 2: LOAD MODEL + BASELINE")
    base_model, tok = make_model()
    base_model.print_trainable_parameters()
    base_snap = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

    baseline = state.get("baseline")
    if baseline:
        print("Baseline: RESUMED from checkpoint")
    else:
        print("Computing baseline on all held-out topic splits...")
        baseline = {
            "topic_evals": eval_all_topics(base_model, tok, data),
            "anchor_nll": AnchorEval(base_model, tok, ANCHORS).nll(),
        }
        state["baseline"] = baseline
        _save_checkpoint(state)
        print(f"  baseline anchor_nll={baseline['anchor_nll']:.4f}")

    pr("PHASE 3: RUN CONFIGS")
    all_results = {}
    for config_name, dream_on in CONFIGS:
        updates = run_config(config_name, dream_on, base_model, base_snap, tok, data, state)
        summary = forgetting_summary(updates, [t["name"] for t in data])
        all_results[config_name] = {
            "dream_on": dream_on,
            "updates": updates,
            "summary": summary,
        }
        state.setdefault("configs", {}).setdefault(config_name, {})["summary"] = summary
        _save_checkpoint(state)

    print_analysis(all_results, baseline, data)

    output = {
        "meta": state["meta"],
        "data_stats": {
            topic["name"]: {
                "queries": topic["queries"],
                "stats": topic["stats"],
            }
            for topic in data
        },
        "baseline": baseline,
        "configs": strip_private(all_results),
    }
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {OUT_PATH}")
    pr("DONE")


if __name__ == "__main__":
    main()
