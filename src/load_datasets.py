from pathlib import Path
import pandas as pd
import json, re, random

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"          # where your CSVs live
OUT  = ROOT / "data" / "processed"    # where JSONLs should be written
OUT.mkdir(parents=True, exist_ok=True)

def _strip_html(x: str) -> str:
    if not isinstance(x, str): return ""
    return re.sub(r"<[^>]+>", "", x).replace("&nbsp;", " ").strip()

def _to_jsonl(rows, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            if r.get("input") and r.get("output"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Saved:", path, f"({len(rows)} rows)")

def make_counselchat_from_csv(csv_path: Path, out_prefix: str = "counselchat", seed: int = 42):
   
    assert csv_path.exists(), f"CSV not found: {csv_path}"
    df = pd.read_csv(
        csv_path,
        engine="python",      # tolerant parser
        encoding="utf-8",
        on_bad_lines="skip",
        dtype=str
    )

    # Ensure required columns
    for col in ["questionTitle", "questionText", "answerText"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}. Found: {list(df.columns)}")

    # Drop very short/empty answers
    df["answerText"] = df["answerText"].fillna("").astype(str)
    df = df[df["answerText"].str.len() >= 30].copy()

    # Build input/output
    def mk_input(row):
        t = _strip_html(row.get("questionTitle", ""))
        q = _strip_html(row.get("questionText", ""))
        return (t + "\n\n" + q).strip()

    rows = [{"input": mk_input(r), "output": _strip_html(r.get("answerText", ""))}
            for _, r in df.iterrows()]

    # Deduplicate by input
    seen, uniq = set(), []
    for r in rows:
        if r["input"] and r["input"] not in seen:
            uniq.append(r); seen.add(r["input"])

    # 90/10 split
    random.Random(seed).shuffle(uniq)
    n = len(uniq)
    cut = max(1, int(0.9 * n))
    train, val = uniq[:cut], uniq[cut:]

    p_train = OUT / f"{out_prefix}_train.jsonl"
    p_val   = OUT / f"{out_prefix}_val.jsonl"
    _to_jsonl(train, p_train)
    _to_jsonl(val,   p_val)
    return str(p_train), str(p_val)

def make_empathetic_dialogues():
    """Optional helper to build from Hugging Face (not used by default)."""
    from datasets import load_dataset
    ds_train = load_dataset("facebook/empathetic_dialogues", split="train")
    ds_val   = load_dataset("facebook/empathetic_dialogues", split="validation")

    def row_map(ex):
        return {"input": str(ex.get("context","")).strip(),
                "output": str(ex.get("utterance","")).strip()}

    train_rows = [row_map(r) for r in ds_train]
    val_rows   = [row_map(r) for r in ds_val]

    p_train = OUT / "empathetic_train.jsonl"
    p_val   = OUT / "empathetic_validation.jsonl"
    _to_jsonl(train_rows, p_train)
    _to_jsonl(val_rows,   p_val)
    return str(p_train), str(p_val)

if __name__ == "__main__":
    csv = RAW / "counselchat-data.csv"   
    print("Building from:", csv)
    print(make_counselchat_from_csv(csv, out_prefix="counselchat"))
