import csv
import os
import re
import difflib

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama

MAPPING_CSV = "mappingtable.csv"
OUTPUT_DIR = "stories"

SKIP_EMPTY_FIELDS = True
FUZZY_THRESHOLD = 0.90
PRINT_FUZZY_MATCHES = False


def norm_key(s: str) -> str:
    if s is None:
        return ""
    x = s.replace("\ufeff", "").replace("\u00a0", " ")
    x = x.strip().lower()
    x = re.sub(r"\bnuts\s*([23])\b", r"nuts\1", x)
    x = re.sub(r"[(),]", " ", x)
    x = x.replace("/", " ").replace("-", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def norm_key_aggressive(s: str) -> str:
    x = norm_key(s)
    if not x:
        return ""
    stop = {"per", "of", "total", "year", "eur", "million"}
    tokens = [t for t in x.split() if t not in stop]
    y = " ".join(tokens).strip()
    y = re.sub(r"\s+", " ", y)
    return y


def clean_text(s: str) -> str:
    if s is None:
        return ""
    x = s.replace('"', "'")
    x = x.replace("\ufeff", "").replace("\u00a0", " ")
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    if x.upper() == "N/A":
        return ""
    return x


def parse_bool(s: str) -> bool:
    x = clean_text(s).lower()
    return x in {"true", "1", "yes", "y"}


def build_sentence(prefix: str, content: str, suffix: str) -> str:
    c = clean_text(content)
    if not c:
        return ""

    if c.lower() in {"yes", "y"}:
        c = "are"
    elif c.lower() in {"no", "n"}:
        c = "are not"

    p = (prefix or "").strip()
    suf = (suffix or "").strip()

    s = f"{p} {c} {suf}".strip()
    if s and not s.endswith("."):
        s += "."
    return s


def csv_cell(s: str) -> str:
    x = (s or "").replace("\r", " ").replace("\n", " ").strip()
    x = x.replace('"', '""')
    return f'"{x}"'


def load_mapping(mapping_path: str):
    mapping_rows = []
    mapping_by_dbkey = {}
    event_order = []
    seen_events = set()

    with open(mapping_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError("mappingtable.csv is empty")

    for i, r in enumerate(rows):
        if i == 0:
            continue
        if len(r) < 5:
            continue

        db_label_raw = r[0]
        event_label = clean_text(r[1])
        prefix = r[2] if len(r) > 2 else ""
        suffix = r[3] if len(r) > 3 else ""
        is_title = parse_bool(r[4])

        db_key = norm_key(db_label_raw)
        db_key_aggr = norm_key_aggressive(db_label_raw)

        if not db_key or not event_label:
            continue

        mapping_rows.append((db_label_raw, db_key, db_key_aggr, event_label, prefix, suffix, is_title))
        mapping_by_dbkey[db_key] = (event_label, prefix, suffix, is_title)

        if event_label not in seen_events:
            seen_events.add(event_label)
            event_order.append(event_label)

    return mapping_rows, mapping_by_dbkey, event_order


def build_header_resolution(headers_raw, mapping_rows, mapping_by_dbkey):
    resolved = {}

    # exact
    for h_raw in headers_raw:
        h_key = norm_key(h_raw)
        m = mapping_by_dbkey.get(h_key)
        if m:
            resolved[h_key] = m

    # fuzzy
    mapping_aggr_keys = [mr[2] for mr in mapping_rows]

    for h_raw in headers_raw:
        h_key = norm_key(h_raw)
        if not h_key or h_key in resolved:
            continue

        h_aggr = norm_key_aggressive(h_raw)
        if not h_aggr:
            continue

        best = difflib.get_close_matches(h_aggr, mapping_aggr_keys, n=1, cutoff=FUZZY_THRESHOLD)
        if not best:
            continue

        best_aggr = best[0]
        idx = mapping_aggr_keys.index(best_aggr)
        db_label_raw, db_key, db_key_aggr, event_label, prefix, suffix, is_title = mapping_rows[idx]

        resolved[h_key] = (event_label, prefix, suffix, is_title)

        if PRINT_FUZZY_MATCHES:
            sim = difflib.SequenceMatcher(None, h_aggr, best_aggr).ratio()
            print(f"[FUZZY] '{h_raw}' -> '{db_label_raw}' (sim={sim:.2f})")

    return resolved


def serialize_story(events: dict, event_order: list) -> str:
    lines = ["title,description"]
    for event_label in event_order:
        data = events[event_label]
        title = clean_text(data.get("title", "")) or event_label
        desc = " ".join([clean_text(x) for x in data.get("desc", []) if clean_text(x)]).strip()
        lines.append(f"{csv_cell(title)},{csv_cell(desc)}")
    return "\n".join(lines) + "\n"


def process_csv(dataset_csv_path: str, mapping_csv_path: str = MAPPING_CSV, output_dir: str = OUTPUT_DIR):
    mapping_rows, mapping_by_dbkey, event_order = load_mapping(mapping_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(dataset_csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    if not all_rows:
        print("Dataset is empty.")
        return

    headers_raw = all_rows[0]
    headers_keys = [norm_key(h) for h in headers_raw]
    resolved_header_map = build_header_resolution(headers_raw, mapping_rows, mapping_by_dbkey)

    mapped_dataset_cols = sum(1 for k in headers_keys if k and k in resolved_header_map)
    print(f"Dataset columns mapped (exact+fuzzy): {mapped_dataset_cols} of {len(headers_keys)}")

    for row_index in range(1, len(all_rows)):
        row = all_rows[row_index]
        events = {ev: {"title": "", "desc": []} for ev in event_order}

        ncols = min(len(headers_keys), len(row))
        for col in range(ncols):
            h_key = headers_keys[col]
            if not h_key:
                continue

            m = resolved_header_map.get(h_key)
            if not m:
                continue

            field = clean_text(row[col])
            if SKIP_EMPTY_FIELDS and not field:
                continue

            event_label, prefix, suffix, is_title = m
            sentence = build_sentence(prefix, field, suffix)
            if not sentence:
                continue

            if is_title:
                if not events[event_label]["title"]:
                    events[event_label]["title"] = sentence
            else:
                events[event_label]["desc"].append(sentence)

        story_csv = serialize_story(events, event_order)
        out_path = os.path.join(output_dir, f"{row_index}.csv")
        with open(out_path, "w", encoding="utf-8", newline="") as out:
            out.write(story_csv)

        print(f"Story {row_index} saved -> {out_path}")


def process_txt(txt_path: str, output_dir: str = OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Output LLM (testo suddiviso in paragrafi)
    llm_output = useLLM(text)

    # Split sui paragrafi (uno o più a capo consecutivi)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", llm_output) if p.strip()]

    # Costruzione CSV
    lines = ["title,description"]

    for i, paragraph in enumerate(paragraphs, start=1):
        title = f"event-{i}"
        lines.append(f"{csv_cell(title)},{csv_cell(paragraph)}")

    story_csv = "\n".join(lines) + "\n"

    # Nome file output
    base_name = os.path.splitext(os.path.basename(txt_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}.csv")

    with open(out_path, "w", encoding="utf-8", newline="") as out:
        out.write(story_csv)

    print(f"CSV creato da TXT -> {out_path}")
    
def useLLM(narrative: str):
    
    llm = Ollama(
        model="gemma2:9b-instruct-q8_0", 
        system="Divide the provided text into paragraphs without deleting, adding or changing words.", 
        num_ctx=4096, 
        temperature=0.01, 
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    
    events = llm(narrative)
    
    return events


def run(input_path: str):
    """
    input_path è SEMPRE una stringa (percorso file).
    Se finisce con .csv -> genera storie.
    Se finisce con .txt -> stampa contenuto.
    """
    if not isinstance(input_path, str) or not input_path.strip():
        raise ValueError("input_path must be a non-empty string")

    path = input_path.strip()

    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    lower = path.lower()
    if lower.endswith(".csv"):
        process_csv(path, mapping_csv_path=MAPPING_CSV, output_dir=OUTPUT_DIR)
    elif lower.endswith(".txt"):
        process_txt(path)
    else:
        raise ValueError("Input path must end with .csv or .txt")


# Example:
# run("MOVING_VCs_DATASET_FINAL_V2.csv")
# run("input.txt")

run("MOVING_VCs_DATASET_FINAL_V2.csv")