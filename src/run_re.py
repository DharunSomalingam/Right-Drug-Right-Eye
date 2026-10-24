"""Stage 2: run one model over the preprocessed notes (relation extraction).

For every note it builds the prompt (zero-shot or few-shot), sends ONE request
with all of that note's tagged medications, parses the JSON answer and
appends the result to <outdir>/<model>/predictions.jsonl straight away
(flush + fsync), so a job that dies can be resumed: notes already in
predictions.jsonl are skipped on the next launch. One line per launch is
written to runs.jsonl (settings, prompt version, git commit, load time,
wall time, GPU).

Two backends:
  --backend hf      Hugging Face Transformers on a local GPU (fp16, greedy,
                    model chat template). This is the setup in Section 3.6.
                    gen_seconds is timed around model.generate() only, with
                    torch.cuda.synchronize() on both sides. The run aborts if
                    device_map="auto" puts any layer on CPU/disk.
  --backend ollama  any Ollama server (--host or $OLLAMA_HOST).
  --stub            no model at all; returns "Not Stated" for every drug.
                    For checking the pipeline on a CPU node.

Examples:
  python src/run_re.py --model-index 0 --input data/preprocessed.jsonl --outdir results/h1
  python src/run_re.py --model-index 0 --prompt grounded --input ... --outdir ...
  python src/run_re.py --model-index 0 --stub --input ... --outdir results/h1_dry
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_h1 import SYSTEM, parse_output  # noqa: E402
import prompt_h1  # noqa: E402
import prompt_h1_grounded  # noqa: E402

_TAGS = re.compile(r"</?Drug[^>]*>", re.I)

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# --model-index -> checkpoint. Order matches run_h1.sh (SLURM array 0-3).
# "hf" is the Hugging Face repo (Table 3.6); "ollama" is the equivalent tag
# used for quick runs against an Ollama server.
MODELS = [
    {"hf": "meta-llama/Llama-3.1-8B-Instruct", "ollama": "llama3.1:8b"},
    {"hf": "Qwen/Qwen2.5-7B-Instruct",         "ollama": "qwen2.5:7b"},
    {"hf": "Qwen/Qwen2.5-14B-Instruct",        "ollama": "qwen2.5:14b"},
    {"hf": "BaekSeungJu/Ophtimus-Instruct-8B", "ollama": "ophtimus_8b_q8_0:latest"},
]

NS = 1e9  # Ollama reports durations in nanoseconds

# Errors that mean "server unreachable": stop without checkpointing the note,
# so a resume retries it instead of scoring it as No output.
TRANSIENT = (ConnectionError, TimeoutError)
try:
    import httpx
    TRANSIENT += (httpx.TransportError,)
except ImportError:
    pass


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit():
    """Short commit hash of the code, with '+dirty' if there are local edits."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here,
                               capture_output=True, text=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def safe_name(name):
    """'meta-llama/Llama-3.1-8B-Instruct' -> 'Llama-3.1-8B-Instruct' (dir name)."""
    return re.sub(r"[^\w.-]+", "_", name.split("/")[-1])


# ------------------------------------------------------------ HF backend

# Llama-3 chat format, used only if a checkpoint ships without a chat template.
_LLAMA3 = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{s}<|eot_id|>"
           "<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
           "<|start_header_id|>assistant<|end_header_id|>\n\n")


def make_hf(name, max_new, rep_pen, seed):
    """Load a checkpoint in fp16 on the GPU and return (generate_fn, info)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        sys.exit("FATAL: no CUDA GPU visible -- refusing to run on CPU")
    torch.manual_seed(seed)

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(name)
    # fp16, not bf16: V100 (Volta) has no bfloat16. No quantisation, so model
    # size is the only thing that differs between models.
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    off = {k: v for k, v in (getattr(model, "hf_device_map", None) or {}).items()
           if v in ("cpu", "disk")}
    if off:
        sys.exit(f"FATAL: {len(off)} module(s) offloaded to CPU/disk {list(off)[:5]} "
                 f"-- timing would not be comparable. Use a bigger GPU.")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    eos = {tok.eos_token_id} | {tok.convert_tokens_to_ids(t)
                                for t in ("<|eot_id|>", "<|im_end|>")
                                if t in tok.get_vocab()}
    eos.discard(None)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def gen(system, user):
        if tok.chat_template:
            enc = tok.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True)
        else:
            enc = tok(_LLAMA3.format(s=system, u=user), return_tensors="pt",
                      add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        n_in = int(enc["input_ids"].shape[1])

        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 repetition_penalty=rep_pen,
                                 eos_token_id=list(eos), pad_token_id=pad)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t

        new = out[0, n_in:]
        n_out = int(new.shape[0])
        stopped = n_out > 0 and int(new[-1]) in eos
        return {"raw": tok.decode(new, skip_special_tokens=True).strip(),
                "input_tokens": n_in, "output_tokens": n_out,
                "gen_seconds": dt,
                "tokens_per_second": round(n_out / dt, 2) if dt else None,
                "done_reason": "stop" if stopped else "length",
                "truncated": not stopped and n_out >= max_new,
                "near_ctx_limit": False}

    info = {"gpu": torch.cuda.get_device_name(0), "load_seconds": round(load_s, 2),
            "torch": torch.__version__}

    def end_info():
        return {"peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}

    return gen, info, end_info


# -------------------------------------------------------- Ollama backend

def ollama_connect(host, name, pull, timeout):
    import ollama
    client = ollama.Client(host=host, timeout=timeout)
    try:
        client.show(name)
    except ollama.ResponseError as e:
        if e.status_code != 404:
            raise
        if not pull:
            sys.exit(f"model '{name}' not found on {host} -- "
                     f"run `ollama pull {name}` there, or pass --pull")
        print(f"pulling {name} on {host} ...", flush=True)
        client.pull(name)
    return client


def ollama_loaded_info(client, name):
    """VRAM use of the loaded model, and whether Ollama spilled it to CPU."""
    try:
        for m in client.ps()["models"]:
            mid = m.get("model") or m.get("name") or ""
            if mid == name or mid.split(":")[0] == name.split(":")[0]:
                size, vram = m.get("size") or 0, m.get("size_vram") or 0
                return {"vram_gb": round(vram / 1e9, 2),
                        "model_size_gb": round(size / 1e9, 2),
                        "offloaded": bool(size and vram < size)}
    except Exception as e:
        return {"ps_error": f"{type(e).__name__}: {e}"}
    return {}


def make_ollama(args, name):
    options = {"temperature": 0.0, "seed": args.seed,
               "num_predict": args.max_new_tokens, "num_ctx": args.num_ctx,
               "repeat_penalty": args.repetition_penalty}
    client = ollama_connect(args.host, name, args.pull, args.timeout)

    # empty prompt = load the model, so load time is not counted as note 1
    t0 = time.perf_counter()
    client.generate(model=name, prompt="", options=options, keep_alive=args.keep_alive)
    load_s = time.perf_counter() - t0

    def gen(system, user):
        t = time.perf_counter()
        r = client.chat(model=name,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                        options=options, keep_alive=args.keep_alive)
        dt = time.perf_counter() - t
        n_in = r.get("prompt_eval_count") or 0
        n_out = r.get("eval_count") or 0
        eval_s = (r.get("eval_duration") or 0) / NS
        reason = r.get("done_reason")
        return {"raw": (r["message"]["content"] or "").strip(),
                "input_tokens": int(n_in), "output_tokens": int(n_out),
                "gen_seconds": dt,
                "prompt_eval_seconds": (r.get("prompt_eval_duration") or 0) / NS,
                "eval_seconds": eval_s,
                "tokens_per_second": round(n_out / eval_s, 2) if eval_s else None,
                "done_reason": reason,
                "truncated": reason == "length" or n_out >= args.max_new_tokens,
                # Ollama silently cuts the input if it exceeds num_ctx
                "near_ctx_limit": n_in + n_out >= 0.98 * args.num_ctx}

    info = {"load_seconds": round(load_s, 2), "server": args.host,
            **ollama_loaded_info(client, name)}
    return gen, info, lambda: {"end_state": ollama_loaded_info(client, name)}


# ---------------------------------------------------------------- stub

def make_stub():
    """Fake model: answers Not Stated for every tagged id in the target note."""
    def gen(system, user):
        ids = re.findall(r'<Drug id="(D\d+)">', user.split("### Note:")[-1])
        time.sleep(0.01)
        raw = json.dumps([{"id": i, "laterality": "Not Stated", "evidence": ""}
                          for i in ids])
        return {"raw": raw, "input_tokens": len(system + user) // 4,
                "output_tokens": len(raw) // 4, "gen_seconds": 0.01,
                "done_reason": "stub", "truncated": False, "near_ctx_limit": False}
    return gen, {"load_seconds": 0.0, "server": "stub", "gpu": "stub"}, dict


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model-index", type=int, help="0-3, see MODELS")
    g.add_argument("--model-name", help="any HF repo id or Ollama tag")
    ap.add_argument("--input", required=True, help="preprocessed JSONL")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--backend", choices=["hf", "ollama"], default="hf")
    ap.add_argument("--prompt", choices=["zero", "grounded"], default="zero",
                    help="zero-shot, or few-shot with the five solved examples")
    ap.add_argument("--exclude-donors", action="store_true",
                    help="drop the few-shot donor notes (automatic for --prompt "
                         "grounded); use with --prompt zero so both conditions "
                         "see the same notes")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="1.0 = off. >1 penalises the JSON keys that repeat per drug")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="only the first N remaining notes")
    ap.add_argument("--stub", action="store_true", help="no model, CPU only")
    # Ollama-only options
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--num-ctx", type=int, default=8192,
                    help="Ollama's default context is small and silently truncates long notes")
    ap.add_argument("--keep-alive", default="30m")
    ap.add_argument("--timeout", type=float, default=600, help="per request, seconds")
    ap.add_argument("--pull", action="store_true", help="pull the Ollama model if missing")
    args = ap.parse_args()

    t_job = time.perf_counter()
    name = (MODELS[args.model_index][args.backend] if args.model_name is None
            else args.model_name)
    exclude = args.prompt == "grounded" or args.exclude_donors
    short = (safe_name(name)
             + ("_grounded" if args.prompt == "grounded" else
                "_zero_nodonors" if args.exclude_donors else "")
             + ("_STUB" if args.stub else ""))
    build_user = (prompt_h1_grounded.build_user if args.prompt == "grounded"
                  else prompt_h1.build_user)

    out = Path(args.outdir) / short
    out.mkdir(parents=True, exist_ok=True)
    pred_path, runs_path = out / "predictions.jsonl", out / "runs.jsonl"

    notes = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if exclude:
        notes = [n for n in notes if n["note_id"] not in prompt_h1_grounded.DONOR_NOTES]

    # resume: skip notes that already have a checkpointed prediction
    done = set()
    if pred_path.exists():
        for l in open(pred_path, encoding="utf-8"):
            try:
                done.add(json.loads(l)["note_id"])
            except (ValueError, KeyError):
                pass
    todo = [n for n in notes if n["note_id"] not in done][:args.limit]
    print(f"[{short}] notes={len(notes)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    run = {"model": name, "backend": "stub" if args.stub else args.backend,
           "started": now(), "host": socket.gethostname(), "stub": args.stub,
           "max_new_tokens": args.max_new_tokens,
           "repetition_penalty": args.repetition_penalty, "seed": args.seed,
           "prompt": args.prompt, "prompt_version": prompt_h1.PROMPT_VERSION,
           "code_commit": git_commit(), "exclude_donors": exclude}

    if args.stub:
        gen, info, end_info = make_stub()
    elif args.backend == "hf":
        gen, info, end_info = make_hf(name, args.max_new_tokens,
                                      args.repetition_penalty, args.seed)
    else:
        run["num_ctx"] = args.num_ctx
        gen, info, end_info = make_ollama(args, name)
    run.update(info)
    print(f"[{short}] loaded in {info['load_seconds']:.1f}s "
          f"{ {k: v for k, v in info.items() if k != 'load_seconds'} }", flush=True)

    t_inf, n_run = time.perf_counter(), 0
    try:
        with open(pred_path, "a", encoding="utf-8") as fh:
            for i, n in enumerate(todo, 1):
                ids = [d["id"] for d in n["drugs"]]
                user = build_user(n["tagged_text"])
                rec = {"note_id": n["note_id"], "model": name, "backend": run["backend"],
                       "prompt": args.prompt, "n_drugs": len(ids)}
                try:
                    g = gen(SYSTEM, user)
                    # evidence is checked against the note with the tags removed
                    preds, evid, errs = parse_output(g["raw"], ids,
                                                     _TAGS.sub("", n["tagged_text"]))
                    if g["truncated"]:
                        errs.append("hit max_new_tokens: output truncated")
                    if g["near_ctx_limit"]:
                        errs.append("near num_ctx: input may be truncated")
                    rec.update(g, preds=preds, evidence=evid, errors=errs)
                except TRANSIENT:
                    raise  # server down: don't checkpoint, a resume will retry
                except Exception as e:  # keep going; the note is scored as No output
                    import traceback
                    traceback.print_exc()
                    rec.update(raw=None, preds={}, evidence={},
                               errors=[f"{type(e).__name__}: {e}"], gen_seconds=None)

                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                n_run += 1
                print(f"[{short}] {i}/{len(todo)} {n['note_id']} "
                      f"{(rec.get('gen_seconds') or 0):.2f}s in={rec.get('input_tokens')} "
                      f"out={rec.get('output_tokens')} errors={len(rec['errors'])}",
                      flush=True)
    except TRANSIENT as e:
        run["aborted"] = f"{type(e).__name__}: {e}"
        print(f"[{short}] ABORT (server unreachable): {e}", flush=True)
    finally:
        run["inference_seconds"] = round(time.perf_counter() - t_inf, 2)
        run["job_seconds"] = round(time.perf_counter() - t_job, 2)
        run["notes_this_run"] = n_run
        run.update(end_info())
        run["finished"] = now()
        with open(runs_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(run) + "\n")
        print(f"[{short}] done: {json.dumps(run)}", flush=True)
    if "aborted" in run:
        sys.exit(1)


if __name__ == "__main__":
    main()
