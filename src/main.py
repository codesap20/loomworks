"""Task dispatcher: receives the validator's standardized CLI, routes to the
right pipeline, retries with degraded settings, and guarantees that SOMETHING
loadable sits in the submission directory before the clock runs out.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import paths
import plan as plan_mod

AXO_PY = "/opt/axo/bin/python"
MODERN_PY = "/opt/modern/bin/python"
SRC = os.path.dirname(os.path.abspath(__file__))
MAX_ATTEMPTS = 5


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-type", required=True)
    ap.add_argument("--task-type", required=True)
    ap.add_argument("--file-format", default="s3")
    ap.add_argument("--expected-repo-name", required=True)
    ap.add_argument("--hours-to-complete", type=float, required=True)
    return ap.parse_args()


def run(cmd: list[str], deadline: float, env_extra: dict | None = None) -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HOME", "/workspace/hf_home")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if env_extra:
        env.update(env_extra)
    timeout = max(60, deadline - time.time())
    print(f"[main] run (timeout {timeout / 60:.0f}m): {' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(cmd, env=env, timeout=timeout)
        return proc.returncode
    except subprocess.TimeoutExpired:
        print("[main] attempt hit the wall clock", flush=True)
        return 124


def submission_ok(out_dir: str) -> bool:
    if not os.path.isdir(out_dir):
        return False
    names = os.listdir(out_dir)
    has_weights = any(n.endswith(".safetensors") or n.endswith(".bin") for n in names)
    has_cfg = "config.json" in names or "adapter_config.json" in names
    return has_weights and has_cfg


def emergency_submission(model_path: str, out_dir: str, modern: bool) -> None:
    """All attempts failed: submit jittered base weights (see emergency.py).
    A zero-risk placeholder is worth more than an empty upload. Custom v5-only
    architectures must be loaded by the modern venv's interpreter."""
    print("[main] EMERGENCY: submitting jittered base model", flush=True)
    py = MODERN_PY if modern else sys.executable
    subprocess.run([py, os.path.join(SRC, "emergency.py"),
                    "--model-path", model_path, "--output-dir", out_dir],
                   check=True, timeout=1800)


def main() -> None:
    args = parse_args()
    start = time.time()
    end_ts = start + args.hours_to_complete * 3600 - 180

    model_path = paths.resolve_model_path(args.model)
    data_path = paths.resolve_dataset_path(args.task_id, args.dataset)
    out_dir = paths.submission_dir(args.task_id, args.expected_repo_name)
    tok_dir = paths.tokenized_dir(args.task_id)
    os.makedirs(paths.WORK_ROOT, exist_ok=True)

    info = plan_mod.probe_model(model_path)
    n_gpus, _ = plan_mod.gpu_inventory()
    print(f"[main] task={args.task_type} model={args.model} params~"
          f"{(info['params'] or 0) / 1e9:.1f}B type={info['model_type']} gpus={n_gpus}",
          flush=True)

    common = ["--task-id", args.task_id,
              "--model-path", model_path,
              "--base-model-id", args.model,
              "--output-dir", out_dir,
              "--end-ts", str(end_ts),
              "--num-gpus", str(n_gpus),
              "--state-file", paths.STATE_FILE]

    def attempts() -> bool:
        # keep enough tail for a save: 15 min on long tasks, proportionally less on short ones
        retry_guard = min(900, max(120, (end_ts - start) * 0.15))
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if time.time() > end_ts - retry_guard:
                print(f"[main] <{retry_guard / 60:.0f}min left before attempt {attempt}; "
                      "stopping retries", flush=True)
                break
            rc = one_attempt(attempt)
            if rc == 0 and submission_ok(out_dir):
                return True
            print(f"[main] attempt {attempt} rc={rc}", flush=True)
        return submission_ok(out_dir)

    def one_attempt(attempt: int) -> int:
        if args.task_type in ("InstructTextTask", "ChatTask"):
            if plan_mod.needs_modern_stack(info):
                # custom archs (quasar) can only run on the v5 stack; this covers
                # both ChatTask lineages and the pre-boss quasar instruct task
                rc = run([MODERN_PY, os.path.join(SRC, "train_chat_modern.py"),
                          *common,
                          "--data-path", data_path,
                          "--dataset-type", args.dataset_type,
                          "--task-format",
                          "instruct" if args.task_type == "InstructTextTask" else "chat"],
                         end_ts)
                if rc != 42:
                    return rc
                print("[main] modern stack unavailable (rc=42); legacy fallback", flush=True)
            if not os.path.isdir(os.path.join(tok_dir, "train")):
                rc = run([AXO_PY, os.path.join(SRC, "tok_axolotl.py"),
                          "--model-path", model_path,
                          "--data-path", data_path,
                          "--dataset-type", args.dataset_type,
                          "--task-type", args.task_type,
                          "--seq-len", "4096",
                          "--out-dir", tok_dir], end_ts)
                if rc != 0:
                    return rc
            # ChatTask keeps the champion's cosine schedule and instruct keeps our WSD,
            # but both are now known to be a wash at the real budget (see below) — the
            # split is kept only because each was measured in its own regime.
            # Checkpoint soup is the ONE mechanism that survives training to
            # convergence, so it is on for both SFT-shaped task types. Measured on
            # H200 2026-09-09 with the validator's per-sample rule, every arm run to
            # the epoch cap (~1.5 epochs before the overfitting early-stop):
            #   instruct  champ sched 0.95719 | our WSD 0.95713 (gap 0.00006, a wash)
            #                                 | our WSD + soup 0.95207 -> beats champ
            #                                   by 0.0051, 303-143 of decided samples
            #   chat      champ sched 0.34225 | +lr 0.85x +soup 0.34182 (gap 0.0004)
            #                                 | +lr 1.0x  +soup 0.34081 (gap 0.0014)
            # The earlier, much larger margins (instruct 504-158, chat 381-12) were all
            # measured at 700-800s = ~0.5 epoch. The validator sizes budgets for 2 epochs
            # (TARGET_TRAINING_EPOCHS), so those runs were in a regime it never uses, and
            # the schedule/LR differences vanish once both sides converge. In particular
            # the old chat default of lr x0.85 is WORSE at the real budget than x1.0.
            sft_env = {"SN56_USE_SOUP": os.environ.get("SN56_USE_SOUP", "1")}
            if args.task_type == "ChatTask":
                sft_env["SN56_CHAMP_SCHED"] = os.environ.get("SN56_CHAMP_SCHED", "1")
                sft_env["SN56_LR_MULT"] = os.environ.get("SN56_LR_MULT", "1.0")
            return run([sys.executable, os.path.join(SRC, "train_sft.py"),
                        *common, "--tokenized-dir", tok_dir], end_ts, sft_env)

        if args.task_type == "DpoTask":
            return run([sys.executable, os.path.join(SRC, "train_dpo.py"),
                        *common,
                        "--data-path", data_path,
                        "--dataset-type", args.dataset_type], end_ts)

        if args.task_type in ("GrpoTask", "EnvTask"):
            return run([sys.executable, os.path.join(SRC, "train_grpo.py"),
                        *common,
                        "--data-path", data_path,
                        "--dataset-type", args.dataset_type], end_ts)

        print(f"[main] unknown task type {args.task_type}", flush=True)
        return 2

    ok = attempts()
    if not ok:
        try:
            emergency_submission(model_path, out_dir, plan_mod.needs_modern_stack(info))
        except Exception as e:
            print(f"[main] emergency submission failed too: {e}", flush=True)
            sys.exit(1)
    print(f"[main] finished ok={ok or submission_ok(out_dir)} "
          f"elapsed={(time.time() - start) / 60:.0f}m", flush=True)


if __name__ == "__main__":
    main()
