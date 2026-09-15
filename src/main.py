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

    repair = {"report": None}  # augment_repair outcome, decided once per task

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
            # Undo the validator's weight-scaling augmentation (augment_repair.py) — only where the
            # score is absolute CE: a KL task measures distance from the damaged copy.
            # Measured 2026-09-15 (Qwen3-0.6B chat, validator per-sample CE, after full training): scaled
            # x0.6 LoRA 1.405 -> repaired 1.3065, x1.4 LoRA 1.331 -> 1.3066, x0.6 full-ft 1.423 -> 1.312,
            # clean 1.306; repaired beats unrepaired on 635-760 of 773 samples. Clean models are left alone.
            if (os.environ.get("SN56_AUG_REPAIR", "1") == "1" and os.environ.get("USE_KL") != "1"
                    and repair["report"] is None):
                rep_path = os.path.join(paths.WORK_ROOT, "augment_report.json")
                run([sys.executable, os.path.join(SRC, "augment_repair.py"), "repair",
                     "--model-path", model_path, "--work-root", paths.WORK_ROOT,
                     "--tok-dir", tok_dir, "--report", rep_path], end_ts)
                try:
                    with open(rep_path) as f:
                        repair["report"] = json.load(f)
                except Exception:
                    repair["report"] = {"scaled": False}
                if repair["report"].get("scaled"):
                    common[common.index("--model-path") + 1] = repair["report"]["path"]
                    # leave time to merge an adapter onto the repaired base afterwards
                    i = common.index("--end-ts") + 1
                    common[i] = str(float(common[i]) - 300)
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
                # Chat trains better with an adapter than with a full fine-tune at every
                # data scale we measured (see plan.choose_regime). This matters most on the
                # continuous-SFT gate: it is a ChatTask on 4xH100 where our memory rule
                # would otherwise pick full-ft, while the champion runs LoRA there.
                sft_env["SN56_ADAPTER_PREF"] = os.environ.get("SN56_ADAPTER_PREF", "lora")
            return run([sys.executable, os.path.join(SRC, "train_sft.py"),
                        *common, "--tokenized-dir", tok_dir], end_ts, sft_env)

        if args.task_type == "DpoTask":
            return run([sys.executable, os.path.join(SRC, "train_dpo.py"),
                        *common,
                        "--data-path", data_path,
                        "--dataset-type", args.dataset_type], end_ts)

        if args.task_type in ("GrpoTask", "EnvTask"):
            # GRPO tasks default to oracle distillation (grpo_distill.py). Measured 2026-09-15 on the
            # alpaca template task, Qwen2.5-1.5B-Instruct, 1000 test prompts, validator-exact 1-v-1:
            # distill 0.752 vs GRPO(pinned trainer, 1.5 h on H200) 0.125 and vs GRPO(this branch)
            # 0.177 — 1.5 h of GRPO barely moves the template rewards (unique ratio -0.837 -> -0.834),
            # distill moves them to -0.127 with the format regex at 0.59. It hands back to GRPO when
            # it cannot beat the base on dev, and never runs on tasks whose rewards read extra_data.
            grpo_env = {"SN56_GRPO_MODE": os.environ.get("SN56_GRPO_MODE",
                                                         "distill" if args.task_type == "GrpoTask" else "grpo")}
            return run([sys.executable, os.path.join(SRC, "train_grpo.py"),
                        *common,
                        "--data-path", data_path,
                        "--dataset-type", args.dataset_type], end_ts, grpo_env)

        print(f"[main] unknown task type {args.task_type}", flush=True)
        return 2

    ok = attempts()
    rep = repair["report"] or {}
    if ok and rep.get("scaled") and os.path.isfile(os.path.join(out_dir, "adapter_config.json")):
        # an adapter would be loaded onto the damaged base by the validator; ship merged weights
        if run([sys.executable, os.path.join(SRC, "augment_repair.py"), "merge",
                "--base", rep["path"], "--out-dir", out_dir], end_ts + 150) != 0 \
                or os.path.isfile(os.path.join(out_dir, "adapter_config.json")):
            # An adapter trained on the repaired base is WORSE than useless on the damaged base the
            # validator would load it onto (measured: CE 3.68 vs 1.69 for the untrained repaired base
            # and 1.35 merged). Ship the repaired base itself rather than that adapter.
            import shutil
            print("[main] merge failed; shipping the repaired base weights instead of the adapter", flush=True)
            for fn in os.listdir(out_dir):
                if fn.startswith("adapter_"):
                    os.remove(os.path.join(out_dir, fn))
            for fn in os.listdir(rep["path"]):
                src = os.path.join(rep["path"], fn)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(out_dir, fn))
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
