"""Build a side-by-side demo video: SeqFT vs CIVD (λ = 0.1) after one CL stage, same initial states.

Usage: python make_demo.py [--results ../results] [--out demo/civd_demo.mp4]

It reuses the evaluation rollouts recorded by the 03_cl* notebooks. The evaluation seed is 1000 for every run, so
episode k of a task starts from the same initial state for every method, and the two panels are directly comparable.
Needs opencv-python, Pillow and ffmpeg (for H.264 output).
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30          # rollouts are 20 Hz control; 30 fps playback = 1.5x real time
PANEL = 560
BG, INK, MUTED = (246, 247, 249), (27, 36, 48), (91, 102, 118)
GOOD, BAD, SEQ, CIVD = (31, 122, 68), (179, 38, 30), (235, 104, 52), (29, 95, 180)
FONT_DIR = Path("C:/Windows/Fonts") if Path("C:/Windows/Fonts").exists() else Path("/usr/share/fonts/truetype/dejavu")


def font(size, bold=False):
    for name in (["segoeuib.ttf"] if bold else ["segoeui.ttf"]) + ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "arial.ttf"]:
        if (FONT_DIR / name).exists():
            return ImageFont.truetype(str(FONT_DIR / name), size)
    return ImageFont.load_default()


def symbol_font(size):
    p = FONT_DIR / "seguisym.ttf"
    return ImageFont.truetype(str(p), size) if p.exists() else font(size, True)


# (task id, instruction, episode, caption). Episodes picked where the outcomes differ (old tasks) or agree (new task).
CLIPS = [
    (8, "put the bowl on the plate", 0,
     "Old task. SeqFT carries the bowl to the plate but never releases it: forgetting is a motor skill, not the goal."),
    (6, "put the cream cheese in the bowl", 1,
     "Old task. After fine-tuning on the new task, SeqFT no longer completes the placement; CIVD still does."),
    (2, "put the wine bottle on top of the cabinet", 2,
     "Old task. CIVD stored no old demonstrations, only the 9 old instruction strings."),
    (3, "open the top drawer and put the bowl inside", 1,
     "New task (learned in this stage). Both methods learn it: CIVD keeps the new-task gain."),
]


def read_video(p):
    v = cv2.VideoCapture(str(p))
    frames = []
    while True:
        ok, f = v.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return frames


def success(run_dir, name, ep):
    info = json.load(open(run_dir / "cl" / "eval" / name / "eval_info.json"))
    return bool(info["per_task"][0]["metrics"]["successes"][ep])


def card(lines):
    """lines: (text or [(x, text), ...] columns, size, bold, color)"""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    y = 150
    for text, size, bold, color in lines:
        f = font(size, bold)
        for x, t in (text if isinstance(text, list) else [(90, text)]):
            d.text((x, y), t, font=f, fill=color)
        y += int(size * 1.55)
    return np.array(im)


def frame(task, instr, ep, left, right, left_done, right_done, caption):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((40, 16), f"Task {task}: \u201c{instr}\u201d", font=font(30, True), fill=INK)
    d.text((40, 56), f"same initial state (episode {ep}, eval seed 1000) · 1.5× speed", font=font(18), fill=MUTED)
    for x0, img, label, col, done in ((40, left, "SeqFT (fine-tune)", SEQ, left_done),
                                      (680, right, "CIVD, λ = 0.1 (mine)", CIVD, right_done)):
        d.rectangle([x0, 92, x0 + PANEL, 96], fill=col)
        d.text((x0, 100), label, font=font(22, True), fill=col)
        im.paste(Image.fromarray(img).resize((PANEL - 40, PANEL - 40), Image.BILINEAR), (x0 + 20, 132))   # keep the square aspect
        if done is not None:
            ok, msg = done
            badge = ("\u2714  success" if ok else "\u2718  failed (time limit)")
            bc = GOOD if ok else BAD
            d.rounded_rectangle([x0 + 12, 144, x0 + 262, 190], 10, fill=bc)
            d.text((x0 + 26, 150), badge, font=symbol_font(24), fill=(255, 255, 255))
    d.text((40, 662), caption, font=font(19), fill=INK)
    return np.array(im)


def clip(results, task, instr, ep, caption, out):
    seq_dir, civd_dir = results / "cl_run2_kaggle", results / "cl_lambda_kaggle"
    a = read_video(seq_dir / f"cl/eval/m0_task{task}/videos/libero_goal_{task}/eval_episode_{ep}.mp4")
    b = read_video(civd_dir / f"cl/eval/m3l01_task{task}/videos/libero_goal_{task}/eval_episode_{ep}.mp4")
    sa, sb = success(seq_dir, f"m0_task{task}", ep), success(civd_dir, f"m3l01_task{task}", ep)
    n = max(len(a), len(b)) + FPS            # hold the final frame for one extra second
    for k in range(n):
        fa, fb = a[min(k, len(a) - 1)], b[min(k, len(b) - 1)]
        da = (sa, "") if k >= len(a) - 1 else None
        db = (sb, "") if k >= len(b) - 1 else None
        out.write(cv2.cvtColor(frame(task, instr, ep, fa, fb, da, db, caption), cv2.COLOR_RGB2BGR))
    return sa, sb


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--results", default=str(root.parent / "results"))
    ap.add_argument("--out", default=str(root.parent / "phase5_deliverables" / "demo" / "civd_demo.mp4"))
    a = ap.parse_args()
    results, out_path = Path(a.results), Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = out_path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    def hold(img, sec):
        for _ in range(int(sec * FPS)):
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    hold(card([("Continual learning without forgetting", 44, True, INK),
               ("SmolVLA (450M) on LIBERO-Goal · Lôi Đình Nhất · FFI26", 24, False, MUTED),
               ("", 12, False, INK),
               ("Stage 0: the model knows 10 tasks. Stage 1: it is fine-tuned on one new task (1200 steps).", 24, False, INK),
               ("Left: plain fine-tuning (SeqFT). Right: CIVD, which distils the previous model's velocity", 24, False, INK),
               ("field on new-task observations paired with OLD instructions, with no stored old demos.", 24, False, INK),
               ("Both panels start from the same initial state.", 24, True, INK)]), 6)
    for task, instr, ep, cap in CLIPS:
        sa, sb = clip(results, task, instr, ep, cap, vw)
        print(f"task {task} ep {ep}: SeqFT {'ok' if sa else 'fail'} | CIVD {'ok' if sb else 'fail'}")
    hold(card([("Over all 9 old tasks, 3 training seeds × 10 episodes", 34, True, INK),
               ("", 10, False, INK),
               ([(90, ""), (520, "old tasks"), (760, "new task (stage 0: 6 / 10)")], 22, False, MUTED),
               ([(90, "Stage 0"), (520, "91 %"), (760, "—")], 28, False, MUTED),
               ([(90, "SeqFT"), (520, "57 %"), (760, "24 / 30")], 28, False, SEQ),
               ([(90, "CIVD, λ = 0.1 (mine)"), (520, "88 %"), (760, "24 / 30")], 28, True, CIVD),
               ("", 10, False, INK),
               ("Forgetting 34.1 → 3.0 pp (Fisher p = 3·10⁻¹⁶); cost: 9 stored instruction strings, 1.8× training time.", 22, False, INK),
               ("Code: code/civd/civd.py · numbers: report §5.3", 20, False, MUTED)]), 7)
    vw.release()
    ff = shutil.which("ffmpeg")
    if ff:   # re-encode to H.264 so it plays everywhere and stays small
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "28", "-preset", "slow", "-movflags", "+faststart", str(out_path)], check=True)
        raw.unlink()
    else:
        raw.rename(out_path)
    print("wrote", out_path, f"{out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
