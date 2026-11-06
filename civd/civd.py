"""CIVD: Cross-Instruction Velocity Distillation for SmolVLA (LeRobot ba83e4c).

Stage-k training loss:
    L = L_FM(new-task batch) + lambda_d * E || v_student(x_t, t, o, l_old) - v_teacher(x_t, t, o, l_old) ||^2

- o: observations (images + state) of the CURRENT new-task batch, no old demos are stored.
- l_old: instructions of previously learned tasks (strings only).
- teacher: frozen copy of the previous-stage checkpoint.
- x_t = t * eps + (1 - t) * A_T, with A_T sampled by the teacher under l_old (NOISE_AT_ONE convention,
  same as VLAFlowMatching.forward), so the student is supervised on the old task's flow, not the new one.
- mode "new" (ablation M3b): distill with the batch's own (new) instruction instead of l_old.

Usage: this file is a drop-in launcher for lerobot_train. All CLI args are passed through to
LeRobot; CIVD is configured with environment variables:
    CIVD_TEACHER      path to the previous-stage `pretrained_model` dir (required)
    CIVD_OLD_TASKS    old instructions separated by "|" (required for mode "old")
    CIVD_LAMBDA       distillation weight (default 1.0)
    CIVD_DISTILL_BS   how many batch samples get the distillation term (default 8)
    CIVD_MODE         "old" (CIVD, default) or "new" (M3b ablation)
    CIVD_LOG_EVERY    print distill stats every N steps (default 100)
"""

import os
import random
import sys
import time

import torch
import torch.nn.functional as F

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def velocity(model, images, img_masks, lang_tokens, lang_masks, state, x_t, time):
    """v_theta(x_t, t | o, l). Same computation as VLAFlowMatching.forward, without the loss."""
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, time)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    (_, suffix_out), _ = model.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
    )
    suffix_out = suffix_out[:, -model.config.chunk_size :].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


class CIVD:
    def __init__(self, student: SmolVLAPolicy):
        cfg = os.environ
        self.mode = cfg.get("CIVD_MODE", "old")
        self.lam = float(cfg.get("CIVD_LAMBDA", "1.0"))
        self.distill_bs = int(cfg.get("CIVD_DISTILL_BS", "8"))
        self.log_every = int(cfg.get("CIVD_LOG_EVERY", "100"))
        self.device = next(student.parameters()).device

        self.teacher = SmolVLAPolicy.from_pretrained(cfg["CIVD_TEACHER"]).to(self.device).eval()
        self.teacher.requires_grad_(False)

        self.old_tokens = []
        if self.mode == "old":
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(student.config.vlm_model_name)
            for task in cfg["CIVD_OLD_TASKS"].split("|"):
                # Same text processing as the preprocessor: NewLineTaskProcessorStep + tokenizer.
                enc = tok(task.strip() + "\n", return_tensors="pt", truncation=True,
                          max_length=student.config.tokenizer_max_length)
                self.old_tokens.append(
                    (enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device).bool())
                )
        self.step = 0
        self.t_extra = 0.0
        print(f"[civd] mode={self.mode} lambda={self.lam} distill_bs={self.distill_bs} "
              f"old_tasks={len(self.old_tokens)} teacher={cfg['CIVD_TEACHER']}", flush=True)

    def loss(self, student: SmolVLAPolicy, batch):
        t0 = time.perf_counter()
        n = min(self.distill_bs, batch[OBS_LANGUAGE_TOKENS].shape[0])
        sub = {k: (v[:n] if torch.is_tensor(v) else v) for k, v in batch.items()}
        images, img_masks = student.prepare_images(sub)
        state = student.prepare_state(sub)

        if self.mode == "old":
            ids, mask = random.choice(self.old_tokens)
            lang_tokens, lang_masks = ids.expand(n, -1), mask.expand(n, -1)
        else:
            lang_tokens, lang_masks = sub[OBS_LANGUAGE_TOKENS], sub[OBS_LANGUAGE_ATTENTION_MASK]

        tm = self.teacher.model
        with torch.no_grad():
            a_teacher = tm.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
            eps = tm.sample_noise(a_teacher.shape, a_teacher.device)
            t = tm.sample_time(n, a_teacher.device)
            x_t = t[:, None, None] * eps + (1 - t[:, None, None]) * a_teacher
            v_teacher = velocity(tm, images, img_masks, lang_tokens, lang_masks, state, x_t, t)

        v_student = velocity(student.model, images, img_masks, lang_tokens, lang_masks, state, x_t, t)
        d = student.config.action_feature.shape[0]  # ignore padded action dims
        l_d = F.mse_loss(v_student[..., :d], v_teacher[..., :d])

        self.step += 1
        self.t_extra += time.perf_counter() - t0
        if self.step % self.log_every == 0:
            mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
            print(f"[civd] step:{self.step} distill:{l_d.item():.4f} "
                  f"extra_s:{self.t_extra / self.step:.3f} peak_mem_gb:{mem:.2f}", flush=True)
        return l_d


_orig_forward = SmolVLAPolicy.forward


def _civd_forward(self, batch, noise=None, time=None, reduction="mean"):
    loss, out = _orig_forward(self, batch, noise=noise, time=time, reduction=reduction)
    if not self.training:
        return loss, out
    if not hasattr(self, "_civd"):
        self._civd = CIVD(self)
    l_d = self._civd.loss(self, batch)
    out["civd_distill"] = l_d.item()
    return loss + self._civd.lam * l_d, out


def install():
    SmolVLAPolicy.forward = _civd_forward


if __name__ == "__main__":
    install()
    from lerobot.scripts import lerobot_train

    sys.argv[0] = "lerobot_train"
    lerobot_train.main()
