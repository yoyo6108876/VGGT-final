"""
distill2/loss2.py

二次蒸餾用的 loss（response KD + camera-token 特徵式 KD，抗塌縮）。

【設計要點：約束要打在「原始」student camera token 上，不能被 projector 遮蔽】
  第一版把 proj(student token) 去對齊 teacher，結果可學的 projector 會把「塌縮的」
  student token 硬投影成相似 → loss 降了但 camera head 讀的原始 token 仍塌縮。
  這一版改成：
    (1) teacher 端投影：projector 把 teacher token 投到 student 空間，拿「原始」
        student token 逐幀對齊 → student backbone 本身被逼學，projector 擋不掉。
    (2) 跨幀結構對齊：student 各幀 token 的兩兩 cosine 矩陣要 match teacher 的
        （teacher 跨幀不同 → 逼 student 不能全塌；逐幀 projector 無法造假幀間關係）。

  - response KD：student 的 pose / depth vs 原始 1B soft labels（disk）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import camera_losses, depth_loss


def _pairwise_cos(x):
    """x:[B,S,D] → 跨幀兩兩 cosine [B,S,S]。"""
    xn = F.normalize(x, dim=-1)
    return torch.matmul(xn, xn.transpose(1, 2))


class DistillLoss2(nn.Module):
    def __init__(self, student_cam_dim, teacher_cam_dim,
                 rot_weight=4.0, cam_weight=2.0, depth_weight=1.0,
                 feat_weight=3.0, struct_weight=3.0,
                 momentum=0.99, warmup=100, gamma=0.6):
        super().__init__()
        # projector 放在 teacher 端：teacher_cam_dim → student_cam_dim（不會遮蔽 student）
        self.proj = nn.Linear(teacher_cam_dim, student_cam_dim)
        self.rot_w, self.cam_w, self.depth_w = rot_weight, cam_weight, depth_weight
        self.feat_w, self.struct_w = feat_weight, struct_weight
        self.momentum, self.warmup, self.gamma = momentum, warmup, gamma
        for n in ["_ema_rot", "_ema_tf", "_ema_depth", "_ema_feat", "_ema_struct"]:
            self.register_buffer(n, torch.tensor(1.0))
        self.register_buffer("_step", torch.tensor(0))

    def _norm(self, loss, ema):
        v = loss.detach().abs().clamp(min=1e-8)
        ema.mul_(self.momentum).add_(v * (1 - self.momentum))
        return loss if self._step < self.warmup else loss / (ema + 1e-8)

    def forward(self, pred, teacher_cam, batch):
        # ── response KD（disk 上的 1B soft labels）──
        rot, tf, cam_comb = camera_losses(pred["pose_enc_list"], batch["pose_enc"], self.gamma)
        d = depth_loss(pred["depth"], pred.get("depth_conf"),
                       batch["depth"], batch["depth_conf"], batch["point_mask"])

        # ── feature KD ──
        sc = pred["camera_token"].float()            # 原始 student token [B,S,Ds]
        tc = teacher_cam.detach().float()            # teacher token       [B,S,Dt]
        tc_in_s = self.proj(tc)                      # teacher → student 空間 [B,S,Ds]
        feat = (1.0 - F.cosine_similarity(sc, tc_in_s, dim=-1)).mean()   # 逐幀對齊（打在原始 sc）

        # 跨幀結構對齊（抗塌縮：student 幀間關係要像 teacher）
        struct = (_pairwise_cos(sc) - _pairwise_cos(tc)).pow(2).mean()

        total = (self._norm(rot, self._ema_rot) * self.rot_w
                 + self._norm(tf, self._ema_tf) * self.cam_w
                 + self._norm(d, self._ema_depth) * self.depth_w
                 + self._norm(feat, self._ema_feat) * self.feat_w
                 + self._norm(struct, self._ema_struct) * self.struct_w)

        self._step += 1
        logs = {"loss_rot": rot.detach(), "loss_transfov": tf.detach(),
                "loss_depth": d.detach(), "loss_feat": feat.detach(),
                "loss_struct": struct.detach(), "loss_camera": cam_comb, "objective": total}
        return total, logs
