"""
distill2/loss.py

蒸餾 loss（乾淨自包含版，不依賴舊 training/）：
  - depth：teacher-conf 加權 MSE
  - camera：translation L1 + rotation(1-|dot| 絕對+相對, scalar-last) + FoV L1
            rotation 與 (T+FoV) 各自獨立 EMA 正規化（避免 rotation 被稀釋）
四元數為 VGGT 慣例 scalar-last (x,y,z,w)，實部在 index[3]。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _quat_mul(a, b):  # scalar-last (x,y,z,w)
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], dim=-1)


def _quat_conj(q):
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)


def camera_losses(pose_enc_list, target_pose, gamma=0.6):
    """回傳 (loss_rot, loss_transfov, loss_cam_combined.detach())。"""
    n = len(pose_enc_list)
    tot_rot, tot_tf = 0.0, 0.0
    tgt = target_pose.float()
    for s, pred in enumerate(pose_enc_list):
        w = gamma ** (n - s - 1)
        pred = pred.float()
        loss_T = (pred[..., :3] - tgt[..., :3]).abs().mean()
        loss_FL = (pred[..., 7:] - tgt[..., 7:]).abs().mean()

        pq = F.normalize(pred[..., 3:7], dim=-1)
        tq = F.normalize(tgt[..., 3:7], dim=-1)
        dot_abs = (pq * tq).sum(-1).abs().clamp(0, 1)
        loss_R = (1.0 - dot_abs).mean()

        if pq.shape[1] > 1:
            p0, p1 = pq[:, :-1], pq[:, 1:]
            t0, t1 = tq[:, :-1], tq[:, 1:]
            pr = F.normalize(_quat_mul(_quat_conj(p0), p1), dim=-1)
            tr = F.normalize(_quat_mul(_quat_conj(t0), t1), dim=-1)
            dot_rel = (pr * tr).sum(-1).abs().clamp(0, 1)
            loss_R = loss_R + (1.0 - dot_rel).mean()

        tot_rot = tot_rot + loss_R * w
        tot_tf = tot_tf + (loss_T + loss_FL) * w
    rot = tot_rot / n
    tf = tot_tf / n
    return rot, tf, (rot + tf).detach()


def depth_loss(pred_depth, pred_conf, tgt_depth, tgt_conf, mask):
    """teacher-conf 加權 MSE。pred_depth:[B,S,H,W,1], tgt_depth:[B,S,H,W]"""
    if mask.sum() < 100:
        return (pred_depth * 0).mean()
    pd = pred_depth.float().squeeze(-1)[mask]
    td = tgt_depth.float()[mask]
    w = tgt_conf.float()[mask].clamp(min=0.0)
    mse = ((pd - td) ** 2) * w
    return mse.mean()


class DistillLoss(nn.Module):
    def __init__(self, rot_weight=4.0, cam_weight=2.0, depth_weight=1.0,
                 momentum=0.99, warmup=100, gamma=0.6):
        super().__init__()
        self.rot_w, self.cam_w, self.depth_w = rot_weight, cam_weight, depth_weight
        self.momentum, self.warmup, self.gamma = momentum, warmup, gamma
        self.register_buffer("_ema_rot", torch.tensor(1.0))
        self.register_buffer("_ema_tf", torch.tensor(1.0))
        self.register_buffer("_ema_depth", torch.tensor(1.0))
        self.register_buffer("_step", torch.tensor(0))

    def _norm(self, loss, ema):
        v = loss.detach().abs().clamp(min=1e-8)
        ema.mul_(self.momentum).add_(v * (1 - self.momentum))
        if self._step < self.warmup:
            return loss
        return loss / (ema + 1e-8)

    def forward(self, pred, batch):
        dev = batch["pose_enc"].device
        if self._ema_rot.device != dev:
            for b in ["_ema_rot", "_ema_tf", "_ema_depth", "_step"]:
                setattr(self, b, getattr(self, b).to(dev))

        logs = {}
        total = 0.0

        rot, tf, cam_comb = camera_losses(pred["pose_enc_list"], batch["pose_enc"], self.gamma)
        total = total + self._norm(rot, self._ema_rot) * self.rot_w
        total = total + self._norm(tf, self._ema_tf) * self.cam_w
        logs["loss_rot"] = rot.detach()
        logs["loss_transfov"] = tf.detach()
        logs["loss_camera"] = cam_comb

        d = depth_loss(pred["depth"], pred.get("depth_conf"),
                       batch["depth"], batch["depth_conf"], batch["point_mask"])
        total = total + self._norm(d, self._ema_depth) * self.depth_w
        logs["loss_depth"] = d.detach()

        self._step += 1
        logs["objective"] = total
        return total, logs
