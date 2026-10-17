r"""
distill2/proj_init.py   —— 實驗性（④ 投影初始化縮寬度）

假設檢驗：conv-512 student 從零會塌縮（82°）。若改用「把已 work 的 Path B（1024）的
aggregator + camera head 用 PCA/SVD 投影進 512」來初始化，是否能避免塌縮？
若可 → 有機會 8.6× 壓縮又不塌；若否 → ④ 風險確認。

作法（近似）：
  取一個共用投影 P（512×1024，Path B aggregator「寫入 residual 的權重」之 SVD 左奇異向量前 512）。
  每個 transformer block 用 W' = P W Pᵀ 一致投影；MLP hidden 另用 per-block SVD 降維。
  student num_heads=8 → head_dim 64（與 teacher 相同）→ q_norm/k_norm 直接複製、RoPE 一致。
  ⚠ LayerNorm / LayerScale 的投影是近似（基底相依）；這是探路實驗，不是精確等價。

用法：
  python proj_init.py --teacher_ckpt runs/distill_b1/best.pt --out proj_student.pt
  python train.py --student_init proj_student.pt --data_root ... --out_dir runs/proj512 --epochs 8 --img_per_seq 2
"""

import argparse
import torch

from vggt.models.vggt_student import VGGTStudent
from vggt.layers.layer_scale import LayerScale
from eval import build_model


def topk_left_sv(W, k):
    """W:[out,in] → top-k 左奇異向量 [k,out]（正交列）。
    k 可能 > rank(=min(out,in))，此時用 full SVD 取正交補足（供 init 用）。"""
    Wf = W.float()
    U, _, _ = torch.linalg.svd(Wf, full_matrices=(k > min(Wf.shape)))
    return U[:, :k].T.contiguous()


def fill_lin(dst, src, Pout=None, Pin=None):
    W = src.weight.data.float()
    if Pin is not None:  W = W @ Pin.T
    if Pout is not None: W = Pout @ W
    assert W.shape == dst.weight.shape, f"{tuple(W.shape)} vs {tuple(dst.weight.shape)}"
    dst.weight.data.copy_(W)
    if getattr(src, "bias", None) is not None and getattr(dst, "bias", None) is not None:
        b = src.bias.data.float()
        if Pout is not None: b = Pout @ b
        dst.bias.data.copy_(b)


def fill_vec(dst_param, src_vec, P):
    dst_param.data.copy_(P @ src_vec.data.float() if P is not None else src_vec.data.float())


def fill_qkv(dst_qkv, src_qkv, P):
    d = P.shape[1]
    W = src_qkv.weight.data.float()           # [3d, d]
    parts = [P @ W[i*d:(i+1)*d] @ P.T for i in range(3)]  # 各 [d',d']
    dst_qkv.weight.data.copy_(torch.cat(parts, 0))
    if src_qkv.bias is not None:
        b = src_qkv.bias.data.float()
        dst_qkv.bias.data.copy_(torch.cat([P @ b[i*d:(i+1)*d] for i in range(3)], 0))


def project_block(dst, src, P):
    fill_vec(dst.norm1.weight, src.norm1.weight, P); fill_vec(dst.norm1.bias, src.norm1.bias, P)
    fill_vec(dst.norm2.weight, src.norm2.weight, P); fill_vec(dst.norm2.bias, src.norm2.bias, P)
    fill_qkv(dst.attn.qkv, src.attn.qkv, P)
    fill_lin(dst.attn.proj, src.attn.proj, P, P)
    # head_dim 相同 → q_norm/k_norm 直接複製
    for a in ["q_norm", "k_norm"]:
        s, t = getattr(src.attn, a), getattr(dst.attn, a)
        if hasattr(s, "weight") and hasattr(t, "weight"):
            t.weight.data.copy_(s.weight.data.float());
            if getattr(t, "bias", None) is not None and getattr(s, "bias", None) is not None:
                t.bias.data.copy_(s.bias.data.float())
    if isinstance(src.ls1, LayerScale) and isinstance(dst.ls1, LayerScale): fill_vec(dst.ls1.gamma, src.ls1.gamma, P)
    if isinstance(src.ls2, LayerScale) and isinstance(dst.ls2, LayerScale): fill_vec(dst.ls2.gamma, src.ls2.gamma, P)
    # mlp：hidden 也降維
    h2 = dst.mlp.fc1.weight.shape[0]
    Ph = topk_left_sv(src.mlp.fc1.weight, h2)   # [h2, h]
    fill_lin(dst.mlp.fc1, src.mlp.fc1, Ph, P)
    fill_lin(dst.mlp.fc2, src.mlp.fc2, P, Ph)


def compute_P(blocks, k):
    """由所有 block「寫入 residual 的權重」(attn.proj + mlp.fc2) 之 SVD 取前 k 主方向。"""
    cols = []
    for b in blocks:
        cols.append(b.attn.proj.weight.data.float())   # [d,d]
        cols.append(b.mlp.fc2.weight.data.float())      # [d,h]
    M = torch.cat(cols, dim=1)                          # [d, sum]
    U, _, _ = torch.linalg.svd(M, full_matrices=False)
    return U[:, :k].T.contiguous()                     # [k, d]


def project_aggregator(t_agg, s_agg, P):
    with torch.no_grad():
        # camera/register token: 投影最後一維（1024→512）
        s_agg.camera_token.data.copy_((t_agg.camera_token.data.float() @ P.T))
        s_agg.register_token.data.copy_((t_agg.register_token.data.float() @ P.T))
        for i in range(len(s_agg.frame_blocks)):
            project_block(s_agg.frame_blocks[i], t_agg.frame_blocks[i], P)
            project_block(s_agg.global_blocks[i], t_agg.global_blocks[i], P)


def project_camera_head(t_ch, s_ch, P2):
    with torch.no_grad():
        fill_vec(s_ch.token_norm.weight, t_ch.token_norm.weight, P2); fill_vec(s_ch.token_norm.bias, t_ch.token_norm.bias, P2)
        fill_vec(s_ch.trunk_norm.weight, t_ch.trunk_norm.weight, P2); fill_vec(s_ch.trunk_norm.bias, t_ch.trunk_norm.bias, P2)
        s_ch.empty_pose_tokens.data.copy_(t_ch.empty_pose_tokens.data.float())  # 9 維不變
        fill_lin(s_ch.embed_pose, t_ch.embed_pose, P2, None)                    # (9→dim) 出投影
        # poseLN_modulation = Sequential(SiLU, Linear(dim,3dim))
        blk = torch.block_diag(P2, P2, P2)
        fill_lin(s_ch.poseLN_modulation[1], t_ch.poseLN_modulation[1], blk, P2)
        for i in range(len(s_ch.trunk)):
            project_block(s_ch.trunk[i], t_ch.trunk[i], P2)
        # pose_branch = Mlp(dim, dim//2, 9)
        h2 = s_ch.pose_branch.fc1.weight.shape[0]
        Ph = topk_left_sv(t_ch.pose_branch.fc1.weight, h2)
        fill_lin(s_ch.pose_branch.fc1, t_ch.pose_branch.fc1, Ph, P2)
        fill_lin(s_ch.pose_branch.fc2, t_ch.pose_branch.fc2, None, Ph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", required=True, help="Path B best.pt（1024，投影來源）")
    ap.add_argument("--out", default="proj_student.pt")
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--num_heads", type=int, default=8, help="8→head_dim 64（與 teacher 相同）")
    args = ap.parse_args()

    tck = torch.load(args.teacher_ckpt, map_location="cpu")
    teacher = build_model(tck.get("meta", {}), "cpu")
    teacher.load_state_dict(tck.get("model", tck), strict=False)
    teacher.eval()
    d = tck.get("meta", {}).get("embed_dim", 1024)
    print(f"投影來源 Path B：embed_dim={d}")

    student = VGGTStudent(img_size=518, patch_size=14, embed_dim=args.embed_dim, agg_depth=12,
                          num_heads=args.num_heads, patch_embed="conv",
                          enable_camera=True, enable_depth=True, enable_point=True, enable_track=False)

    P = compute_P(list(teacher.aggregator.frame_blocks) + list(teacher.aggregator.global_blocks), args.embed_dim)
    print(f"aggregator 投影 P: {tuple(P.shape)}（正交列? {torch.allclose(P@P.T, torch.eye(args.embed_dim), atol=1e-4)}）")
    project_aggregator(teacher.aggregator, student.aggregator, P)
    print("✓ aggregator 已投影")

    try:
        dim_in = 2 * args.embed_dim
        P2 = compute_P(list(teacher.camera_head.trunk), dim_in)  # 2048→1024
        project_camera_head(teacher.camera_head, student.camera_head, P2)
        print(f"✓ camera_head 已投影（P2 {tuple(P2.shape)}）")
    except Exception as e:
        print(f"⚠ camera_head 投影失敗（保留隨機初始化）：{type(e).__name__}: {e}")

    meta = {"embed_dim": args.embed_dim, "agg_depth": 12, "num_heads": args.num_heads,
            "patch_embed": "conv", "init": "proj_from_pathb"}
    torch.save({"model": student.state_dict(), "meta": meta}, args.out)
    n = sum(p.numel() for p in student.parameters())
    print(f"\n✓ 已存 {args.out}｜conv-512 student {n/1e6:.1f}M（約 8.x× 壓縮，若不塌縮）")
    print("下一步：python train.py --student_init " + args.out + " --data_root ... --out_dir runs/proj512 --epochs 8 --img_per_seq 2")


if __name__ == "__main__":
    main()
