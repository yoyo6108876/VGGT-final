# -*- coding: utf-8 -*-
"""GFLOPs at S=4 (torch FlopCounterMode) for the two Table 6 rows not covered by
measure_budget_incl_context.py: the 8-block pruned student and the conv-512 scratch student.
FLOPs do not depend on weight values, so the scratch student is built with random init."""
import json

import torch
from torch.utils.flop_counter import FlopCounterMode

from eval import load_student
from vggt.models.vggt_student import VGGTStudent


@torch.no_grad()
def gflops(m):
    m.requires_grad_(False)
    x = torch.rand(1, 4, 3, 518, 518, device="cuda", dtype=torch.bfloat16)
    with FlopCounterMode(display=False) as fc, torch.autocast("cuda", dtype=torch.bfloat16):
        m(x)
    return round(fc.get_total_flops() / 1e9, 1)


out = {}
m = load_student(r"runs\prune\d8\best.pt", "cuda").eval()
out["pruned_d8"] = gflops(m)
del m
torch.cuda.empty_cache()
m = VGGTStudent(embed_dim=512, agg_depth=12, num_heads=8, patch_embed="conv", enable_track=False).cuda().eval()
out["scratch_conv512"] = gflops(m)
print(out)
json.dump(out, open(r"reports\flops_extra.json", "w"), indent=1)
