#!/usr/bin/env python3
"""convert_bin_safetensors.py <model_dir> - pytorch_model.bin -> model.safetensors 就地补一份.

transformers 5.x 在 torch<2.6 上拒绝 torch.load .bin 权重 (CVE-2025-32434),
只认 safetensors (同 Weibo21 链 chinese-roberta 的修复). 原 .bin 保留不动;
clone() 打断共享存储防 save_file 报错. 冒烟: AutoModel 能从该目录加载.
用法: python convert_bin_safetensors.py /home/hydro/code/models/<model>
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

d = Path(sys.argv[1])
assert (d / "pytorch_model.bin").exists(), f"missing {d/'pytorch_model.bin'}"
out = d / "model.safetensors"
if out.exists():
    print("model.safetensors already present, skip", flush=True)
    raise SystemExit(0)

sd = torch.load(d / "pytorch_model.bin", map_location="cpu", weights_only=True)
save_file({k: v.contiguous().clone() for k, v in sd.items()},
          str(out), metadata={"format": "pt"})
print(f"saved {len(sd)} tensors -> {out}", flush=True)

from transformers import AutoModel
m = AutoModel.from_pretrained(str(d))
print(f"smoke load ok: {type(m).__name__}", flush=True)
