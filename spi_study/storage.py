"""
Lưu trữ (protocol 6.7).

<root>/meta/<image>.json           id, lớp đích, D, f(x), f(x′), nguồn ảnh
<root>/meta/<image>_mask.pt        mask (và valid cho Pet)
<root>/attr/<image>/<spec>__L<level>__b<b>.pt   attribution float32 (3, H, W)
<root>/compute.jsonl               một dòng mỗi sub-estimate: K, N, n_backward, residual, thời gian, seed
<root>/metrics.jsonl               bảng dài: image, spec, level, b ('mean' = tham chiếu của mức), metric
<root>/ladder_ext.json             các mức thang được thêm do chưa hội tụ
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterable

import torch


def safe(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=+-]+", "_", spec)


class Store:
    def __init__(self, root: str):
        self.root = root

    # đường dẫn --------------------------------------------------------------
    def p(self, *parts) -> str:
        return os.path.join(self.root, *parts)

    def attr_path(self, image: str, spec: str, level: int, b: int) -> str:
        return self.p("attr", image, f"{safe(spec)}__L{int(level)}__b{int(b)}.pt")

    # attribution ------------------------------------------------------------
    def has_attr(self, image, spec, level, b) -> bool:
        return os.path.exists(self.attr_path(image, spec, level, b))

    def save_attr(self, image, spec, level, b, A: torch.Tensor, meta: dict) -> None:
        path = self.attr_path(image, spec, level, b)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        torch.save({"attr": A.detach().to(torch.float32).cpu(), **meta}, tmp)
        os.replace(tmp, path)

    def load_attr(self, image, spec, level, b, device="cpu") -> torch.Tensor:
        obj = torch.load(self.attr_path(image, spec, level, b), map_location=device)
        return obj["attr"]

    def load_level(self, image, spec, level, n_sub, device="cpu") -> tuple[torch.Tensor, list[torch.Tensor]]:
        """-> (trung bình của các sub-estimate, danh sách sub-estimate)."""
        subs = [self.load_attr(image, spec, level, b, device) for b in range(n_sub)]
        return torch.stack(subs).double().mean(dim=0).float(), subs

    def complete(self, images: Iterable[str], spec: str, level: int, n_sub: int) -> bool:
        return all(self.has_attr(im, spec, level, b) for im in images for b in range(n_sub))

    # meta -------------------------------------------------------------------
    def meta_path(self, image: str) -> str:
        return self.p("meta", f"{image}.json")

    def has_meta(self, image: str) -> bool:
        return os.path.exists(self.meta_path(image))

    def save_meta(self, image: str, meta: dict, mask: torch.Tensor, valid) -> None:
        os.makedirs(self.p("meta"), exist_ok=True)
        with open(self.meta_path(image), "w") as fh:
            json.dump(meta, fh, indent=1)
        torch.save({"mask": mask.bool().cpu(), "valid": None if valid is None else valid.bool().cpu()},
                   self.p("meta", f"{image}_mask.pt"))

    def load_meta(self, image: str) -> dict:
        with open(self.meta_path(image)) as fh:
            return json.load(fh)

    def load_mask(self, image: str):
        obj = torch.load(self.p("meta", f"{image}_mask.pt"))
        return obj["mask"], obj["valid"]

    # bảng JSONL -------------------------------------------------------------
    def append(self, table: str, row: dict) -> None:
        os.makedirs(self.root, exist_ok=True)
        with open(self.p(f"{table}.jsonl"), "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")

    def rows(self, table: str) -> list[dict]:
        path = self.p(f"{table}.jsonl")
        out = []
        if not os.path.exists(path):
            return out
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass                      # dòng cuối bị cắt khi dừng giữa chừng
        return out

    def metric_keys(self) -> set:
        return {(r["image"], r["spec"], r["level"], r["b"]) for r in self.rows("metrics")}

    # mở rộng thang ----------------------------------------------------------
    def extensions(self) -> dict:
        path = self.p("ladder_ext.json")
        if not os.path.exists(path):
            return {}
        with open(path) as fh:
            return json.load(fh)

    def add_extension(self, spec: str, level: tuple[int, int]) -> None:
        ext = self.extensions()
        ext.setdefault(spec, []).append([int(level[0]), int(level[1])])
        os.makedirs(self.root, exist_ok=True)
        with open(self.p("ladder_ext.json"), "w") as fh:
            json.dump(ext, fh, indent=1)

    # báo cáo ----------------------------------------------------------------
    def write_report(self, name: str, text: str, obj: dict | None = None) -> str:
        os.makedirs(self.root, exist_ok=True)
        path = self.p(f"{name}.md")
        with open(path, "w") as fh:
            fh.write(text)
        if obj is not None:
            with open(self.p(f"{name}.json"), "w") as fh:
                json.dump(obj, fh, indent=1, default=float)
        return path
