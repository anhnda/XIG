# attr_probe — Lấy mẫu đường SPI theo infidelity (intra-path & inter-path, protocol v1.0)

ResNet-50 `IMAGENET1K_V2`, logit lớp dự đoán, baseline đen, ImageNet-Segmentation (cùng tiền xử lý / chia ảnh như các đợt trước).
Cấu hình: `config.yaml` (mục `infid:`; `pathscore:` + `sampling:` cho đầu vào E0). Kết quả: `runs_probe/pilot/infid/`.
Audit P0: `docs/P0_AUDIT.md` — quy tắc tích phân của cache PATHSCORE là `canon_rand`, không phải điểm trái.
Sửa đổi protocol 1 (sau E0 + E-quadrature): `docs/PROTOCOL_AMENDMENT_1.md` — `rand` M = 1, C3a/C3b/C3c thay yêu cầu hội tụ,
IG không chệch, chẩn đoán làm mượt khối, K chính = 4.
Sửa đổi protocol 2 (sau pilot): `docs/PROTOCOL_AMENDMENT_2.md` — δ₁ = 0,75, δ₃ = 2,4 (so với SPI baseline; IG là đối chứng riêng),
E1a 32 ảnh × m = 16, E1b vòng 1 α ∈ {0,1; 0,3; 1; 10}, E0 thêm độ nhạy theo D và chẩn đoán triệt tiêu, `nbr` giới hạn ở pilot.
Sửa đổi protocol 3: `docs/PROTOCOL_AMENDMENT_3.md` — RQ2 trước, H2 hoãn; so sánh α theo cặp ảnh (leave-one-out), bảng theo từng D
và đánh đổi xoá / chèn, tách L = E_D / C_D.

Paper (stochastic feature-wise path distributions): lý thuyết `docs/PATH_FAMILIES_THEORY.md` (định lý–chứng minh, giả thiết đánh mã),
ranh giới tuyên bố `docs/PAPER_CLAIMS_AND_LIMITATIONS.md`, dàn ý `docs/PAPER_OUTLINE.md`, completeness ở K hữu hạn `docs/COMPLETENESS_SLOPE.md`.

## Chạy

```bash
pip install -r requirements.txt
# sửa local: trong config.yaml (đường dẫn gtsegs_ijcv.mat, các file index cũ)
bash scripts/run.sh test                     # P0 + C3a/C3b: unit test C1, quy tắc tích phân, T10, D, sàn (CPU)
bash scripts/run.sh toy                      # E-toy nhỏ: oracle Phụ lục → infid/toy/TOY_REPORT.md
bash scripts/run.sh toy-large                # E-toy lớn d = 256 (T9, Cov(H), coverage CI tỷ số)
bash scripts/run.sh pathscore                # (nếu chưa có cache) bản đồ từng đường SPI → pilot/pathscore/
bash scripts/run.sh pathscore-infid-decomp   # E0: từ attr.pt, không gradient mới → infid/e0/INFID_E0_REPORT.md (G0)
bash scripts/run.sh quad-check               # E-quadrature / C3c (τ²_D, lặp lại xác định) → infid/quad/INFID_QUAD_REPORT.md
bash scripts/run.sh sampling-infid-pilot     # E1a pilot (cặp rev + indep nhỏ; C1c, C2, chọn m)
#   → điền infid.protocol.{K_main, delta1, delta2, delta3, rule, M} (delta0 nếu muốn quyết định G0), sửa e1a.m / n_images
bash scripts/run.sh infid-lock               # khoá protocol → infid/infid_prereg.json (các bước sau kiểm hash)
bash scripts/run.sh sampling-infid-e1a       # E1a chính: H1 (+ H2 khi bật infid.e1a.coupled sau khi C1 đạt)
bash scripts/run.sh sampling-infid-e1b-dev   # E1b dev: chọn cấu hình → infid/e1b_selection.json (không ghi đè)
bash scripts/run.sh sampling-infid-e1b-test  # E1b test: H3
bash scripts/run.sh sampling-infid-e2        # E2: nhiều D, metric ngoài, (tuỳ chọn) ViT
bash scripts/run.sh infid-posthoc            # kiểm toán sau kiểm định, không GPU → infid/INFID_POSTHOC_REPORT.md
```

Mỗi lệnh có bản `-debug` (vd. `sampling-infid-pilot-debug`) và lệnh `*-report` dựng lại báo cáo từ CSV. Bị ngắt thì chạy
lại cùng lệnh: phần đã xong (cùng hash) được bỏ qua.

## Cấu trúc

| file | nội dung |
|---|---|
| `probe/pathquad.py` | quy tắc tích phân `rand` (cũ) / `left` / `mid` / `theta` / `gl`; lưới lồng nhau cho E-quadrature |
| `probe/sampling.py` | lịch trình `spi`, `patch`, `pix`, `hier`; phép đảo `rev`; thiết kế pha `indep` / `fixed` / `global` / `nbr`; InsDel |
| `probe/infid_core.py` | D (xoá / chèn / đối xứng / phân số, nền đen / blur, ô vuông / superpixel), ΔF, phép chiếu, bộ ước lượng chỉ dùng cặp (T10), sign-flip, sàn L*_free / L*_comp |
| `probe/infid_stats.py` | bootstrap theo ảnh, quy tắc quyết định, G0, cỡ mẫu, chọn m, khoá protocol |
| `probe/toy_spi.py` | E-toy nhỏ / lớn, coverage |
| `probe/infid_e0.py` | E0 (`pathscore-infid-decomp`) |
| `probe/infid_quad.py` | E-quadrature (`quad-check`) |
| `probe/infid_posthoc.py` | kiểm toán sau kiểm định (H3b không kẹp log, E2 theo cặp 8 D, metric ngoài theo cặp, J(K) và K*, độ nhạy C_D post hoc) |
| `probe/sampling_infid.py` | E1a pilot / chính, E1b dev / test, E2, `infid-lock`, báo cáo quyết định H1–H4 |
| `probe/pathscore.py`, `probe/pathscore_metrics.py`, `probe/pathbank.py` | PATHSCORE (đầu vào E0): bản đồ từng đường SPI (khoá ngẫu nhiên của P4 giữ nguyên) và metric |
| `probe/ctx.py`, `config.py`, `data.py`, `model.py`, `metrics.py`, `paths.py`, `quad.py`, `report.py`, `plots.py` | nền: ngữ cảnh / dữ liệu / mô hình / sampler Dirichlet / gradient theo lô / bảng |
| `tests/test_infid.py`, `tests/test_pathscore.py` | unit test (CPU) |
