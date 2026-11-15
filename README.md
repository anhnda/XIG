# spi_study — SPI và tích phân gradient trên phân phối trạng thái

Code cho protocol chốt ngày 15/11/2026. Viết bằng PyTorch, tham khảo `pathattr/` và
`experiments/` của code cũ (dữ liệu, insdelp, ROAD, power từ JSONL cũ).

**Chưa chạy lần nào.** Code được viết và rà soát bằng cách đọc. Lần chạy đầu tiên phải bắt đầu
bằng `bash scripts/run.sh test`. Sau đó chạy `smoke` trên 2 ảnh trước khi chạy thật.

## Lệch khỏi protocol (đọc trước)

| # | Protocol | Bộ code | Lý do |
|---|---|---|---|
| 1 | 6.2: SPI_DP và COORD_MIX lấy gradient tại trung điểm bước | Mặc định `node=rand`: điểm đánh giá Z_{k−1} + V_k·δ_k, V_k ~ U(0,1) dùng chung cho mọi phần tử trong bước. `node=mid` vẫn có | Trung điểm chệch ở các bước nhảy có kích thước O(1), và độ chệch **không giảm khi N tăng**. Atom của DP và phần tử kiểu Shapley của COORD_MIX luôn nhảy cỡ O(1). Khi đó thang hội tụ "hội tụ" về một giá trị sai, mà đó đúng là thứ Cửa 1 muốn loại. Với U5, COORD_MIX λ=½ cho A₁ ≈ −0.375 thay vì −0.25; SPI_DP α=1 cho ≈ −0.333. U6 (ReLU) cũng sai. `node=rand` không chệch với A_{α,N}, và A_{α,N} → A_α. Test `test_mid_node_is_biased` kiểm đúng độ chệch lý thuyết |
| 2 | 6.2: SPI_DP 8×24 trong bảng cùng ngân sách | Có cả `node=mid` (đúng báo cáo gốc) và `node=rand` | So trực tiếp với báo cáo gốc |
| 3 | 6.1: float32 | Tắt thêm TF32 (`model.allow_tf32: false`) | Trên A100, PyTorch mặc định bật TF32 cho conv; khi đó tham chiếu không còn là float32 thật |
| 4 | 6.3: thang | Mỗi mức đều chia thành 4 sub-estimate, không chỉ mức cao nhất | Cùng chi phí, nhưng mức nào cũng có sai số chuẩn |
| 5 | 6.6: "SE" trong \|Q_f − Q_g\| < max(2·SE, 0.01) | SE ghép cặp qua ảnh, tức sd(ΔQ)/√n | Protocol không nói rõ dùng SE nào |
| 6 | Seed theo (ảnh, phương pháp, tham số, mức, b) | Thêm K, N vào khoá hash | Mức được thêm khi mở rộng thang luôn có seed riêng |

Nếu muốn theo đúng 6.2: đổi `node=rand` thành `node=mid` trong `config.yaml` trước khi khoá. Khi đó
U5 và U6 sẽ không qua, vì các test này kiểm theo đáp án population.

## Cài đặt

```bash
pip install -r requirements.txt
```

Dữ liệu: `gtsegs_ijcv.mat` chứa cả ảnh lẫn mask
(http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat). Oxford-IIIT Pet được
torchvision tự tải vào `local.pet_root`.

Sửa mục `local:` trong `config.yaml`. Mục này không được băm: `imagenet_mat`, `old_index_file`
(`runs/seg200.json` cũ, nếu có), `old_jsonl` (ví dụ `runs/axis_resnet50_black.jsonl` cũ).

## Chạy

```bash
# Giai đoạn 0
bash scripts/run.sh test          # U1–U9 + test đường ống, CPU, vài phút. Phải đạt hết.
bash scripts/run.sh test_slow     # thêm U4 phân biệt 251/720 vs 252/720
#  → sửa model.weights (V1/V2) theo báo cáo gốc
bash scripts/run.sh power         # → runs/imagenet/power.md; ghi n vào data.imagenet.n_images
bash scripts/run.sh lock          # khoá config.yaml → config.yaml.lock
bash scripts/run.sh index         # n ảnh mới, không thuộc 200 ảnh cũ → runs/imagenet/index.json
bash scripts/run.sh smoke         # 2 ảnh, ghi vào runs_smoke/: kiểm residual, thời gian, bộ nhớ

# Giai đoạn 1A — Cửa 1
bash scripts/run.sh s1a           # → runs/imagenet/convergence_stage1a.md, gate1.md
bash scripts/run.sh s1a_bq        # tuỳ chọn, chậm

# Giai đoạn 1B — chỉ khi gate1 = POSITIVE
bash scripts/run.sh s1b           # → runs/imagenet/convergence_stage1b.md, collapse.md

# Pet
bash scripts/run.sh pet_train     # SEEDS="0 1 2"
bash scripts/run.sh pet_index
bash scripts/run.sh pet_models    # → runs/pet/models.md
PET_SEED=0 bash scripts/run.sh pet_s1a   # Cửa 1 trên Pet (nếu muốn biết kết quả có chuyển sang Pet không)
bash scripts/run.sh s2            # Giai đoạn 2 → runs/pet/ensemble.md
```

Mọi lệnh đều chạy tiếp được: chạy lại cùng một lệnh sẽ bỏ qua attribution và metric đã có. Biến
môi trường: `OUT=...` thay `local.out`, `EXTRA="--device cuda:1"`.

Lệnh lẻ: `python -m spi_study --config config.yaml <lệnh> --help`. Các lệnh gồm `lock`, `index`,
`power`, `attr`, `converge`, `eval`, `gate1`, `collapse`, `pet-train`, `pet-models`, `ensemble`.

## Đường đi của Cửa 1 (`s1a`)

1. `attr --group stage1a`: tính IG_mid (N = 24, 192), các thang IG_shift và SPI_DP (α = 10, 100,
   `node=rand`), cộng SPI_DP 8×24 với `node=mid`. Mỗi mức gồm 4 sub-estimate. Mỗi sub-estimate lưu
   một file `.pt` float32 kèm một dòng trong `compute.jsonl` (backward, residual, thời gian, seed).
2. `converge --extend`: áp tiêu chí 6.3 giữa hai mức cao nhất. Nếu không đạt, thêm một mức
   (nhân đôi K và N; với α = 1 là 128×192) vào `ladder_ext.json`, trả mã 3. Script khi đó chạy lại
   `attr`. Mỗi spec được mở rộng tối đa `convergence.max_extra_levels` lần.
3. `eval`: tính IoU (phá hoà bằng hoán vị cố định theo ảnh), insdelp_blur, ROAD trên tham chiếu của
   mỗi mức (`b = "mean"`) và trên từng sub-estimate. Kết quả ghi vào `metrics.jsonl`, bảng dạng dài.
4. `gate1`: tính d_j = IoU(SPI ref) − IoU(IG ref), BCa 97.5% với 10.000 lần bootstrap, ba điều kiện
   (i)–(iii), Wilcoxon một phía có Holm, power đạt được từ SD_d thật. Quyết định là một trong
   **POSITIVE / NEGATIVE / UNDERPOWERED**. Báo cáo kèm metric phụ và bảng cùng ngân sách (riêng).

Tham chiếu là mức cao nhất đã đủ cho mọi ảnh. Nếu một spec chưa hội tụ sau khi đã mở rộng, gate1
vẫn chạy nhưng in cảnh báo.

## Data collapse (`s1b`)

- Ba họ cùng trung bình và phương sai tại mọi t, với h = 1/(α+1):
  - `spi_dp:alpha=α,node=rand`
  - `coord_mix:lam=h,node=rand`
  - `line:s=2h`, tính bằng (1 − s)·A_IG + s·A_SPI(1) từ các tham chiếu, ghép sub-estimate theo b
- `collapse` báo cáo:
  - CI_fg(h) và ‖ΔA‖/‖A_IG‖, trung vị qua ảnh
  - Q_f(h) = IoU, ΔQ kèm SE và CI 95%
  - S_ε (ε = 0.05) tính Monte Carlo từ chính sampler
  - Kết luận **COLLAPSE / SPLIT / INCONCLUSIVE**, theo tiêu chí 6.6 tại h ∈ {1/11, 1/31, 1/101}

## Giai đoạn 2 (Pet)

- `pet-train`: ResNet-18, chỉ khác seed.
  - `pet.init: imagenet` (mặc định): tinh chỉnh từ cùng weights ImageNet. Seed đổi head, thứ tự dữ liệu và augmentation.
  - `pet.init: scratch`: huấn luyện từ đầu.
  - Phải chốt một trong hai trước khi khoá.
- `pet-models`: accuracy, đồng thuận dự đoán theo cặp, ECE và D_m trên 50 ảnh đã chọn.
- `s2`: chạy thang cho cả ba họ, với lớp đích = nhãn thật. Kết quả của mỗi seed nằm trong `runs/pet_s<m>_label/`.
  IoU trên trimap loại vùng biên khỏi ứng viên, tử số và mẫu số.
- `ensemble`:
  - V_model(h) trên Ã = A/D_m, so với trung bình bỏ-một-mô-hình, đã trừ phương sai estimator (từ sub-estimate)
  - cosine theo cặp
  - hai nghiệm tầm thường: cực tiểu tại IG; beta không tốt hơn LINE
  - phân rã V1
  - phép thử faithfulness của phần bị loại: xoá theo |R_m|, so với xoá ngẫu nhiên, trên chính m và trên các mô hình khác

Giai đoạn 3 (10 seed, ensemble cục bộ, ViT/Swin) **chưa cài**, vì thiết kế của nó phụ thuộc kết quả
Cổng 3.

## Cấu trúc

```
config.yaml              mọi tham số và ngưỡng; mục local: không băm
scripts/run.sh           chạy theo giai đoạn
spi_study/
  config.py              đọc / khoá / kiểm hash; spec "tên:k=v"
  seeding.py             seed = hash(bộ khoá); torch.Generator riêng mỗi bộ
  data.py                gtsegs_ijcv (giống loader cũ), Oxford Pet + trimap, danh sách ảnh mới
  model.py               ResNet-50/18, TF32, grad_F(Zb) = Δx ⊙ ∂f_c/∂x tại x′ + Zb⊙Δx
  samplers.py            ig_mid, ig_shift, spi_dp (Dirichlet log-space), spi_bq, coord_mix, line (S_ε)
  attribute.py           A = Σ grad_F(Z_eval) ⊙ dZ, trung bình K đường; residual
  reference.py           thang (K, N), sub-estimate, meta theo ảnh, kiểm hội tụ / mở rộng
  metrics.py             IoU (phá hoà, trimap), insdelp_* và ROAD (copy code cũ)
  evaluate.py            metric từ attribution đã lưu; LINE(s)
  stats.py               BCa, Wilcoxon, Holm, power, bảng dài
  power.py               SD_d từ JSONL cũ → n
  gate1.py               Cửa 1
  collapse.py            Giai đoạn 1B
  pet/train.py           huấn luyện ResNet-18 Pet theo seed
  pet/ensemble.py        báo cáo mô hình, V_model, V1, faithfulness phần bị loại
tests/
  test_toy.py            U1–U9 (+ kiểm độ chệch của node=mid)
  test_pipeline.py       spec, seed, gamma, ánh xạ đường, IoU, power, Holm, lưu trữ
```

## Lưu trữ (`runs/<setting>/`)

| File | Nội dung |
|---|---|
| `meta/<ảnh>.json`, `meta/<ảnh>_mask.pt` | id, lớp đích, D, f(x), f(x′), nguồn; mask (và valid) |
| `attr/<ảnh>/<spec>__L<mức>__b<b>.pt` | attribution float32 (3, 224, 224), khoảng 0.6 MB |
| `compute.jsonl` | mỗi sub-estimate: K, N, paths, n_backward, residual, thời gian, seed |
| `metrics.jsonl` | image, spec, level, K, N, b (`"mean"` = tham chiếu), n_backward, residual, iou, insp/delp/insdelp_blur, road_morf/lerf/road |
| `ladder_ext.json` | các mức thang đã thêm |
| `convergence_<nhóm>.md/.json`, `gate1.md/.json`, `collapse.md/.json` | báo cáo |

## Chi phí và ghi chú thực hành

- Mỗi ảnh ở 1A cần khoảng 40.000 backward. Metric trên mỗi bản đồ tốn khoảng 220 forward
  (insdel 2×101 + ROAD 18); mỗi ảnh có khoảng 70 bản đồ.
- SPI_DP với N = 192 cấp phát vài tensor float64 cỡ (192, 3, 224, 224), khoảng 230 MB mỗi cái. Nếu
  thiếu bộ nhớ, giảm `model.grad_batch`; sampler luôn rút từng đường một.
- `estimator.rng: device` sinh ngẫu nhiên trên GPU: nhanh, tái lập được trên cùng loại GPU.
  Đặt `cpu` nếu cần tái lập giữa các máy khác nhau (chậm hơn nhiều ở N lớn).
- SPI_BQ dùng `scipy.special.betaincinv` trên CPU, mất cỡ 10–60 s mỗi đường trên ảnh. Chỉ dùng khi α ≥ 10.
- Power trong `power.md` tính từ estimator 8×24 so với IG N = 24, không phải từ tham chiếu
  population. Gate 1 tính lại power từ SD_d thật.
