# LIG revision — code mới (Bước 0, dựng lại dnoise, P0–P2)

Code cũ ở thư mục gốc vẫn giữ nguyên để tái lập số liệu của bài. Code mới gồm
package `pathattr/`, các script trong `experiments/` và bộ test trong `tests/`.
Tất cả đều viết bằng PyTorch. **Chưa chạy lần nào:** code được viết và rà soát
bằng cách đọc, nên lần chạy đầu tiên cần bắt đầu bằng `pytest`.

## 1. Thay đổi trong code cũ

| Lỗi | File | Thay đổi |
|---|---|---|
| B1: Δf lệch một bước | `utility.py`, `lig.py` | Thêm cờ `utility.FIX_DF_ALIGNMENT`. Mặc định `False` để tái lập đúng bài; `True` dùng Δf_k = f(γ_{k+1}) − f(γ_k) |
| B5: cache nhóm pixel trả nhóm cũ | `lig.py` | Xoá `_group_cache` mỗi lần gọi `compute_lig` |

Các lỗi sau **không** sửa trong code cũ, vì code mới thay thế chúng:
- B2 (baseline): runner mới yêu cầu khai báo baseline rõ ràng.
- B3 (Guided IG): có bản port chuẩn trong `pathattr.methods.guided_ig`.
- B6 (benchmark không paired): dùng `experiments/run.py`.

## 2. Dựng lại "dnoise"

Mã gốc đã mất, nên phần này dựng lại từ slide 20–22.

**Căn cứ:**
- Hàng dưới của hình ở slide 20 có nhãn `dither s=0.5`.
- Ô phóng to tại t = 0.5 cho thấy một họa tiết đều, mịn (kiểu ordered dither).
- Slide mô tả: "mỗi pixel xuất hiện tại một thời điểm riêng, xen kẽ đều trên ảnh", "lịch trình theo từng pixel", "thêm nhiễu nhỏ ở mỗi bước", 24 gradient.
- Slide 22: các kiểu xen kẽ "ngẫu nhiên, theo kênh màu" cho kết quả ngang hoặc kém hơn.

**Cách hiểu được chọn** (`pathattr.paths.dither`, preset `dnoise`):

```
γ_i(t_k) = x'_i + s_i(t_k)·(x_i − x'_i),   t_k = k/N, N = 24
s_i(t)   = clip((t − s·T_i) / (1 − s), 0, 1),   s = 0.5
T_i      = ngưỡng Bayer 8×8 lặp trên ảnh, dùng chung cho 3 kênh
gradient lấy ở đầu mỗi bước (left Riemann), không rescale
```

- "Xen kẽ đều" chính là ordered dither: mỗi ô 8×8 chứa đủ 64 mức ngưỡng, nên lịch trình được phân tầng theo không gian.
- s = 0 cho ra đường thẳng (IG). s = 1 cho ra mỗi pixel bật hẳn tại T_i (đường batch).
- Tạm hiểu "nhiễu" chính là họa tiết dither. Nhiễu Gauss cộng thêm vẫn có sẵn qua tham số `noise=…`, mặc định là 0.

**Các điểm chưa chắc chắn, nên quét để khớp con số 0.465 / IoU 0.513:**

| Tham số | Mặc định | Nên thử |
|---|---|---|
| `spread` | 0.5 | 0.25, 0.75, 1 |
| `thresh` | bayer | stratified, random |
| `tile` | 8 | 4, 16 |
| `noise` | 0 | 0.02, 0.05 |
| `per_channel` | false | true |
| `est` | left | mid, random |

Các preset khác có sẵn:
- `pixel_random`: từng pixel, thứ tự ngẫu nhiên, bật hẳn (slide 21).
- `pixel_strat`: phân tầng theo tile, bật hẳn.
- `block14`: khối 14×14 ngẫu nhiên (slide 21).
- `spi:alpha=…`: SPI (Jeon et al., ICCV 2023).

## 3. Cấu trúc

```
pathattr/oracle.py      đếm forward/backward (trả lời R3)
pathattr/path_eval.py   eval_path(): bản đúng duy nhất của (f, g, h, d, Δf)
pathattr/paths.py       straight, threshold_map, pixel_schedule, dither, spi
pathattr/estimators.py  left / mid / random / gl, cv_complete (thay rescale)
pathattr/methods.py     ig, idgi (complete, không rescale), guided_ig (port chuẩn), dither, spi
pathattr/metrics.py     baseline black/gray/white/blur, ins/del, overlap/IoU/pointing, checker
pathattr/data.py        ImageNet-Seg (gtsegs_ijcv.mat), thư mục ảnh; danh sách ảnh cố định
experiments/run.py      runner paired, JSONL theo ảnh, chạy tiếp được
experiments/summarize.py  mean ± std, Δ paired, bootstrap CI, Wilcoxon
experiments/p0_mu_projection.py  kiểm tra mệnh đề suy biến của Phase 1
experiments/p1_estimators.py     giữ đường, đổi estimator
tests/                  pytest trên mô hình đồ chơi float64
```

**Định nghĩa localisation:** đây là giả định của mình, vì code gốc đã mất.
- overlap: phần năng lượng Σ_c|attr| nằm trong mask.
- IoU: IoU giữa top-|mask| pixel và mask.
- pointing: điểm có attribution lớn nhất có nằm trong mask hay không.
- checker: năng lượng tần số cao ngoài mask.

## 4. Chạy

**Chỉ cần file `gtsegs_ijcv.mat`:** ảnh (`/value/img`) và mask (`/value/gt`) đều nằm
trong file này. Link tải: http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat

Cách chạy nhanh nhất là dùng script theo từng giai đoạn:

```bash
bash scripts/run_all.sh /path/gtsegs_ijcv.mat test
bash scripts/run_all.sh /path/gtsegs_ijcv.mat calib            # xem runs/calib_black.md trước khi đi tiếp
BASE=gray bash scripts/run_all.sh /path/gtsegs_ijcv.mat calib  # nếu IG chưa khớp slide
bash scripts/run_all.sh /path/gtsegs_ijcv.mat dnoise           # sau đó lần lượt: b1, p0, p1, p2
```

Lần chạy đầu sẽ tạo `runs/seg200.json`, danh sách 200 ảnh theo thứ tự ngẫu nhiên cố định.
Mọi giai đoạn sau đều dùng lại danh sách này (`b1` lấy 50 ảnh đầu). Có thể dừng giữa chừng:
chạy lại cùng một lệnh sẽ bỏ qua các dòng đã xong.

Chi tiết từng lệnh nằm bên dưới.

```bash
pip install pytest h5py scipy        # h5py cho ImageNet-Seg, scipy cho Wilcoxon (tuỳ chọn)
pytest -q tests                      # chạy cái này trước tiên
```

**Hiệu chỉnh giao thức.** IG và Guided IG phải khớp slide (0.282/0.423 và 0.463/0.491) trước khi so dnoise:

```bash
python -m experiments.run --data seg --seg-mat /path/gtsegs_ijcv.mat --n 200 \
  --index-file runs/seg200.json --baseline black --insdel black blur \
  --methods ig guided_ig legacy_guided_ig dnoise pixel_random block14 \
  --seeds 0 1 2 --out runs/calib.jsonl
python -m experiments.summarize runs/calib.jsonl --ref ig \
  --metrics insdel_black insdel_blur overlap iou pointing checker
```

Nếu IG chưa khớp, hãy thử `--baseline gray --insdel gray` (đó là baseline mà batch mode cũ đã dùng), hoặc `--model resnet50_v1`.

**Quét dnoise:**

```bash
python -m experiments.run ... --methods dnoise "dnoise:spread=0.25" "dnoise:spread=0.75" \
  "dnoise:spread=1" "dnoise:thresh=stratified" "dnoise:thresh=random" "dnoise:tile=4" \
  "dnoise:noise=0.02" "dnoise:per_channel=true" "dnoise:est=mid" --seeds 0 1 2 --out runs/dnoise.jsonl
```

**Tác động của B1** (ảnh cũ, baseline xám như batch mode cũ):

```bash
python -m experiments.run --data folder --images ./sample_imagenet1k --n 50 \
  --index-file runs/old50.json --baseline gray --insdel gray \
  --methods legacy_idig "legacy_idig:fix_df=1" idgi legacy_lig_idig "legacy_lig_idig:fix_df=1" \
            legacy_lig "legacy_lig:fix_df=1" --out runs/b1.jsonl
python -m experiments.summarize runs/b1.jsonl --ref legacy_idig --metrics insdel_gray
```

Trong `info` có thêm `Q`, `mu_argmax`, `mu_support` cho các phương pháp legacy. Cột `residual` của `idgi` phải xấp xỉ 0.

**P0, μ của Phase 1:**

```bash
python -m experiments.p0_mu_projection --data seg --seg-mat /path/gtsegs_ijcv.mat --n 200 \
  --index-file runs/seg200.json --out runs/p0_mu.jsonl
```

Cách đọc kết quả:
- `theorem_applies=true` nghĩa là e_k* là nghiệm toàn cục của J.
- So `adam.argmax` với `k_star` để biết Adam có hội tụ không.
- `adam_legacy.argmax == 0` và `L1_adam_leg_vs_e0 ≈ 0` nghĩa là bản gốc dồn μ vào bước 0, chính là bước có Δf = 0 do lỗi B1.

**P1, giữ đường dnoise, đổi estimator:**

```bash
python -m experiments.p1_estimators --data seg --seg-mat /path/gtsegs_ijcv.mat --n 200 \
  --index-file runs/seg200.json --path dnoise --seeds 0 1 2 --out runs/p1.jsonl
python -m experiments.summarize runs/p1.jsonl --ref "dnoise|gl" --metrics insdel_black iou overlap
```

- `gl` tốt bằng hoặc hơn `left`: đi theo nhánh A (ước lượng tích phân đường).
- `left` tốt hơn rõ rệt: đi theo nhánh B.

**P2, so các đường** (SPI ở nhiều giá trị α):

```bash
python -m experiments.run ... --methods "ig:N=24" "ig:N=200" "ig:N=24,est=gl,Q=4" dnoise \
  pixel_random "spi:alpha=0.01" "spi:alpha=0.1" "spi:alpha=1" "spi:alpha=10" \
  "dnoise:K=2,N=12" --seeds 0 1 2 --out runs/p2.jsonl
```
