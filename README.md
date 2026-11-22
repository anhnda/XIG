# attr_probe — Attribution Mechanism Pilot

Pilot kiểm cơ chế attribution theo task P0–P3. Không train gì. ResNet-50 `IMAGENET1K_V2`, logit lớp dự đoán,
baseline đen, cùng tiền xử lý và cách chia ảnh như các đợt trước.
Đầu ra cuối: `runs_probe/pilot/ATTR_PROBE_PILOT_REPORT.md` (tự sinh, kèm CSV và PNG).

## Chạy

```bash
pip install -r requirements.txt
# sửa local: trong config.yaml (đường dẫn gtsegs_ijcv.mat, các file index cũ)
bash scripts/run.sh test       # CPU, vài giây → tests_report.txt
bash scripts/run.sh debug      # 1–2 ảnh, ít bước, toàn pipeline → runs_probe_debug/pilot/ATTR_PROBE_PILOT_REPORT.md
bash scripts/run.sh estimate   # đo GPU, in ước tính phút cho conv và main
bash scripts/run.sh pilot      # conv → main → report  → runs_probe/pilot/ATTR_PROBE_PILOT_REPORT.md
```

Bị ngắt thì chạy lại cùng lệnh: ảnh đã xong (cùng hash config) được bỏ qua. `bash scripts/run.sh report` dựng
lại đồ thị + báo cáo từ CSV mà không chạy GPU.

## P4 — Real-CNN Stochastic Path Sampling Benchmark

```bash
bash scripts/run.sh sampling-test      # unit test P4 + test cũ (CPU) → runs_probe/pilot/sampling/tests_report.txt
bash scripts/run.sh sampling-debug     # 2 ảnh, N = 8, K ∈ {2,4} → runs_probe_debug/pilot/ATTR_PROBE_SAMPLING_REPORT.md
bash scripts/run.sh sampling-estimate  # đo GPU, in phút từng khối + kế hoạch theo sampling.budget (không chạy)
bash scripts/run.sh sampling-pilot     # ref → primary → khối phụ trong ngân sách → runs_probe/pilot/ATTR_PROBE_SAMPLING_REPORT.md
bash scripts/run.sh sampling-report    # dựng lại thống kê/đồ thị/báo cáo từ CSV (DEBUG=1 cho thư mục debug)
bash scripts/run.sh sampling-quad      # C: cùng tập đường, M ∈ quad_check.M trên vài ảnh (chạy trước khi tăng số ảnh)
```

Thí nghiệm theo sau: A = khối `pixel` (S2p/S3p/S4p: cân bằng giữa các đường, mỗi phần tử lịch riêng, đúng q_SPI);
B = khối `eta025/eta05/eta075/eta1` (quét η, đối chứng S6 cùng η); C = `sampling-quad`. Báo cáo có mục A, B, C riêng.

Cấu hình ở mục `sampling:` của `config.yaml` (hash riêng theo khối, không làm mất cache P0–P3). Sampler: `probe/sampling.py`;
runner: `probe/sampling_benchmark.py`; thống kê + báo cáo: `probe/sampling_report.py`; test: `tests/test_sampling.py`.
Bị ngắt thì chạy lại cùng lệnh: (ảnh × khối × seed) đã xong được bỏ qua. Nếu ước tính phần chính vượt
`sampling.budget.minutes × slack` thì chương trình dừng trước khi chạy (đặt `budget.enforce: false` để bỏ qua).

## Cấu trúc

| file | nội dung |
|---|---|
| `probe/paths.py` | Sinh nút: IG (thẳng), SPI (Dirichlet log-space, coupling giữa lưới bằng cộng gộp số gia), BlurIG (điểm đầu mờ), Guided IG (thuật toán tham chiếu PAIR `saliency`) |
| `probe/quad.py` | Đánh giá mạng tại nút + điểm cầu phương; quy tắc `canon_left/mid/rand`, `idgi`, `idgi_raw`, `idgi_rand`; bỏ bước ‖g‖²≈0 minh bạch; chẩn đoán IDGI q_k(M), P_M, entropy, top-1%; `shift`/`unshift` áp thành phần cộng thêm lên đánh giá đã có |
| `probe/components.py` | Tín hiệu S = a·Σ_R(x−x′); bẫy H (∇H bằng autograd), κ theo η, dự đoán a(ρ) cho SPI |
| `probe/pilot.py` | Stage `conv` (P0) và `main` (P1+P2+P3 một lượt/ảnh), cache theo ảnh, ước tính chi phí |
| `probe/metrics.py` | IoU, năng lượng trong mask (khử nhiễu MC), insertion/deletion (blur, đen), cosine, Spearman, top-k |
| `probe/plots.py`, `probe/report.py` | Đồ thị, báo cáo mục A–G + 5 câu hỏi, phán quyết theo luật cố định |
| `tests/test_units.py` | Test A–D của task; bẫy (autograd, η, công thức ΔA_IG); tín hiệu; coupling DP; IDGI ‖g‖=0; Guided/BlurIG trên CNN nhỏ |

## Thiết kế chính

- **Một lượt mạng cho mọi biến thể.** Với IG, SPI, BlurIG (đường không phụ thuộc mô hình), F+S và F+H được áp
  giải tích lên chính f, ∇f đã tính: cùng nút, cùng U_k. Hiệu A(F+c) − A(F) không mang nhiễu MC của đường.
  Guided IG chạy lại cho từng biến thể (cùng U_k), và được phân rã thành phần thành phần/phần đổi đường.
- **Estimator.** `canon_rand` không chệch với tích phân trên đoạn nội suy giữa các nút; đó là canonical chính.
  `canon_left` (bản cũ) vẫn báo để so. Ba biến thể IDGI có tên riêng, không tự rơi sang biến thể khác.
- **Ngân sách.** Mặc định ≈ 15–25 phút trên một GPU lớp A100 (lệnh `estimate` in số đo thật trước khi chạy).
  Phần đắt nhất: SPI 32 đường × 129 eval × 16 ảnh, và Guided IG chạy tuần tự cho 17 biến thể × 6 ảnh.
