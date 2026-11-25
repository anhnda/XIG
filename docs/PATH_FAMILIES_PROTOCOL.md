# PATH_FAMILIES_PROTOCOL — Phase 1: Dirichlet có cần thiết? (bản đã duyệt 24/11, chưa chạy GPU)

Mục tiêu khoa học: kiểm tra Dirichlet có thực sự cần để tạo đường ngẫu nhiên theo feature hiệu quả, và tính chất nào của phân phối
đường giải thích kết quả — không mặc định DP là họ tối ưu. Objective KHÔNG đổi: J_D(θ, B) = L_D(μ_θ) + V_D(θ, B), B = K·N·M (M = 1, `rand`),
D chính `del_sq`. Kết quả BUDGET đã khoá giữ nguyên. Kiểm toán Phase 0: `docs/PATH_FAMILIES_AUDIT.md`.
Mã: `probe/pathfam.py` (runner), `probe/path_families.py` (lịch trình, kiểm tính chất, toy); cấu hình: mục `pathfam:`.

## 1. Quyết định đã duyệt

| # | Quyết định |
|---|---|
| 1 | ε = 10 % J; n_max = 128 ảnh test. **Non-inferiority là chính**; TOST tương đương là phụ. Log-tỷ số theo ảnh chỉ là chẩn đoán. Không nới ε. |
| 2 | Lưới B: N = 32 cho mọi họ; mở rộng N chỉ khi có lý do khoa học rõ ràng và protocol riêng. |
| 3 | smooth giữ trong benchmark, **chỉ mô tả** (không đạt vùng α = 1 khi đơn điệu); không nằm trong kiểm định xác nhận. |
| 4 | Thêm đối chứng DP α = 10, N = 64 (nối mạch E10 / BUDGET). |
| 5 | **Chọn bằng J_rev** trên dev; so với incumbent cố định SPI α = 1, N = 32, `rev` bằng **J_rev**. J_indep báo riêng; không bao giờ so J_indep của ứng viên với J_rev của incumbent. |
| 6 | Kiểm định chính: ln và cmix tại B = 256 và 1024 — 4 phép NI, Holm qua 4. |
| 7 | Phân tích cùng h giữ lại (mô tả); khớp h không được coi là khớp J. |
| 8 | Lực tính trên dev trước khi mở test; thiếu lực ⇒ báo trung thực, không đổi ε / endpoint. |

## 2. Cấu hình (20; dev fresh 1200–1231, 32 ảnh, m = 16 cặp)

| Họ | Cấu hình | Tham số |
|---|---|---|
| spi (DP lưới) | α ∈ {0,1; 0,3; 1; 3; 10}, N = 32 | α (gồm incumbent α = 1) |
| ln | α khớp ∈ {0,1; 0,3; 1; 3; 10}, N = 32 | σ hiệu chuẩn (D̂ lưới = DP): 23,6 / 9,18 / 3,99 / 2,26 / 1,37 |
| cmix | α khớp như trên, N = 32 | λ = 1/(α + 1): 0,909 / 0,769 / 0,5 / 0,25 / 0,091 |
| smooth | δ ∈ {0,5; 0,674; 1,0}, N = 32 | h = δ²/5: 0,05 / 0,091 / 0,2 |
| đối chứng | SPI α = 0,1 N = 64; SPI α = 10 N = 64 | cố định |

σ là tất định (seed 10, d = 20 000, như E10) và được ghi vào `meta.json`. Tính chất mọi lịch trình đã kiểm ở Phase 0 (`pathfam-audit`).

## 3. Lựa chọn (dev, bộ ô CHỌN)

Với mỗi họ F ∈ {ln, cmix} (chính) và {smooth, spi} (mô tả), mỗi B ∈ {256, 1024}: θ*_F(B) = argmin TB J_rev trong lưới của F; báo phân
phối xác suất chọn đầy đủ (bootstrap theo ảnh). θ*_spi chỉ mô tả — phép so chính luôn với incumbent cố định (không chọn lại DP).

## 4. Lực (dev, bộ ô ĐÁNH GIÁ, trước khi mở test)

D_i = J_inc,i − J_F,i + ε·J_inc,i. n = ⌈((z_{1−α/4} + z₀.₈) · SD(D) / (ε·J̄_inc))²⌉ với **giả định g thật = 0** (α/4 = 0,0125: cận Bonferroni
của Holm qua 4); cột phụ dùng g quan sát (tham khảo). n test = max qua 4 phép, kẹp [48, 128]. Phép nào cần > 128 ⇒ ghi "thiếu lực" vào
khoá và báo cáo; kết luận NI của phép đó có thể là "chưa kết luận" — chấp nhận trước.

## 5. Kiểm định (test fresh từ 1300, bộ ô ĐÁNH GIÁ, D chính, J_rev)

g_F(B) = (J̄_inc − J̄_F)/J̄_inc. p một phía bootstrap-t (thuần numpy, tái lập theo seed) trên D tuyến tính hoá chính xác:

| Loại | H0 | Họ phép so | Điều chỉnh |
|---|---|---|---|
| **NI (chính)** | g ≤ −0,10 | ln, cmix × B 256, 1024 | Holm qua 4 |
| Superiority (phụ) | g ≤ 0 | cùng 4 | Holm qua 4 (riêng) |
| Tương đương TOST (phụ) | \|g\| ≥ 0,10 | cùng 4 | Holm qua 4 (riêng) |

Kết luận NI: **KHÔNG KÉM HƠN** nếu p Holm < 0,05; **KÉM HƠN thực tiễn** nếu cận trên CI 95 % của g < −0,10; còn lại **CHƯA KẾT LUẬN**
(không được gọi là tương đương). Kèm CI percentile và Fieller của g, số ảnh thắng.

Mô tả: smooth (g, CI); J_indep riêng (ứng viên indep so với incumbent indep); phân rã L(μ), V₁, 2V_e, ρ, J_rev / J_indep ở mọi B
(128 … 4096 — khả năng "họ đường theo ngân sách"); cùng h: ΔL(μ), ΔV₁ của ln / cmix so với spi; nối mạch đối chứng; độ bền theo 8 D;
kiểm J thực nghiệm với phương pháp CI đã hiệu chuẩn (`budget2-audit`); metric ngoài (không dùng để chọn); kiểm toán chi phí thực
(`cost.csv`). Không dùng dữ liệu test BUDGET / BUDGET2.

## 6. Kế hoạch cuối + chi phí gradient (đo bằng `pathfam-plan`, CPU)

Chi phí đo bằng chính pipeline `schedule → nodes → integrate(rand)` với hàm giả: **mọi 20 cấu hình đều tốn đúng N·M gradient / đường**
(32 với N = 32; 64 với N = 64), sai số completeness với F tuyến tính ≤ 1,8·10⁻¹⁵.

| Giai đoạn | Cấu hình | Đường / cấu hình | Gradient / ảnh | Ảnh | Tổng gradient |
|---|---|---|---|---|---|
| dev | 20 (18 × N = 32, 2 × N = 64) | 32 | 22 528 | 32 | 720 896 (≈ 1,38× BUDGET G1) |
| test | ≤ 6 nghiệm theo họ (N = 32) + incumbent + 2 đối chứng | 64 | ≤ 22 528 | 48–128 (khoá) | 1,08 M – 2,88 M (+ metric ngoài, chỉ forward) |

Thời gian = gradient × thời gian một lượt (đặt `pathfam.per_eval_seconds` để `pathfam-plan` quy ra phút). Trần GPU: dev 240, test 300
phút; vượt ⇒ dừng trước khi chạy.

| B | K (`rev`) ở N = 32 | ở N = 64 |
|---|---|---|
| 256 | 8 | 4 |
| 1024 | 32 | 16 |

## 7. Trạng thái kiểm thử trước khi chạy GPU (đã chạy thật, CPU)

- `pytest tests/`: **76 passed, 0 failed** (gồm 23 test path families, 1 test tích hợp runner trên dữ liệu tổng hợp). Một test cũ của
  budget2-audit (`test_crossfit_removes_selection_optimism`) sai điều kiện (so |g cross-fit| với g chọn-trong-mẫu) — đã sửa điều kiện,
  không sửa mã.
- `pathfam-audit`: đầu mút đúng, đơn điệu đúng, E Z = t (sai lệch MC ≤ 0,007), bất biến khi đảo, toy Mệnh đề (bậc 3, 4 đúng; phản ví dụ
  bậc 5) — xem `PATH_FAMILIES_CHECKS.md`.
- `pathfam-plan`: chi phí gradient như §6.
- Phần GPU của runner (`pathfam-dev`, `pathfam-test`: tải mô hình / ảnh, `budget_opt.process`) chưa chạy ở đây (không có dữ liệu ảnh và
  trọng số trong môi trường này); cùng đường mã đã dùng cho BUDGET / BUDGET2. Nên chạy `bash scripts/run.sh pathfam-dev-debug` trước
  (2 ảnh, N = 4).

## 8. Lệnh

```
bash scripts/run.sh pathfam-dev-debug   # kiểm pipeline GPU (nhỏ)
bash scripts/run.sh pathfam-dev         # dev → selection_dev.json + bảng lực trong INFID_PATHFAM_REPORT.md
bash scripts/run.sh pathfam-lock        # khoá (n test từ lực)
bash scripts/run.sh pathfam-test        # test → runs_probe/pilot/infid/INFID_PATHFAM_REPORT.md
```

## 9. Sau test: kiểm toán, chẩn đoán, lý thuyết, đóng băng (CPU, không GPU, không chọn lại)

```
bash scripts/run.sh pathfam-verify   # tính lại J, g, NI / sup / TOST, Holm từ pairs.csv bằng mã riêng; kiểm khoá → pathfam/PATHFAM_VERIFY.md
bash scripts/run.sh pathfam-diag     # J ↔ metric ngoài (mức B và mức μ), smooth → pathfam/PATHFAM_J_VS_EXTERNAL.md
bash scripts/run.sh pathfam-theory   # kiểm số Phase 2 → pathfam/PATH_FAMILIES_THEORY_CHECKS.md (lý thuyết: docs/PATH_FAMILIES_THEORY.md)
bash scripts/run.sh pathfam-proofcheck  # kiểm toán chứng minh (sympy + MC) → pathfam/PATH_FAMILIES_PROOF_CHECKS.md (docs/PATH_FAMILIES_THEORY_AUDIT.md)
bash scripts/run.sh pathfam-signdiag    # GPU, hậu kiểm có dấu / không dấu trên ảnh test đã khoá → pathfam/PATHFAM_SIGNED_UNSIGNED.md
bash scripts/run.sh pathfam-freeze   # CHẠY CUỐI: sha256 mọi artifact + seed + σ → pathfam/FROZEN.json; chặn pathfam-dev / lock / test
```
Kiểm thử: `pytest tests/` 86 passed (gồm kiểm tích hợp verify / diag / freeze / signdiag-report trên dữ liệu tổng hợp, kiểm số và kiểm
symbolic các định lý Phase 2).

**pathfam-signdiag** (hậu kiểm, mô tả, không chọn lại): dựng lại bản đồ của các cấu hình test đã khoá trên chính các ảnh test đã khoá
(luồng ngẫu nhiên riêng `pfsd`; m = 16 cặp ⇒ bản đồ ở K của B = 256 / 1024 và Ē của 32 đường). Chi phí ≈ Σ_cấu hình 2·m·N gradient / ảnh
(≈ 8 cấu hình × 32 × ≤ 64 ≈ 11–16 nghìn / ảnh) + 2 × InsDel (chỉ forward); trần `pathfam.signdiag.gpu_minutes` = 180. Giả thuyết đặt trước
**H-SU**: metric xếp hạng CÓ DẤU (InsDel có dấu, năng lượng A⁺ trong mask, IoU theo A⁺) đồng thuận với J nhiều hơn metric KHÔNG DẤU
tương ứng (so cặp τ theo ảnh). Đo kèm completeness thực nghiệm, tỷ lệ triệt tiêu ‖A‖₁/|ΣA|, khối lượng âm, triệt tiêu giữa kênh,
infid_β (thang đo tối ưu).

## 10. Vòng chẩn đoán cuối (hậu kiểm, mô tả; không chọn lại, không đổi endpoint, không tạo tuyên bố xác nhận)

Vòng 1 (`pathfam-signdiag`) chỉ lưu thống kê tổng hợp, không lưu attribution ⇒ không có cache để tính β, sai số completeness tuyệt đối hay
so sánh cùng h-N-K mà không chạy lại. `pathfam-signdiag2` dựng lại bản đồ MỘT lần (luồng `pfsd2`) trên ảnh / cấu hình test đã khoá, thêm
cấu hình đối ứng spi / ln / cmix cùng h, N = 32 (chỉ để so sánh có kiểm soát), và lưu: yᵀy, yᵀd, dᵀd trên bộ chọn và bộ đánh giá (β trong /
ngoài mẫu), F(x), F(x′), ΣA cho bản đồ lồng nhau K ∈ {2, 4, 8, 16, 32} và Ē, InsDel có dấu / không dấu với nền blur VÀ nền đen. Định nghĩa
metric và phát hiện lệch nền: `docs/PATHFAM_METRIC_AUDIT.md`. Sau vòng này đóng phần thực nghiệm (`pathfam-freeze`).

**Kiểm toán cuối** (`pathfam-final-audit`, CPU): (1) β = (‖d‖/‖y‖)·cos(y, d) theo ảnh; L ở β = 1 (đã khoá), β riêng từng ảnh (trong mẫu, cross-fit ô —
chẩn đoán, KHÔNG triển khai được vì dùng ΔF_S của chính ảnh), β chung (trong mẫu) và β chung ngoài mẫu theo ảnh (bỏ-một-ảnh — hiệu chỉnh
triển khai được duy nhất); (2) luật completeness K^{−1/2}: mô phỏng CPU với cầu phương `rand` M = 1, cặp `rev` độc lập, thiết kế lồng nhau và
độc lập — độ dốc −0,496 [−0,508; −0,483] và −0,492 [−0,503; −0,481], sai số có dấu có CI chứa 0 ở mọi K (4000 lần lặp); cùng phân tích
trên dữ liệu thật với CI theo ảnh; (3) khoá chẩn đoán (`DIAG_LOCK.json`). Thứ tự đóng: `pathfam-final-audit` → `pathfam-freeze`.
