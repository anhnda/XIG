# PATH_FAMILIES_AUDIT — Phase 0: kiểm toán E10, COORD_MIX và 4 họ đường ngẫu nhiên theo feature

Mục tiêu: trước khi so các họ đường theo J, xác định chính xác (i) các họ đã được định nghĩa thế nào, (ii) cái gì tái lập được,
(iii) E10 thực sự đã kiểm định điều gì. Không có GPU trong Phase 0. Kiểm tra bằng máy: `bash scripts/run.sh pathfam-audit` (CPU)
→ `runs_probe/pilot/infid/pathfam/PATH_FAMILIES_CHECKS.md`.

Câu "DP chỉ là một design choice" có trong lịch sử thảo luận: `docs/SPI_Discuss.md` dòng ~13408 ("Đúng. DP chỉ là một design choice.
Không có lý do gì phải ép lý thuyết tổng quát của chúng ta về Dirichlet Process…"), dựa trên E10; tóm tắt ở `docs/SUMMARY_SPI_DISCUSS.md`
dòng ~380.

## 1. Kiểm kê artifact

| Thành phần | Ở đâu | Trạng thái |
|---|---|---|
| Mã E10 (`e10.py`, `e10_report.py`), luật equivalence (`e8_report.analyse`), mục `e10:` của config | `Archive.zip` gốc (đã bị gỡ khỏi repo hiện tại ở bước dọn file ngoài proposal) | **Có.** Định nghĩa họ, hiệu chuẩn, thiết kế đọc được chính xác |
| Kết quả E10: `e10_fam.csv`, `e10_paths.csv`, `e10_gate.csv`, `meta.json`, `e10_prereg.json` (σ đã hiệu chuẩn), `E10_REPORT.md` | `runs_probe/e10_fresh/` trên máy chạy cũ | **Không có trong repo / upload.** Số liệu E10 hiện chỉ còn là trích dẫn trong `SPI_Discuss.md` (dòng ~13451–13462, ~15312–15325) |
| COORD_MIX, LINE mixture | Không có mã trong `Archive.zip`; định nghĩa trong `SPI_Discuss.md` (dòng ~7776–7800, ~8090–8110, ~8508–8530, ~15262–15279); kết quả từ "báo cáo giai đoạn đầu" (`Pasted text(20261116-103215).txt`) | **Không tái lập được từ artifact.** Chỉ có định nghĩa bằng lời + vài con số làm tròn |
| SMOOTH, LN, DP trong khung INFID/BUDGET | — | Chưa từng được đo bằng J. Mã lịch trình mới: `probe/path_families.py` |

Cảnh báo tên gọi: `SPI_Discuss.md` dòng ~12378 dùng "E10" cho một kế hoạch KHÁC (E10-A/B/C: grouped-DP, RGB, tương tác bậc ba). E10 đã
chạy là "Random Path Family Ablation". Tài liệu này chỉ nói về cái sau.

**Để tái lập E10 không cần GPU**: đặt `path_families.e10_csv` trỏ tới `e10_fam.csv` cũ rồi chạy `pathfam-audit`; mục 2 của báo cáo sẽ tính
lại TOST thật sự (t và bootstrap) cho từng metric.

## 2. Định nghĩa chính xác (đối chiếu mã)

Mọi họ: γ_i(t) = x′_i + Z_i(t)(x_i − x′_i), các phần tử (pixel × kênh) độc lập, Z_i đơn điệu, Z(0) = 0, Z(1) = 1, E Z_i(t) = t.

| Họ | Định nghĩa (mã) | Tham số | Ghi chú kiểm toán |
|---|---|---|---|
| `dp` | số gia Dirichlet(α/N, …, α/N) trên lưới N (`spi_increments`) | α, N | Var Z(t_k) = t_k(1 − t_k)/(α + 1) đúng tại nút. Luật phụ thuộc N (không phải DP liên tục chính xác) |
| `ln` (E10 "lognormal") | số gia = softmax_k(σ ε_k), ε iid N(0, 1), cộng dồn | σ, N | σ hiệu chuẩn Monte Carlo cho D̂ lưới = D̂ của DP (E10: d = 20000, seed 10, chia đôi log σ). Chỉ khớp **tích phân** D, không khớp Var Z(t) từng t. Giá trị σ của E10 nằm trong `e10_prereg.json` — không có ở đây |
| `smooth` | Z_i(t) = t + δ s_i t(1 − t), s_i = ±1 đều; E10: δ = √(5/(α + 1)) | δ | **Mỗi phần tử chỉ có 2 lịch trình** (một bit ngẫu nhiên). Đơn điệu ⇔ δ ≤ 1 ⇔ D ≤ 1/30 ⇔ α khớp ≥ 4. Mã E10 có `clamp(0, 1)` nhưng không kích hoạt ở δ = 0,674. **Không đạt được vùng α = 1 của incumbent** (D = 1/12) mà vẫn đơn điệu |
| `cmix` (COORD_MIX) | với xác suất λ: phần tử nhảy trọn 0 → 1 tại một thời điểm đều; ngược lại đi theo IG | λ = h = 1/(α + 1) | Khớp CHÍNH XÁC E Z(t) và Var Z(t) = h t(1 − t) với DP ở mọi t; độ nhọn Z(1/2) = 1/h (DP ≈ 3) — cùng hai moment đầu, hình dạng khác hẳn |

### 2.1 Bước nhảy và quy tắc cầu phương

Trong thời gian liên tục, `cmix` (và DP với α nhỏ) có bước nhảy: tích phân đường dọc bước nhảy phải lấy theo đoạn thẳng của tọa độ nhảy
(các tọa độ khác cố định), không phải giá trị gradient tại một điểm của một đường trơn. Trong pipeline INFID/BUDGET, mọi họ được định
nghĩa **trên lưới N nút** và đường là nội suy tuyến tính từng đoạn: bước nhảy nằm trọn trong một đoạn rộng 1/N, nên đường vẫn liên tục,
đơn điệu, và quy tắc `rand` (M = 1) vẫn **không chệch cho tích phân trên chính đường đã rời rạc hoá** (định lý đã kiểm ở P0/C3). Hệ quả cần
ghi rõ: `cmix(N)` là một họ phụ thuộc N (độ rộng bước nhảy 1/N), không phải COORD_MIX liên tục; điều này giống hệt DP lưới. Không cần quy
tắc cầu phương riêng; chỉ cần không diễn giải kết quả như giới hạn N → ∞. Phương sai mỗi đường của tọa độ nhảy lớn (một điểm ngẫu nhiên
đại diện cho cả cú nhảy) — đây chính là thành phần V mà J đo.

### 2.2 Coupling `rev`

`rev`: Z^R_i(t) = 1 − Z_i(1 − t). Đảo giữ nguyên phân phối biên (do đó kỳ vọng attribution) của cả 4 họ:
- `dp`, `ln`: số gia hoán đổi được (exchangeable) ⇒ đảo thứ tự số gia giữ nguyên phân phối;
- `cmix`: vị trí nhảy đều ⇒ đối xứng; phần IG bất biến;
- `smooth`: Z^R = t − δ s t(1 − t), tức s ↦ −s, s đối xứng.
Vì vậy J_indep tính từ dữ liệu cặp (dùng phân phối biên một đường) và J_rev đều hợp lệ cho cả 4 họ. Kiểm bằng KS trong
`PATH_FAMILIES_CHECKS.md` (`ks_rev_max`). Lưu ý riêng `smooth`: cặp rev là (s, −s) cho MỌI phần tử cùng lúc — antithetic mạnh nhất có
thể với một bit; V_e của smooth có thể rất nhỏ. Theo yêu cầu, `indep` là chuẩn chung; `rev` báo kèm.

### 2.3 Chi phí

Với cùng N, M, mọi họ tốn đúng N·M gradient mỗi đường; chi phí sinh lịch trình (CPU/GPU, không gradient) không đáng kể. Không họ nào cần
gradient bổ sung khi định nghĩa trên lưới. Ngân sách B = K·N·M so sánh được trực tiếp giữa các họ.

## 3. E10 thực sự đã kiểm định gì

**Thiết kế (từ mã + config):** 50 ảnh fresh (bỏ qua 100 ảnh đầu), baseline đen, α = 10 (h = 1/11), N = 64, **M = 4**, 2 × 8 đường rời nhau,
đầu ra chính = bản đồ trung bình 16 đường. Khớp exploration D ≈ 0,015 (DP lưới), δ = √(5/11) ≈ 0,674, σ hiệu chuẩn Monte Carlo.
Giả thuyết: H1 smooth − dp và H2 lognormal − dp, **equivalence**, biên **±0,02 tuyệt đối** trên IoU, InsDel-blur, InsDel-black; H3–H5
hai phía (ổn định, dp − ig) với Holm trong "primary".

**Luật equivalence trong mã:** "TƯƠNG ĐƯƠNG" khi CI bootstrap percentile 90 % của TB hiệu theo cặp ảnh nằm trong (−0,02; +0,02). Đây đúng
là TOST mức 5 % mỗi phía (phiên bản percentile). Kết luận "tương đương ở mọi metric" là phép giao–hợp (IUT), không cần hiệu chỉnh đa
kiểm định; kết luận từng metric riêng thì chưa hiệu chỉnh. Ngoài ra có nhánh "KHÁC (vượt biên)" và "CHƯA KẾT LUẬN".

**Con số còn lại (trích dẫn, chưa tái lập):**

| | IoU (trung vị) | InsDel-blur | InsDel-black |
|---|---|---|---|
| DP α = 10 | 0,415 | 0,149 | 0,089 |
| smooth | 0,401 | 0,107 | 0,107 |
| logistic-normal | 0,410 | 0,139 | 0,093 |

logistic-normal − DP: ΔIoU = +0,002 [−0,0006; 0,005] (CI 95 %), Δblur ≈ −0,004, Δblack ≈ 0. smooth − DP: TB hiệu blur −0,008 (theo trích dẫn).

**Đánh giá:**
1. Với logistic-normal, CI 95 % của IoU nằm gọn trong ±0,02 ⇒ CI 90 % cũng vậy ⇒ **TOST đạt** cho IoU. Hai metric còn lại chỉ có điểm
   ước lượng trong trích dẫn; muốn khẳng định TOST cần CSV.
2. Với smooth, tài liệu tự mâu thuẫn: một đoạn nói "cả smooth và logistic-normal đều đạt tiêu chí tương đương", đoạn khác nói smooth
   "vẫn hơi kém DP ở một số metric". Trung vị blur 0,107 so với 0,149 là chênh lớn; TB hiệu −0,008 thì trong biên. **Không phân xử được
   khi chưa có CSV** — không nên trích "smooth tương đương DP" cho tới khi tính lại.
3. E10 **không đo J** và không tách L(μ) khỏi phương sai: metric tính trên bản đồ trung bình 16 đường (trộn kỳ vọng và phương sai K = 16).
   Tương đương về IoU / InsDel không suy ra tương đương về J.
4. E10 chỉ ở **một** mức exploration (h = 1/11, α = 10). Incumbent BUDGET ở α = 1 (h = 1/2), ngoài vùng E10 đã thử và ngoài miền khả thi
   của `smooth`.
5. Biên ±0,02 IoU so với hiệu ứng DP − IG ≈ +0,09 IoU (cùng giai đoạn) là khoảng 20 % hiệu ứng — biên khá rộng; điều này nên được nêu
   khi trích E10.
6. M = 4 trong E10 khác M = 1 của INFID/BUDGET; với `rand` cả hai không chệch theo đường, chỉ khác phương sai.

## 4. COORD_MIX / LINE (giai đoạn sớm)

Định nghĩa: COORD_MIX như bảng mục 2; LINE = chọn ngẫu nhiên CẢ đường giữa IG và SPI(α = 1) với tỷ lệ khớp phương sai biên (s = 2h) —
không độc lập giữa phần tử. Kết quả: ở h = 1/11, tỷ lệ năng lượng attribution trong mask (đã hiệu chỉnh nhiễu MC) DP 0,648, COORD_MIX 0,647,
LINE 0,508, IG 0,508. Chỉ số E theo h có giá trị làm tròn (0,5–0,7) đọc từ hình; các điểm h thấp được **thêm sau khi xem kết quả**
(exploratory). Không có phép kiểm equivalence nào cho COORD_MIX. Kết luận được phép giữ: "khớp mean và variance schedule không đủ —
COORD_MIX ≈ DP ở h vừa, khác ở h nhỏ; LINE ≈ IG". Không tái lập được số liệu; Phase 1 phải đo lại từ đầu với mã mới.

## 5. Kết quả lý thuyết dùng được ngay

Giả định chung: các phần tử độc lập, E Z_i(t) = t, Z_i liên tục tuyệt đối (đường lưới nội suy tuyến tính thoả), cùng phân phối cho mọi
phần tử. Ký hiệu v(t) = Var Z(t), κ₃(t), κ₄(t) là cumulant bậc 3, 4 của Z(t), D = ∫₀¹ v(t) dt.

**Mệnh đề.** Với F đa thức:
1. bậc ≤ 2: attribution kỳ vọng μ = IG cho mọi họ;
2. bậc 3: μ chỉ phụ thuộc phân phối đường qua **một số D**;
3. bậc 4: μ phụ thuộc qua D, ∫ t v(t) dt và ∫ κ₃(t) dt; nếu phân phối **bất biến khi đảo** (Z(1 − t) ~ 1 − Z(t) — đúng cho cả 4 họ, §2.2)
   thì v đối xứng, κ₃ phản đối xứng ⇒ ∫ t v = D/2, ∫ κ₃ = 0 ⇒ **chỉ qua D**;
4. bậc 5: nói chung KHÔNG chỉ qua D, kể cả khi khớp toàn bộ hàm hiệp phương sai (xuất hiện ∫ v², ∫ t² v, ∫ t κ₃, ∫ κ₄).

### 5.1 Chứng minh

A_i = Δ_i ∫₀¹ ∂_iF(γ(t)) dZ_i(t); lấy kỳ vọng từng đơn thức Z_j Z_k … dZ_i (độc lập, E Z = t, tích phân từng phần với Z_i dZ_i =
d(Z_i²/2), v(0) = v(1) = 0).
- Bậc 2: E∫ dZ_i = 1, E∫ Z_j dZ_i = 1/2 (j ≠ i), ∫ Z_i dZ_i = 1/2 — như IG.
- Bậc 3: j ≠ k ≠ i: 1/3; j = k ≠ i: ∫(t² + v) = 1/3 + D; j = i ≠ k: ½∫ t d(t² + v) = 1/3 − D/2; j = k = i: 1/3.
- Bậc 4 (∂_iF bậc 3, bảy mẫu chỉ số): j, k, l khác nhau ≠ i: 1/4; j = k ≠ l (≠ i): 1/4 + ∫ t v; j = k = l ≠ i: ∫ E Z³ = 1/4 + 3∫ t v + ∫ κ₃;
  một chỉ số = i, hai chỉ số còn lại khác nhau: 1/4 − ∫ t v; một chỉ số = i, hai chỉ số còn lại bằng nhau: ∫(t² + v) d(t² + v)/2 = 1/4;
  hai chỉ số = i: 1/3 − ⅓∫ E Z³ = 1/4 − ∫ t v − ⅓∫ κ₃; ba chỉ số = i: 1/4.
- Bậc 5: ví dụ j = k ≠ l = m (≠ i) cho ∫(t² + v)² ⇒ ∫ v²; j = k = l = m ≠ i cho ∫ E Z⁴ ⇒ ∫ κ₄ + 3∫ v² + …. ∎

### 5.2 Kiểm chứng số (đã chạy, `pathfam-audit` §1b và `tests/test_path_families.py`)

Toy d = 3, N = 8, 200 000 đường mỗi họ, attribution tính chính xác trên đường nội suy (Gauss–Legendre), D tính chính xác:
- bậc 3 và bậc 4: E A_f − A_IG = D_f·a₁ với cùng a₁ cho dp (α = 1, 3), cmix (λ = ½, ¼), ln (khớp α = 1), smooth (δ = 1): max |z| ≤ 2,5;
- phản ví dụ bậc 5 có chủ đích F = x₀(x₁ − ½)⁴: dp α = 1 và cmix λ = ½ có CÙNG D và cùng hàm hiệp phương sai nhưng E A₀ khác nhau,
  z = 31; ln (chỉ khớp D) khác dp z = −9,6. Hệ số bậc 5 ngẫu nhiên thì đóng góp cumulant nhỏ hơn nhiễu MC (không phát hiện).

**Hệ quả cho thiết kế (giả thuyết, không phải kết quả):** ở cùng D, khác biệt L_D(μ) giữa các họ chỉ đến từ phần bậc ≥ 5 / không đa thức
của mô hình (kink ReLU), còn V phụ thuộc toàn bộ phân phối đường. Dự đoán kiểm trong Phase 1 (mô tả): ở cùng h, các họ khác nhau
chủ yếu ở V hơn ở L(μ). Lưu ý bất biến khi đảo vừa là điều kiện cho `rev`, vừa là điều kiện để bậc 4 chỉ phụ thuộc D.

## 6. Tình trạng tái lập

| Mục | Tái lập được? |
|---|---|
| Định nghĩa 4 họ, hiệu chuẩn σ (thủ tục), khớp λ = h | Có — `probe/path_families.py`, kiểm bằng `pathfam-audit` và `tests/test_path_families.py` |
| Thiết kế, giả thuyết, luật equivalence E10 | Có — đọc từ mã trong `Archive.zip` |
| Số liệu E10 | Chỉ khi có `e10_fam.csv`; hiện chỉ là trích dẫn |
| σ chính xác dùng trong E10 | Chỉ khi có `e10_prereg.json` (hoặc chạy lại thủ tục hiệu chuẩn với cùng seed 10, d = 20000 — tất định) |
| COORD_MIX / LINE số liệu | Không |
| So sánh theo J | Chưa từng làm — Phase 1 |
