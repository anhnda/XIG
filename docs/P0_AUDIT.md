# P0 — Audit code cho proposal "Lấy mẫu đường SPI theo infidelity" (protocol v1.0)

Audit đọc trực tiếp từ code (không chạy), trên repo TRƯỚC khi dọn; sau khi dọn, phần P4 còn dùng nằm ở `probe/pathbank.py` (Env, Bank, khoá ngẫu nhiên giữ nguyên). Mục tiêu: xác định đúng những gì `probe/sampling.py`, `probe/sampling_benchmark.py`,
`probe/pathscore.py`, `probe/pathscore_metrics.py` (bản trước khi dọn repo) đang làm, trước khi dùng các mệnh đề T1–T10.

## 1. Sinh lịch trình Dirichlet (C1a)

- `paths.spi_increments(α, N, shape, gen)`: với a = α/N, rút log Gamma(a+1) + log U / a rồi softmax theo trục N
  ⇒ số gia (δ_1..δ_N) ~ Dirichlet(a, …, a) **độc lập theo từng phần tử**. Gamma(a+1) rút bằng Marsaglia–Tsang chính xác
  (`gamma_ge1`, rejection), mọi ngẫu nhiên đi qua `gen` (GPU nếu ctx trên GPU), float64.
- Tham số Dirichlet bằng nhau ⇒ dãy số gia trao đổi được ⇒ phép đảo δ ↦ flip(δ) giữ nguyên phân phối. Đây là chứng minh C1a
  cho `rev` trên họ `spi`.
- `paths.cum_nodes`: Z_0 = 0, nửa đầu cộng dồn xuôi, nửa sau tính bằng phần bù của tổng đuôi, rồi `clamp` + `cummax`.
  Về số học khác thứ tự cộng giữa hai nửa nhưng chính xác trong số học thực; test cũ
  `test_C1b_reverse_indexing_endpoints` (tests/test_infid.py) kiểm sai khác < 1e-13.
- Họ `patch` (`patch_schedule`): mỗi patch một DP(α, N), xoay vòng để số gia lớn nhất nằm ở ⌊N·φ_g⌋. Đảo đường đưa anchor về
  N−1−a; với pha thiết kế `indep` (φ iid U(0,1)) vị trí này vẫn đều ⇒ `rev` giữ phân phối. Họ `hier` = (1−η)·patch + η·spi
  với cùng chỉ số đường; đảo đồng thời hai thành phần ⇒ giữ phân phối.
- "Phần tử": trong họ `spi` là **pixel × kênh** (C·H·W = 3·224·224 = 150 528 lịch trình độc lập); trong họ `patch` là
  **patch** (mọi pixel và kênh của một patch dùng chung một lịch trình); `pix` (S2p/S3p/S4p) là pixel × kênh.

## 2. Phép `rev`

- `sampling.reverse_nodes(Z) = 1 − Z.flip(0)` ⇒ Z^R_k = 1 − Z_{N−k}, giữ hai đầu mút và tính đơn điệu; số gia của Z^R là
  flip(δ). Đúng định nghĩa R: Z(t) ↦ 1 − Z(1 − t) trên lưới nút.
- Trong P4, cặp antithetic là khoá `("spi", j)` và `("spiR", j)`; khoá `qtag` của cầu phương chứa cả khoá đường ⇒ U của hai
  thành viên trong cặp **độc lập**. Với quy tắc `rand` điều này vẫn giữ phân phối của T(RZ) (U′ ⟂ đường, U′ ~ U(0,1)).
- Hệ quả của U độc lập (quy tắc `rand`): T = T_poly + ε với E[ε | Z] = 0 ⇒ nhiễu cầu phương cộng **bằng nhau** vào V_e và V_o
  (Var((ε₁ ± ε₂)/2) như nhau). `rev` không bị hại bởi nhiễu này nhưng nó kéo ρ về gần 1.

## 3. Định nghĩa K, N, M, G (trả lời câu hỏi mở thứ nhất)

| Ký hiệu | Trong code | Giá trị P4 / PATHSCORE |
| --- | --- | --- |
| N | số đoạn của đường gấp khúc = số số gia Dirichlet (`paths.spi10.steps`) | 64 |
| M | **số điểm gradient MỖI ĐOẠN** (`aw_stream`: s = (m + U_{k,m})/M) | 1 |
| K | số đường trong ensemble; cặp `rev` tính cả hai thành viên | 16 (PATHSCORE), {2,4,8,16} (P4) |
| G | cạnh lưới patch (`sampling.grid`) | 7 (patch 32 px) |
| α | `paths.spi10.alpha` | 10 |

Số gradient: N·M mỗi đường, K·N·M mỗi ensemble. PATHSCORE: 16 đường × 2 seed × 64 × 1 = 2 048 gradient mỗi ảnh.
Bộ ước lượng chỉ dùng cặp với m cặp: 2m·N·M gradient.

## 4. Quy tắc tích phân thực tế — **khác giả định của proposal**

- P4 và PATHSCORE dùng `aw_stream` (nay ở `probe/pathquad.py`) với M = `sampling.M` = 1 ⇒ **canon_rand**: một điểm ngẫu nhiên đều trên mỗi đoạn,
  **dùng chung cho mọi phần tử** (U có shape (N, M), không phải (N, M, d)). Không phải Riemann trái.
- Điểm trái (`canon_left`) chỉ có ở `quad.rules` (P0–P3), không được dùng trong `attr.pt` của PATHSCORE.
- Hệ quả cho lý thuyết:
  1. `rand` không chệch cho tích phân trên đường gấp khúc với **mọi** F ⇒ E[T] = attribution chính xác trên polyline. Các phân
     tích riêng của điểm trái (T9, ρ = 6/(N+1), E[Q_L] = (1 − 1/N)·α/(α+1), phản ví dụ z₁²) **không mô tả cache PATHSCORE**.
  2. Phần chẵn do sai số điểm trái không tồn tại; thay vào đó có nhiễu cầu phương cộng đều vào V_e, V_o (mục 2).
  3. Câu "`left` mặc định cũ" trong proposal nên đọc là "`rand` mặc định cũ". Code mới giữ `rand` làm mặc định cho PATHSCORE (`probe/pathbank.py`)
     (khớp cache), thêm `left`, `mid`, `theta`, `gl` (`probe/pathquad.py`); `sampling.rule` (nếu khác `rand`) đi vào hash.
     Thí nghiệm mới dùng `infid.protocol.rule` (mặc định `mid`, khoá sau E-quadrature).

## 5. Chiếu theo D trong PATHSCORE

- `pathscore_metrics.subset_masks`: 96 ô vuông, cạnh luân phiên [16, 32, 56] px, vị trí đều; generator CPU
  `cgen("pathscore-subset", image)` ⇒ **tái lập được không cần mô hình**. ΔF_S = logit(x) − logit(x với S thay bằng baseline đen),
  cần 96 forward.
- `subset_scores`: dự đoán p_Sᵀ A = Σ_{c, i∈S} A (tổng có dấu theo kênh) — đúng p_S của proposal với hệ số 1.
- `infid` = Σ_S (p_Sᵀ A − ΔF_S)² / Σ_S ΔF_S² **chuẩn hoá theo từng ảnh**, không co giãn β. Bảng 54,1 / 25,6 trong proposal phải
  được tái lập theo đúng định nghĩa này trong E0 (trung bình theo ảnh của tỷ số từng ảnh; tỷ số gộp chỉ là phân tích độ nhạy).

## 6. `global`, `nbr` và RQ1

- S2 / S4 (`pat`) đổi operator sang q_patch ⇒ **không** cùng attribution kỳ vọng với S0 (q_SPI). So `rev` với S2/S4 trộn RQ1
  và RQ2.
- S2p / S4p (`pix`, `pixel_phase_design` + `anchor_rotate` theo phần tử) giữ q_SPI trên từng đường (lập luận xoay vòng trong
  docstring `sampling.py`) ⇒ là `global` / `nbr` đúng cho RQ1 ở q = q_SPI. Trong q_patch, đối chứng đúng là S6 (`indep`).
  `sampling_infid.py` dùng `pix` cho họ `spi` và `pat` cho họ `patch`, kèm chẩn đoán bằng nhau của kỳ vọng (C1 cho thiết kế ghép).

## 7. Unit test đã thêm (`tests/test_infid.py`, chạy bằng `bash scripts/run.sh test`)

- C1a/C1b: đảo hai lần = đồng nhất; số gia đảo = flip; hai đầu mút; Z^R cùng phân phối biên với Z (mô-men Dirichlet và KS tại
  vài nút) cho `spi`, `patch`, `hier`; `pix` giữ phân phối biên.
- Quy tắc: `mid` chính xác cho F bậc 2 trên polyline; `left` lệch đúng (θ − ½)·B∘H (T6); `rand` khớp bit với `aw_stream`;
  lưới lồng nhau cho đúng left/mid ở mọi mức.
- T10: V̂_e, V̂_o, L̂(μ) không chệch trên dữ liệu tổng hợp; phép thử sign-flip giữ mức ý nghĩa.
- Sàn: L*_free = 0, L*_comp = 0,25 cho f = z₁z₂ (Phụ lục).
