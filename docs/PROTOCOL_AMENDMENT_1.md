# Sửa đổi protocol 1 — sau E0 và E-quadrature (23/11/2026)

Áp dụng trước E1a pilot. Ghi lại để mọi thay đổi so với protocol v1.0 đều có dấu vết trước khi `infid-lock`.

## Căn cứ

- **E0:** L̂(μ) ≈ 23,65; phần phương sai ở K = 16 ≈ 1,91 (8%); sàn có completeness mức patch 28×28 ≈ 0,44; infid_β ≈ 0,84–0,88
  (⇔ cos²(PA, ΔF) ≈ 0,16). Completeness không phải nguyên nhân chính của khoảng cách tới sàn (trong phạm vi sàn patch đã đo).
  Giả thuyết "dương/âm tần số cao" CHƯA được kiểm chứng → thêm chẩn đoán làm mượt khối (mục 3).
- **E-quadrature:** 0/28 đường hội tụ tới tol_ref trong phạm vi L_max = 4; không cấu hình nào đạt tol 0,01. `mid` không có lợi thế
  bậc 2 trên ReLU; sai số của `mid` / `left` ở M = 1 phụ thuộc α. Ngoại suy M ≈ 150 chưa phải kết quả đã xác nhận.

## Quyết định

1. **Quy tắc tích phân:** `rand`, M = 1 — khoá TẠM cho E1a pilot. Lý do: không chệch có điều kiện theo đường (không đưa lệch
   quadrature phụ thuộc α vào attribution kỳ vọng) và rẻ nhất. Không khẳng định một mẫu `rand` chính xác hơn một mẫu `mid`.
2. **C3 thay thế** (bỏ yêu cầu hội tụ tới 0,01 làm điều kiện dừng cứng):
   - **C3a** — `rand` không chệch có điều kiện: `tests/test_infid.py::test_C3a_*`, oracle toy "C3a".
   - **C3b** — nhiễu quadrature độc lập trong cặp `rev`: khoá generator riêng cho Z và RZ (`test_C3b_pair_quadrature_streams_independent`),
     và V_o − V_e không đổi so với tích phân chính xác (`test_C3b_rand_noise_cancels_in_VoVe_toy`, oracle toy "C3b"). Khi đó
     V_e^rand = V_e^exact + τ²/2, V_o^rand = V_o^exact + τ²/2 ⇒ ΔV_D không đổi.
   - **C3c** — `quad-check` báo sai số chiếu (lưới lồng nhau) và phương sai quadrature τ²_D (cùng đơn vị infid với σ̂² của E0),
     cùng kiểm lặp lại xác định (cudnn.deterministic, `local.deterministic`). Chỉ là số liệu báo cáo.
   - Hệ quả ghi rõ: σ̂² của E0 gồm cả τ²_D ⇒ σ̂²/K là TRẦN TRÊN của mức giảm có thể đạt; phần `rev` có thể loại bỏ nằm trong σ̂² − τ²_D.
3. **IG không chệch:** với `rand`, r = `e1b.ig_repeats` lần rút U trên cùng đường thẳng, L̂(μ_IG) = L(Ā) − σ̂²_IG / r. Thêm đối chứng
   phụ cùng ngân sách gradient J_IG(K) so với J_rev(K) (mô tả, không thuộc H3). Metric ngoài của IG ở E2 tính trên trung bình K lần rút.
4. **E0 — chẩn đoán làm mượt khối** S_h (h = 8, 16, 32 px, bảo toàn tổng): L_D(A), L_D(S_h A), L̂(S_h μ) trên tập ô đánh giá độc lập,
   so với sàn patch cùng lưới. Là chẩn đoán, không phải phương pháp SPI mới (không giữ dummy).
5. **Endpoint:** giữ ΔV_D(K) là endpoint chính, K chính = 4; K ∈ {2, 8, 16} và 1 − ρ̂ là endpoint PHỤ (không đồng chính).
6. **G0:** δ₀ = 1,0 — dư địa đo được ≈ 25 [9,9; 45], nên E1 mở với ngân sách đầy đủ. Dư địa gần như toàn bộ là phần lệch ⇒ chủ yếu
   biện minh cho RQ2.
7. **E1a chính:** 48 ảnh mới (split fresh, offset 600, không giao ảnh pilot dùng để chọn K / δ). Cỡ mẫu cuối chốt từ sd của ΔV_D trong
   pilot (báo cáo pilot có bảng chọn m × số ảnh).

## Trọng tâm nghiên cứu

Câu hỏi chính chuyển sang RQ2: vì sao attribution kỳ vọng của SPI có infid ≈ 24 trong khi tồn tại bản đồ cộng tính có completeness đạt
< 1. Kết quả âm tính hợp lệ: nếu mọi đường trong họ SPI đều cho độ lớn cục bộ không khớp D, vấn đề nằm ở họ phương pháp đường. Nghiệm
bình phương tối thiểu có completeness (sàn patch) là benchmark về khả năng biểu diễn, không phải ứng viên thay thế.

## Trình tự

`bash scripts/run.sh test` → `toy` (C3a/C3b) → `pathscore-infid-decomp` (làm mượt khối) → `quad-check` (C3c; dùng lại cache lưới lồng
nhau, chỉ tính thêm τ²_D) → `sampling-infid-pilot` → điền δ₁–δ₃, xác nhận K_main / n_images → `infid-lock` → E1a / E1b / E2.
