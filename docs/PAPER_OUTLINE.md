# PAPER_OUTLINE — Stochastic feature-wise path distributions

**Tiêu đề làm việc:** *Which properties of a stochastic path distribution matter? Theory and pre-registered experiments for feature-wise path
attribution at a fixed gradient budget.*

**Câu hỏi trung tâm:** Những đặc tính nào của phân phối đường ngẫu nhiên theo feature quyết định attribution kỳ vọng, phương sai, và do đó
chất lượng ở ngân sách gradient cố định? Dirichlet chỉ xuất hiện như một thể hiện (incumbent), không phải chủ đề.

Nhãn: **[T]** định lý/mệnh đề (chứng minh ở phụ lục) · **[E-xác nhận]** thí nghiệm đăng ký trước · **[E-mô tả]** phân tích mô tả đặt trước ·
**[E-hậu kiểm]** chẩn đoán sau test · **[L]** giới hạn. Mọi tuyên bố truy về `PAPER_CLAIMS_AND_LIMITATIONS.md`.

---

## 1. Introduction
- Bối cảnh: IG một đường; các biến thể ngẫu nhiên theo feature (SPI/Dirichlet và họ khác) — đặt bài toán ở mức **phân phối Q của lịch trình**.
- Câu hỏi: thống kê nào của Q quan trọng; với ngân sách B = K·N gradient, nên phân bổ thế nào.
- Đóng góp (ba, tách bạch):
  1. **[T]** Lý thuyết ở mức phân phối: thống kê đủ (moment Stieltjes), vai trò của bất biến khi đảo, exploration D, hàm hiệp phương sai $c$,
     giới hạn của D (bậc 5), luật kích hoạt → Shapley; đồng nhất thức ngân sách J và phân rã thang đo/hướng.
  2. **[E-xác nhận]** Tối ưu ở ngân sách cố định (BUDGET) và không-kém-hơn của hai họ không-Dirichlet (PATHFAM).
  3. **[E-hậu kiểm]** Chẩn đoán: J thô phụ thuộc thang đo; metric có dấu/không dấu; completeness ở K hữu hạn.
- Thông điệp: Dirichlet là một lựa chọn thiết kế; điều quan trọng là bất biến khi đảo, mức exploration và cấu trúc hiệp phương sai — trong phạm vi một mô hình và một objective.

## 2. Setup
- 2.1 Lịch trình, đường, attribution Stieltjes; μ_Q. **[T]** Bổ đề completeness theo từng đường.
- 2.2 Họ đường như thể hiện của Q: DP lưới, logistic-normal, coordinate-mix, smooth, nhóm (patch/hier). Bảng tính chất: độc lập, bất biến khi đảo,
  E Z = t, D, $c(s,t)$, độ nhọn, chi phí N gradient/đường. (Không xếp Dirichlet riêng.)
- 2.3 Ước lượng: lưới N, cầu phương `rand` M = 1, thiết kế `indep` / `rev`. **[T]** B5: không chệch cho luật nội suy.
- 2.4 Objective: infidelity theo phép xoá $L_{\mathcal D}$, $J=\mathbb E L(\bar A_K)$. **[T]** T6.2 đồng nhất thức chính xác; M6.4 thứ hạng theo B đổi tối đa một lần.
- 2.5 **[T]** M6.5: $L=(r-\cos)^2+\sin^2$ — đặt ngay ở đây để người đọc biết J thô đo gì. **[L]** L1.

## 3. Theory: what the path distribution controls
- 3.1 **[T]** Định lý 1 (thống kê đủ, cần và đủ).
- 3.2 **[T]** Định lý 2, H2.2, Đ2.3: bậc 2 — diện tích Lévy; bất biến khi đảo ⇒ IG (không cần độc lập); `rev` chính xác ở bậc 2.
- 3.3 **[T]** Định lý 3, H3.1, H3.2: độc lập ⇒ moment biên; bậc 3–4 chỉ qua D; **[T]** M3.3: bậc 5 D không đủ (phản ví dụ giải tích).
- 3.4 **[T]** Định lý 4: hiệp phương sai bậc 2 qua $c$; bảng DP/cmix/smooth; H4.1–4.2 (cùng D ≠ cùng phương sai).
- 3.5 **[T]** M3.4: luật kích hoạt rời nhau → Shapley (theo khối trên lưới) — nối "exploration" với "tương tác".
- 3.6 **[T]** H6.3 + M6.4: ngân sách — lệch vs phương sai × chi phí; `rev` loại phần bậc 2 của phương sai đường.
- 3.7 Bảng tổng hợp giả thiết → kết luận (THEORY §8). **[L]** L5: đa thức/trơn vs ReLU.
- Kiểm số (phụ lục): sympy cho mọi mẫu chỉ số; MC với tích phân chính xác.

## 4. Experimental protocol
- Mô hình, dữ liệu, nền, tập phép xoá (chọn/đánh giá rời), dev/test rời, luồng ngẫu nhiên riêng, khoá sha256, trần GPU.
- Thống kê: bootstrap theo ảnh, bootstrap-t trên hiệu tuyến tính hoá, Holm, NI/superiority/TOST, lực trên dev; hiệu chuẩn CI bằng mô phỏng (CPU).
- Nguyên tắc: metric ngoài không bao giờ dùng để chọn; chẩn đoán sau test không đổi endpoint.
- **[L]** L2–L4, L7.

## 5. Experiment 1 — Budget allocation (BUDGET) **[E-xác nhận]**
- Chọn θ*(B) trên dev; R(B) trên test (B = 256, 1024). Bảng C1.
- **[E-mô tả]** so với α = 0,1, IG cùng ngân sách, 8 tập D; phân rã L(μ) vs V/K.
- **[E-mô tả]** BUDGET2 (kết quả âm, dev-only): mở rộng N/patch/hier không qua cổng; cross-fit.
- Diễn giải theo M6.5: cải thiện J thô gồm phần thang đo. **[L]** L1.

## 6. Experiment 2 — Is Dirichlet necessary? (PATHFAM) **[E-xác nhận]**
- Họ ln, cmix khớp D với DP; chọn bằng J_rev trên dev; NI so với incumbent cố định (B = 256, 1024, Holm qua 4). Kết quả C2 (số từ báo cáo gốc).
- **[E-mô tả]** superiority/TOST (phụ), J_indep riêng, phân rã L(μ), V₁, 2V_e, ρ theo B (128…4096), smooth (không đạt α = 1 khi đơn điệu), đối chứng α = 10 N = 64.
- **[E-mô tả]** đối chiếu với lý thuyết: cùng h (H3.2, H4.1), smooth vs DP cùng D (H4.2), ρ (H6.3) — khớp hay không đều báo.
- Thông điệp: các thống kê μ bậc thấp (bất biến khi đảo, D) được khớp ⇒ hiệu năng giữ được mà không cần Dirichlet. **[L]** L5, L6.

## 7. Diagnostics: what J does and does not measure **[E-hậu kiểm]**
- 7.1 Thang đo và hướng: L ở β = 1 là 11,7–47, ≥ 91 % là lệch thang đo (tổng theo ô lớn gấp 2,5–4,3 lần hiệu ứng xoá); β riêng ảnh (chẩn đoán,
  không triển khai được) cho 1 − cos² = 0,81–0,89; β chung ngoài mẫu (triển khai được) chỉ 0,965–0,994 so với 1 của bản đồ 0 ⇒ mô hình cộng tính
  theo ô giải thích ít hiệu ứng xoá; cách biệt J thô giữa cấu hình phần lớn là thang đo. **[L]** L1.
- 7.2 Có dấu vs không dấu: định nghĩa metric, lệch nền InsDel; chỉ đọc nền đen; J ↔ metric ngoài đo hai tính chất khác nhau. **[L]** L8.
- 7.3 Completeness ở K hữu hạn: **[T]** M7.1–7.3 (chỉ nhiễu cầu phương; MSE chính xác; luật $K^{-1/2}$ tiệm cận, cận Wasserstein); dữ liệu: sai số có dấu
  không chệch (CI chứa 0 ở mọi cấu hình tại K = 32); độ dốc tuyệt đối −0,29…−0,52 và phân rã RMS + log G (đuôi nặng ở một số cấu hình, profile RMS ở số khác),
  $\overline{b^2}$ không phân biệt được với 0, mô phỏng ở đúng thiết kế. Không tuyên bố lỗi cài đặt. **[L]** L9.

## 8. Related work
- Path attribution (IG và biến thể đường, Shapley theo đường/Aumann–Shapley), ước lượng Monte Carlo cho attribution, infidelity/sensitivity-n,
  đánh giá có dấu vs không dấu, path signatures (chữ ký) như công cụ.

## 9. Limitations (một mục riêng, đầy đủ)
- L1–L10 của `PAPER_CLAIMS_AND_LIMITATIONS.md`, đặc biệt: một mô hình/tập dữ liệu/nền; J thô phụ thuộc thang đo; lý thuyết đa thức vs ReLU;
  nhiều phân tích hậu kiểm trên cùng ảnh test; không có kết quả tối ưu.

## 10. Conclusion
- Cấp phân phối là mức đúng để thiết kế path attribution ngẫu nhiên; bất biến khi đảo + exploration + hiệp phương sai quyết định phần lớn μ và
  phương sai bậc thấp; Dirichlet là một thể hiện tiện lợi, không cần thiết.
- Hướng tiếp: mục tiêu bất biến thang đo (hoặc β ngoài mẫu như endpoint đăng ký trước), mô hình khác (ViT), lý thuyết cho mạng tuyến tính từng mảnh,
  thiết kế Q theo $c(s,t)$ và đuôi.

## Phụ lục
- A. Chứng minh đầy đủ (THEORY §3–§7).
- B. Kiểm số: `pathfam-theory`, `pathfam-proofcheck`, `pathfam-final-audit` §2a/§2d; danh sách test.
- C. Protocol BUDGET, BUDGET2 (quyết định §7–8), PATHFAM (§1–10), metric audit, khoá chẩn đoán.
- D. Thống kê: bootstrap-t tuyến tính hoá, Fieller, hiệu chuẩn CI bằng mô phỏng, Student t thuần Python.
- E. Bảng số đầy đủ (từ báo cáo gốc), chi phí gradient đo được.

## Bản đồ hình / bảng chính
| # | Nội dung | Loại |
|---|---|---|
| Hình 1 | Sơ đồ: Q → (moment Stieltjes, 𝔄, D, c, đuôi) → μ, Cov → J ở B | [T] |
| Bảng 1 | Tính chất các họ (độc lập, đảo, D, Γ, Λ, độ nhọn, chi phí) | [T] + kiểm số |
| Bảng 2 | Giả thiết → kết luận | [T] |
| Bảng 3 | BUDGET R(B) | [E-xác nhận] |
| Bảng 4 | PATHFAM NI (g, CI, p Holm) | [E-xác nhận] |
| Hình 2 | J_rev theo B cho các họ (L(μ) + 2V_e/K), điểm đổi thứ hạng | [E-mô tả] + M6.4 |
| Hình 3 | Phân rã $(r-\cos)^2$ / $\sin^2$; L ở β = 1, β chung ngoài mẫu, β riêng ảnh | [E-hậu kiểm] |
| Hình 4 | Completeness: TB\|err\|, RMS theo K; G_K | [E-hậu kiểm] + M7.3 |
