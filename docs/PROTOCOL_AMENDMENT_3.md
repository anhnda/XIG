# Sửa đổi protocol 3 — ưu tiên RQ2 (24/11/2026)

## Căn cứ (pilot, 8 ảnh — exploratory)

- L̂_D(μ_q) trên `del_sq`: α = 10 → 25,1; α = 1 → 10,5 (giảm ≈ 58% theo trung bình), cùng ảnh, cùng ô, đã khử phương sai lấy mẫu.
  Chưa có CI của chênh lệch theo cặp ảnh. Phương sai đường ở α = 1 còn bất định (cặp 26,5 vs trực tiếp 36,9, CI chênh rộng).
- `rev` không có lợi ích đáng kể ở α = 10 (pilot). `nbr`: dấu bất lợi, sd theo ảnh ≈ 6 ⇒ H2 không khả thi về power.
- E0: L̂(μ) 23,85 (xoá) → 5,51 (phân số) → 1,45 (đối xứng) → 0,99 (chèn). Sai lệch được KHUẾCH ĐẠI mạnh bởi D kiểu xoá; chèn vẫn
  còn khoảng cách đáng kể tới sàn.

## Quyết định

- Thứ tự ưu tiên chính thức: **RQ2 trước, RQ1 sau, H2 hoãn** (`e1a.coupled: false`, δ₂ = null).
- Khoá tiếp: `rand` M = 1; RQ1 chính α = 10, K = 4; E1a 32 ảnh, m = 16; δ₁ = 0,75; H3 chính trên `del_sq` so với SPI baseline,
  δ₃ = 2,4. 32 ảnh đủ để kiểm ngưỡng δ₁ theo dự báo pilot nhưng KHÔNG bảo đảm power cho hiệu ứng nhỏ hơn δ₁: CI không đủ hẹp ⇒
  "chưa xác định", không phải bác bỏ.
- E1b vòng 1: α ∈ {0,1; 0,3; 1; 3; 10}, N = 64, cùng ảnh và D; D chính `del_sq`, phụ `ins_sq`, `sym_sq`. Chưa mở N = 32 / 128.
- m của E1b giữ 16; quyết định m = 32 sau khi pilot có α = 0,1 và 0,3 (độ rộng CI, độ nhạy với ảnh ngoại lai).
- Hoãn floor a ≥ 0 (không khả thi khi ΔF < 0; khối âm có thể hợp lý). Nếu cần floor có ràng buộc diễn giải: completeness +
  phạt ℓ1 / mức triệt tiêu.

## Triển khai

- **(i)** Báo cáo so sánh theo cặp ảnh (pilot, E1b dev): d_i = L_i(μ_c) − L_i(μ_ref), r_i = log(L_i(μ_c)/L_i(μ_ref)); TB [CI], trung vị,
  số ảnh d_i < 0, leave-one-out (min / max, dấu có ổn định), ảnh có |d| lớn nhất; số ảnh có L ≤ 0 bị kẹp khi lấy log.
- **(ii)** Mọi bảng tách theo D; thêm bảng "đánh đổi giữa các D" (TB log-tỷ số theo ảnh, mỗi cấu hình × D).
- **(iii)** L_D(μ) = E_D / C_D với C_D = E_D[ΔF_S²], E_D = E_D[(p_Sᵀμ − ΔF_S)²] (đã khử phương sai), cùng rms_ratio (độ lớn dự đoán /
  độ lớn ΔF), beta_opt, cos — theo từng D và từng ảnh (E0 và pairs.csv). E0 thêm phân rã log(L_del/L_D) = log(E_del/E_D) − log(C_del/C_D).
  Giả thuyết bão hoà vẫn là giả thuyết cơ chế.

## Câu hỏi cần trả lời trước khi chọn cấu hình kiểm định cuối

Bằng chứng theo cặp ảnh cho thấy α nhỏ cải thiện L_D(μ_q) không; cải thiện có ổn định (leave-one-out) không; có đánh đổi giữa
xoá và chèn không.
