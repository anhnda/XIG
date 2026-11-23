# Sửa đổi protocol 2 — sau E1a pilot, E0 làm mượt, C3c (24/11/2026)

## Căn cứ (kết quả pilot, chưa phải kết luận xác nhận)

- **E1a pilot (α = 10):** C1 (cặp vs trực tiếp 30,7 vs 30,1), C2, C1c đạt. V̂_e = 15,5, V̂_o = 15,2 ⇒ ΔV_D(4) ≈ −0,08: `rev` gần như
  không giúp ở cấu hình này. Đây là kết quả âm tính CỦA PILOT; không suy rộng sang α khác hay D khác.
- **Sửa trình bày:** 0,964 là TB các ρ̂_i theo ảnh; 2·TB(V̂_e)/TB(V̂_e+V̂_o) = 1,010. Hai estimand khác nhau — báo cáo nay ghi cả hai
  (`rho_mean_img`, `rho_ratio_of_means`). Tương tự cho làm mượt: 16/29/39% là giảm hình học theo ảnh, 9,6/17,6/31,8% là giảm theo
  trung bình — báo cáo ghi cả hai.
- **E0 làm mượt:** L̂(S_h μ) giảm nhưng vẫn cách sàn patch ≈ 20 ⇒ nhiễu tần số cao không giải thích phần lớn sai lệch. Làm mượt đổi
  attribution gần biên mask, nên không tự chứng minh lỗi nằm ở một dải tần số.
- **C3c:** τ²/σ̂² ≈ 0,27–0,56 (4 ảnh); kiểm lặp lại = 0. L_debiased (một đường) giảm mạnh khi α nhỏ — nhưng
  L_debiased = L_D(μ_q) + V_geometry(q), nên thứ hạng của nó KHÔNG phải thứ hạng của L_D(μ_q). Shapley là nghiệm LS dưới trọng số
  SHAP kernel, không phải nghiệm LS tối ưu cho D xoá bất kỳ ⇒ "α nhỏ gần Shapley" không tự chứng minh gần sàn của D.

## Quyết định

| Tham số | Giá trị |
| --- | --- |
| Quadrature | `rand`, M = 1 |
| K chính RQ1 / K phụ | 4 / {2, 8, 16} |
| δ₁ | 0,75 (mức cải thiện tuyệt đối tối thiểu). Kết luận "không đạt δ₁" CHỈ khi cận trên CI < 0,75 |
| δ₃ | 2,4, so với SPI baseline trên ảnh test chưa dùng để chọn α (H3 chính) |
| IG | đối chứng RIÊNG (H3b, mô tả), không gộp vào kết luận H3 |
| δ₂ | chưa khoá; phải chọn theo ý nghĩa thực tiễn, độc lập với pilot. `infid-lock` chỉ đòi δ₂ khi `e1a.coupled` bật |
| E1a | 32 ảnh mới, m = 16 (mục tiêu ban đầu; có thể tăng sau power analysis); cấu hình α = 10 (chính), α = 1 (phụ) |
| Endpoint | RQ1: ΔV_D(4); RQ2: L_D(μ_q); 1 − ρ̂_D là chỉ số phụ theo ảnh |

## Triển khai

- **(a) E0 độ nhạy theo D:** `e0.extra_sets = [ins_sq, sym_sq, frac50]` cùng ảnh và attribution, tập fit / đánh giá độc lập.
  Cần forward cho ΔF_S của D mới (không cần gradient).
- **(b) Chẩn đoán nghiệm sàn:** C_cancel = ‖a‖₁/(|Σa|+ε), r_neg, ‖a‖₁/|ΔF|, max|a|/|ΔF|, max|a|/mean|a| cho nghiệm LS (free, comp)
  và SPI, theo từng lưới patch và mức pixel.
- **(c) Thiết kế ghép ở pilot, giới hạn:** `nbr`, K = 4, 8 bản sao, α ∈ {10, 1}; kiểm phân phối biên (KS tại nút N/4, N/2, 3N/4)
  và tỷ số kiểm kỳ vọng TRƯỚC khi đọc; mục đích là ước lượng phương sai hiệu ứng cho power, không chọn δ₂.
- **(d) E1b vòng 1:** chỉ họ spi, α ∈ {0,1; 0,3; 1; 10}, N = 64, cùng ảnh dev; báo cáo CI của L̂_D(μ_q), phương sai đường (q90) và
  ngân sách gradient để đạt cùng độ chính xác. Mở N ∈ {32, 128} chỉ sau vòng 1. Được chọn lại trên dev cho tới khi E1b-test bắt đầu
  (sau đó lựa chọn bị khoá).
- **Cache:** hash tính toán của `sampling-infid` nay chỉ gồm tham số ảnh hưởng kết quả (rule, M, Ks, n_flip, sàn, quad_probe) — điền δ
  không làm mất cache. Đổi công thức hash ⇒ pilot tính lại MỘT lần.

## Câu hỏi quyết định của RQ2

α nhỏ có thực sự làm giảm L_D(μ_q) trên ảnh mới, hay chỉ làm giảm phương sai của đường đơn?

## Ưu tiên

1. RQ2 — α nhỏ và attribution kỳ vọng (E1b vòng 1 → test).
2. Khả năng giải thích của nghiệm sàn và độ nhạy theo D (E0 mục 3c, 3d).
3. RQ1 — xác nhận âm tính của `rev` ở α = 10 với đủ power; thăm dò α nhỏ và `nbr`.
