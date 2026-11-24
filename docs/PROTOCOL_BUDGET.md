# PROTOCOL BUDGET — tối ưu chiến lược lấy mẫu SPI theo J ở ngân sách gradient cố định

Nghiên cứu mới, tách khỏi nghiên cứu infid trước (khoá `infid_prereg.json`, `FROZEN.json` giữ nguyên). Mã: `probe/budget_opt.py`,
cấu hình: mục `budget:` trong `config.yaml`, khoá: `runs_probe/pilot/infid/budget/budget_prereg.json`.

## 1. Câu hỏi và objective

Với ngân sách B lần gọi gradient cho mỗi ảnh, chiến lược θ = (họ đường, α, N, G, coupling, K) nào cho attribution có infidelity kỳ vọng nhỏ nhất?

    J_D(θ, K) = E‖P_D Â_K − ΔF_D‖² / ‖ΔF_D‖² = L_D(μ_θ) + V_K(θ),     B = K · N · M
    V_K = (V_e + V_o) / K      (indep: K đường độc lập)
    V_K = 2 V_e / K            (rev: K/2 cặp (Z, RZ) độc lập, K chẵn)

J ở K hữu hạn là đại lượng người dùng thực sự nhận; L_D(μ) là giới hạn K → ∞. Báo cáo tách hai đại lượng này.

## 2. Thiết kế

| giai đoạn | lệnh | ảnh | nội dung |
|---|---|---|---|
| G1 | `budget-dev1` | dev: fresh 700–731 (32) | spi N = 64, α ∈ {0,1; 0,3; 1; 10}, m = 32 cặp; coupling ∈ {indep, rev} và K suy từ cùng các cặp |
| G2 | `budget-dev2` | cùng dev | chỉ top-3 α của G1; N ∈ {32, 128} (spi), patch G ∈ {7, 14}, hier G ∈ {7, 14} η = 0,5 (N = 64); m = 16 |
| khoá | `budget-lock` | — | băm protocol + D + cấu hình test + sha256 của `selection_dev2.json` |
| G3 + G4 | `budget-test` | test: fresh 800–847 (48) | θ*(B) cho từng B chính, base SPI α = 10 N = 64, α = 0,1 N = 64 (đã xác nhận), IG N = 64 cùng ngân sách; 8 D; metric ngoài |

Ảnh dev / test được kiểm tra rời nhau và rời mọi ảnh của nghiên cứu infid trước (pilot 0–19; fresh 400–447, 500–531, 600–631).
Luồng ngẫu nhiên riêng (`BUD-dev1`, `BUD-dev2`, `BUD-test`, `BUD-floor`).

Ngân sách: B chính {256, 1024} (K = 4, 16 ở N = 64), B phụ {128, 512, 2048} (K = 2, 8, 32). Với N khác, K = B / (N·M); tổ hợp không nguyên
hoặc rev với K lẻ bị loại. M = 1, quy tắc `rand` (giữ từ nghiên cứu trước).

## 3. Lựa chọn trên dev (chỉ bộ ô CHỌN, D chính `del_sq`)

- θ*(B) = argmin TB J theo ảnh. Báo xác suất chọn bootstrap theo ảnh cho mọi ứng viên.
- **Quy tắc dừng G2 (đặt trước):** một chiều (N, họ patch, họ hier) chỉ được vào tập ứng viên cuối ở B nếu nghiệm tốt nhất của chiều đó
  giảm TB J ≥ 5 % so với nghiệm G1 **và** P_boot(J chiều < J G1) ≥ 0,8. Chiều không đạt bị dừng, không mở thêm (không G khác, không η khác).
- Lựa chọn được phép sửa cho tới khi `budget-test` bắt đầu; sau đó `selection_dev2.json` bị khoá cứng (mã từ chối ghi đè).
- Bộ ô ĐÁNH GIÁ trên dev chỉ để mô tả.

## 4. Tiêu chí chính (test, bộ ô ĐÁNH GIÁ, D chính)

    R(B) = [J̄(base, B) − J̄(θ*, B)] / [J̄(base, B) − sàn̄]          (tỷ số các trung bình theo ảnh)

- base = SPI α = 10, N = 64, `indep` ở cùng B (cách dùng SPI phổ biến).
- sàn = L*_comp patch 14×14 out-of-sample (fit trên bộ ô chọn, đánh giá trên bộ ô đánh giá). Sàn patch ≥ sàn pixel và out-of-sample
  ≥ in-sample ⇒ mẫu số nhỏ hơn thật ⇒ R hơi lạc quan; ghi rõ trong báo cáo.
- r₀ = 0,20. p một phía bootstrap cho H0: R ≤ r₀; Holm qua 2 B chính.
  **xác nhận** nếu p Holm < 0,05; **bác bỏ** nếu cận trên CI 95 % của R < r₀; còn lại **chưa kết luận**.
- Báo kèm ΔJ = J base − J θ* (CI theo ảnh), số ảnh θ* tốt hơn, và R trung bình theo ảnh (độ nhạy).

## 5. Phân tích phụ (mô tả, không thay đổi tiêu chí chính)

1. θ* so với: base `rev`, α = 0,1 rev / indep, IG cùng ngân sách (J_IG = L̂(μ_IG) + σ̂²_IG / K, r = 32 lần lặp rand).
2. Độ bền theo 8 D (deletion / insertion / symmetric / fraction 25–50–75 / nền blur / superpixel): base − θ* cho **J ở B** và cho
   **L_D(μ)** riêng; Holm qua D.
3. Kiểm suy luận phương sai: J thực nghiệm của ensemble thật K đường (nhóm rời từ các đường đã lấy; K ∈ {4, 16}) so với L + V_K;
   CI chênh theo ảnh phải chứa 0.
4. G4 metric ngoài (InsDel nền blur, ROAD MoRF/LeRF, IoU, pointing, AUROC/AP mask) trên bản đồ thực tế ở K của ngân sách và trên Ē
   của toàn bộ 2m đường; Holm trong mỗi cặp so sánh. **Không bao giờ dùng metric ngoài để chọn lại θ.** Khác biệt chỉ ở mức ngân sách
   mà biến mất ở Ē ⇒ do phương sai hữu hạn K.

## 6. Trần GPU (đặt trước)

`budget.gpu_minutes`: dev1 240, dev2 180, test 240 phút (ước tính từ số gradient × thời gian một lượt đo bằng `bench`). Ước tính vượt trần
⇒ dừng trước khi chạy; nâng trần phải ghi lại ở đây kèm lý do trước khi chạy tiếp.

Ước tính gradient / ảnh (N = 64, M = 1): G1 4 × 64 × 64 = 16 384; G2 ≈ 3 × (32·32 + 32·128 + 4 × 32·64) ≈ 39 900;
test ≤ 4 × 64 × 64 + 32 × 64 ≈ 18 400 (+ metric ngoài, chỉ forward).

## 7. Ngoài phạm vi

nbr / global coupling, E3, ViT — chưa ưu tiên. Không chạy lưới mới trên test; không hồi tố tiêu chí cho nghiên cứu trước.
