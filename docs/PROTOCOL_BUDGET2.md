# PROTOCOL BUDGET2 — vượt incumbent ở cùng ngân sách gradient

Vòng 2 của nghiên cứu BUDGET. Không đổi objective: J_D(θ, B) = L_D(μ_θ) + V_K(θ), B = K·N·M, D chính `del_sq`.
Mã: `probe/budget2.py`; cấu hình: mục `budget2:`; khoá: `runs_probe/pilot/infid/budget2/budget2_prereg.json`.
Kết quả BUDGET 1 (khoá `9360c3c40efe`) giữ nguyên; kiểm toán riêng bằng `budget-audit` (mục 0).

## 0. Trước vòng mới: kiểm toán BUDGET 1 (`budget-audit`, không GPU)

1. Phân phối xác suất chọn **đầy đủ** trên dev1 / dev2, tái lập từ CSV; ảnh bị rút nhiều hơn trong các bootstrap mà θ* thua;
   bỏ-một-ảnh.
2. Hiệu chuẩn phép kiểm R ≥ r₀. Tuyến tính hoá chính xác: khi E[b − f] > 0, R > r₀ ⇔ E[D] > 0 với
   D_i = (b_i − t_i) − r₀(b_i − f_i). Báo p đã đăng ký (percentile), basic bootstrap, t một phía trên D, bootstrap-t trên D,
   CI percentile / BCa / Fieller, bỏ-một-ảnh. Quyết định đã đăng ký không đổi; nếu chỉ percentile đạt, B đó được ghi
   "xác nhận theo protocol, nhạy với phương pháp hiệu chuẩn".
3. Kiểm J thực nghiệm: sai khác tuyệt đối, tương đối theo J và theo V_K, skew, tỷ lệ ảnh dương, Holm trong giai đoạn,
   lặp lại trên hai nửa ảnh rời nhau. Kèm chứng minh đại số rằng hai ước lượng cùng không chệch.

## 1. Thiết kế

| | Nhánh A (chính) | Nhánh B (dev + cổng) |
|---|---|---|
| Ngân sách | B = 256 (chính), 1024 (phụ) | B = 2048, 4096 |
| Ứng viên | SPI α ∈ {0,3; 1; 3} × N ∈ {8, 16, 32}, `rev` | hier G = 14, α ∈ {0,1; 0,3}, η ∈ {0,25; 0,5; 0,75}, N = 64, `rev` |
| Tham chiếu | **incumbent** SPI α = 1, N = 32, `rev` | SPI `rev` tốt nhất trên dev ở cùng B (incumbent, α = 0,1 N = 64, lưới nhánh A) |
| K ở B = 256 | 32 / 16 / 8 (N = 8 / 16 / 32) | — |
| K ở B = 1024 | 128 / 64 / 32 | — |

Đối chứng phụ: SPI α = 0,1, N = 64, `rev` (nghiên cứu trước). `rev` là coupling chính; J `indep` của cùng α, N, K được báo kèm để
không gán lợi ích của α / N cho coupling. Báo đồng thời L_D(μ), phương sai một đường V₁ = V_e + V_o, 2V_e, ρ và J.

Ảnh: dev fresh 900–931 (32 ảnh, m = 32 cặp); test fresh từ 1000 (n tính lúc khoá). Mã kiểm rời với mọi ảnh infid trước và BUDGET 1.
Luồng ngẫu nhiên `BUD-r2dev`, `BUD-r2test`.

## 2. Cổng dev (bộ ô CHỌN)

Ở mỗi B: ứng viên tốt nhất (TB J) so với tham chiếu; qua cổng nếu giảm TB J ≥ 5 % **và** P_boot(tốt hơn, theo cặp ảnh) ≥ 0,8.
- Không qua ở B ⇒ incumbent giữ nguyên ở B đó; không test B đó.
- Không B nào trong `test.budgets` qua ⇒ dừng, không khoá, không test.
- Nhánh B chỉ được đưa vào `test.budgets` sau khi qua cổng **và** được duyệt riêng (mặc định `test.budgets: [256, 1024]`).
- Lựa chọn được phép chạy lại cho tới khi test bắt đầu; sau đó `selection_dev.json` bị khoá cứng.

## 3. Cỡ mẫu test (tính lúc `budget2-lock`, trước khi có ảnh test)

Trên bộ ô ĐÁNH GIÁ của dev: D_i = (J_ref − J_new) − δ·J_ref, δ = 5 %. Hiệu ứng giả định g* = (g_dev + δ)/2 (co về ngưỡng để trừ lời
nguyền người thắng); hiệu ứng trên thang D = (g* − δ)·J̄_ref. n = ⌈((z₀.₉₅ + z₀.₈) · SD(D) / hiệu ứng)²⌉, lấy max qua các B được kiểm,
kẹp vào [48, 128]. Nếu n cần > 128: khoá với 128 và báo "thiếu lực".

## 4. Phép kiểm chính (test, bộ ô ĐÁNH GIÁ)

Với mỗi B được kiểm: g = (J̄_ref − J̄_new)/J̄_ref; H0: g ≤ δ ⇔ E[D] ≤ 0.
- p một phía **bootstrap-t** trên D (chính; thuần numpy, tái lập theo seed); t một phía (độ nhạy).
- Holm qua các B được kiểm (B = 256 được gọi là chính nhưng vẫn vào Holm — thận trọng).
- **xác nhận** nếu p Holm < 0,05; **bác bỏ** nếu cận trên CI percentile 95 % của g < δ; còn lại **chưa kết luận**.
- Báo kèm CI Fieller của g, ΔJ [CI], số ảnh thắng.

## 5. Phân tích phụ (mô tả)

So với α = 0,1 và incumbent ở cùng B; độ bền theo 8 D cho J ở B và cho L_D(μ) (Holm qua D); kiểm J thực nghiệm; metric ngoài
(InsDel blur, ROAD, IoU, pointing, AUROC / AP mask) ở K của ngân sách và trên Ē của 2m đường. **Metric ngoài không dùng để chọn.**

## 6. Trần GPU

`budget2.gpu_minutes`: dev 300, test 300 phút. Ước tính dev ≈ 39 400 gradient / ảnh (nhánh A 10 752, α = 0,1 N = 64 4 096,
nhánh B 24 576) × 32 ảnh.

## 7. Kết quả dev và quyết định (ghi sau khi xem dev, trước mọi ảnh test)

| B | nhánh | ứng viên tốt nhất | giảm J (bộ chọn) | P_boot(tốt hơn) | cổng |
|---|---|---|---|---|---|
| 256 | A | SPI α = 0,3, N = 32 | −10,6 % | 0,231 | không qua |
| 1024 | A | SPI α = 0,1, N = 64 | +2,2 % | 0,615 | không qua |
| 2048 | B | hier α = 0,3, G = 14, η = 0,75 | +3,2 % | 0,625 | không qua |
| 4096 | B | hier α = 0,1, G = 14, η = 0,75 | +6,8 % | 0,807 | qua (sát ngưỡng) |

Quyết định (theo phản biện):
1. Incumbent SPI α = 1, N = 32, `rev` giữ nguyên ở B = 256 và 1024. Nhánh A đóng, không test.
2. Nhánh B ở B = 4096 là **ứng viên thăm dò**, không phải nghiệm mới. Không đổi ngưỡng δ = 5 % sau khi xem dữ liệu.
3. Kiểm toán cổng (`budget2-audit`): tách P(g > 0) khỏi P(g > δ), CI của ΔJ / g, cross-fit theo ảnh để trừ hiệu ứng chọn;
   kiểm J thực nghiệm + mô phỏng độ phủ của chính quy trình CI.
4. Đóng băng BUDGET2 dev-only (`budget2-freeze`); tập test fresh 1000– được giữ lại. Mở lại chỉ khi có ứng viên mạnh hơn và quyết định
   mới được ghi vào đây trước.

## 8. Sau kiểm toán (budget2-audit lần 1) — quyết định và hiệu chuẩn CI

Cross-fit: g ở B = 4096 còn ≈ +2,0 % (P(g > 0) ≈ 0,61); B = 2048 ≈ +2,4 %; nhánh A âm. Trên bộ đánh giá, B = 4096: g = 5,2 %
[−10,0 %; 17,0 %], p bootstrap-t 0,216 (g > 0) và 0,463 (g > δ). Quyết định: incumbent giữ nguyên; không test nhánh A; hoãn test
hier B = 4096; không mở lưới GPU mới. Ứng viên B = 4096 đã qua cổng sàng lọc — ghi nguyên sự thật này, nhưng không gọi là nghiệm mới.

Hiệu chuẩn CI của kiểm J (CPU, `budget2-audit` mục 3): mô phỏng 7 kịch bản (đuôi trong ảnh gauss / exp / t5 / t3; số chiều hiệu
dụng của nhiễu r ∈ {64, 16, 4, 1}; thang đo giữa ảnh lognormal hoặc lấy lại từ √V thực; ρ = trung vị dữ liệu), 2000 lần lặp mỗi
kịch bản, so 4 phương pháp CI (percentile, BCa, t, bootstrap-t). Quy tắc chọn đặt trước khi áp lên dữ liệu thật: max của độ phủ nhỏ
nhất; hoà trong 1 SE Monte Carlo ⇒ độ rộng tương đối nhỏ hơn. Kiểm J trên dữ liệu thật (BUDGET2 dev, BUDGET 1) được làm lại với
phương pháp đó; số dòng loại 0 so với tỷ lệ trượt mô phỏng của chính phương pháp. p Student t tính thuần Python (không phụ thuộc
scipy) để tái lập trên mọi máy. Sau đó mới `budget2-freeze`.
