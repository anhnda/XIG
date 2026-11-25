# PAPER_CLAIMS_AND_LIMITATIONS — được tuyên bố gì, ở mức nào

Tài liệu này là ranh giới cho bản thảo: mọi câu trong paper phải truy được về một dòng ở đây. Ba mức bằng chứng, không trộn lẫn:

- **[XÁC NHẬN]** kiểm định đăng ký trước, protocol + lựa chọn khoá trước khi mở ảnh test, tập test chỉ dùng một lần cho kiểm định đó.
- **[ĐỊNH LÝ]** chứng minh trong `PATH_FAMILIES_THEORY.md` dưới giả thiết nêu rõ; kiểm số chỉ là kiểm tra đại số.
- **[HẬU KIỂM]** chẩn đoán mô tả trên ảnh test đã dùng (signdiag, signdiag2, final-audit) hoặc trên dev; không có giá trị suy luận xác nhận,
  không dùng để chọn lại, không đổi endpoint.

Số liệu xác nhận lấy từ báo cáo gốc: BUDGET — `INFID_BUDGET_REPORT.md` (mục G3); PATHFAM — `INFID_PATHFAM_REPORT.md` và bản tính lại độc lập
`pathfam/PATHFAM_VERIFY.md`; chẩn đoán — `pathfam/PATHFAM_FINAL_AUDIT.md`, `pathfam/DIAG_LOCK.json`. Tài liệu này không chép lại số PATHFAM;
khi viết paper, lấy trực tiếp từ hai file đó.

## 1. Kết quả xác nhận

### C1 — BUDGET [XÁC NHẬN]

Objective cố định $J_{\mathcal D}(\theta,B)=L_{\mathcal D}(\mu_\theta)+V/K$, $B=KNM$, $M=1$ (`rand`), $\mathcal D$ chính `del_sq` (xoá ô vuông 16/32/56 px về nền
đen; 512 ô chọn, 512 ô đánh giá độc lập). Cấu hình chọn trên dev: SPI α = 1, N = 32, `rev`. So với SPI α = 10, N = 64, `indep` cùng ngân sách,
48 ảnh test, tiêu chí $R(B)=[\bar J_{\rm base}-\bar J_{\theta^*}]/[\bar J_{\rm base}-\text{sàn}]$, $H_0:R\le0{,}20$, Holm qua 2 ngân sách:

| B | R [CI 95 %] | p Holm | ảnh θ* tốt hơn | quyết định |
|---|---|---|---|---|
| 256 (K = 8) | 0,308 [0,187; 0,415] | 0,039 | 41/48 | xác nhận |
| 1024 (K = 32) | 0,351 [0,243; 0,447] | 0,007 | 44/48 | xác nhận |

Phát biểu chính xác được phép: *ở ngân sách gradient cố định và theo objective đã khoá, cấu hình chọn trên dev giảm J thô một phần đáng kể
so với cách dùng SPI phổ biến.* Kèm theo, bắt buộc: sàn là patch 14×14 ngoài mẫu (≥ sàn pixel ⇒ R hơi lạc quan); J thô phụ thuộc thang
đo (§3, L1). Các so sánh phụ (so với α = 0,1, IG cùng ngân sách, 8 tập D) là mô tả; ở B = 1024 hiệu so với α = 0,1 `rev` có CI chứa 0.

### C2 — PATHFAM [XÁC NHẬN]

Kiểm định chính: không-kém-hơn (NI) của họ `ln` (logistic-normal) và `cmix` (coordinate-mix) so với incumbent cố định SPI α = 1, N = 32, `rev`,
so bằng **J_rev**, tại B = 256 và 1024, $H_0: g\le-0{,}10$ với $g=(\bar J_{\rm inc}-\bar J_F)/\bar J_{\rm inc}$, bootstrap-t trên hiệu tuyến tính hoá,
Holm qua 4. Kết quả: **NI được xác nhận cho cả ln và cmix** theo protocol đã khoá; tính lại độc lập (`pathfam-verify`) khớp.

Phát biểu được phép: *hai họ phân phối đường không dùng Dirichlet, chỉ khớp với incumbent ở thống kê exploration D, không kém hơn incumbent quá
10 % J_rev ở cả hai ngân sách.* Do đó Dirichlet là **một lựa chọn thiết kế, không phải điều kiện cần** cho stochastic feature-wise path
attribution hiệu quả theo objective này.
Không được phép: "tương đương" (NI ≠ TOST; xem dòng TOST riêng trong báo cáo), "tốt hơn" (superiority là phép phụ riêng), suy rộng sang
smooth (chỉ mô tả, không đạt vùng α = 1 khi đơn điệu), sang N khác, hay sang mô hình khác.

### C3 — BUDGET2 [kết quả âm, dev-only, đã đóng băng]

Không ứng viên nào ở B = 256 / 1024 vượt cổng sàng lọc đặt trước; ứng viên hier B = 4096 qua cổng sát ngưỡng nhưng cross-fit chỉ còn ≈ +2 %
và không được test. Incumbent giữ nguyên; tập test BUDGET2 chưa mở. Phát biểu được phép: *mở rộng lưới (N, patch, hier) không cho cải thiện
đủ lớn để biện minh kiểm định*. Không được trình bày hier B = 4096 như một nghiệm.

## 2. Kết quả lý thuyết [ĐỊNH LÝ]

Tóm tắt (chi tiết, giả thiết: `PATH_FAMILIES_THEORY.md` §8):
T1 μ chỉ phụ thuộc Q qua moment Stieltjes (cần và đủ cho đa thức). T2 bậc 2: diện tích Lévy; **bất biến khi đảo ⇒ μ = IG mà không cần độc lập**;
cặp `rev` chính xác ở bậc 2. T3 độc lập ⇒ moment biên; bậc 3 qua D; bậc 4 qua D khi bất biến khi đảo; **bậc 5: D không đủ** (phản ví dụ giải tích
13/360 so với 3/80 cùng D, cùng $c$). T4 hiệp phương sai bậc 2 qua hàm $c(s,t)$. M3.4 luật kích hoạt rời nhau → Shapley (theo khối trên lưới).
T6 đồng nhất thức J chính xác; M6.4 thứ hạng theo B đổi tối đa một lần; M6.5 phân rã thang đo / hướng. M7.1–7.3 completeness: chỉ đo nhiễu
cầu phương; MSE $=b^2+2\sigma^2/K$; luật $K^{-1/2}$ của sai số tuyệt đối chỉ tiệm cận.

Cách nối lý thuyết với C2 (diễn giải, không phải kiểm định): các thống kê quyết định μ tới bậc 4 (D, bất biến khi đảo) được khớp giữa ln/cmix
và DP; khác biệt chỉ có thể đến từ bậc ≥ 5 (đuôi) và phương sai bậc ≥ 3 — phù hợp với việc NI đạt được, nhưng mạng ReLU không phải đa thức (L5).

## 3. Chẩn đoán hậu kiểm [HẬU KIỂM]

**D1 — thang đo và hướng** (`PATHFAM_FINAL_AUDIT.md` §1; 12 cấu hình × 128 ảnh test PATHFAM đã khoá, bản đồ dựng lại bằng luồng `pfsd2`,
bộ ô đánh giá, `del_sq`). Theo M6.5, $L(a)=(r-\cos)^2+(1-\cos^2)$ với $r=\|Pa\|/\|\delta\|$; bản đồ 0 có $L=1$, mọi phần $L>1$ là lệch thang đo.

| đại lượng | bản đồ B = 1024 (K = 32; N = 64: K = 16) | bản đồ B = 256 (K = 8; N = 64: K = 4) |
|---|---|---|
| L ở β = 1 (đã khoá) | 11,7–18,6; smooth 37,7 | 15,9–25,1; spi α0,1 N64 31,2; smooth 46,9 |
| trung vị ‖Pa‖/‖δ‖ | 2,5–3,2×; smooth 3,9× | 2,8–3,5×; smooth 4,3× |
| trung vị cos(Pa, δ); tỷ lệ ảnh cos ≤ 0 | 0,23–0,37; 19–23 % | 0,20–0,32; 19–23 % |
| L ở β riêng từng ảnh = TB(1 − cos²) (chẩn đoán) | 0,808–0,874 | 0,843–0,894 |
| β chung (trong mẫu) | 0,021–0,056 | 0,016–0,039 |
| **L ở β chung ngoài mẫu theo ảnh (triển khai được)** | **0,965–0,985** | **0,975–0,994** (4/12 CI chứa 1) |

Hệ quả:
- **J thô phụ thuộc cả thang đo lẫn alignment, và bị thang đo chi phối**: ≥ 91 % (đến 98 %) của L ở β = 1 là số hạng $(r-\cos)^2$. Tổng theo ô của bản đồ
  chiếu lớn hơn hiệu ứng xoá 2,5–4,3 lần về chuẩn; phương sai ở K nhỏ làm chuẩn này lớn thêm (L ở B = 256 > B = 1024 cho mọi cấu hình,
  đúng như $\mathbb E\|P\bar A_K\|^2=\|P\mu\|^2+\operatorname{tr}P\operatorname{Cov}P^\top$).
- **β riêng từng ảnh** dùng $\Delta F_S$ của chính ảnh (kể cả bản cross-fit theo ô: 0,815–0,904) ⇒ là đại lượng chẩn đoán đo $1-\cos^2$; **không phải
  estimator triển khai độc lập**. Ngay cả với thang đo "oracle" này, bản đồ chỉ giải thích 11–19 % $\|\delta\|^2$.
- **β chung ngoài mẫu** (bỏ-một-ảnh) là hiệu chỉnh triển khai được duy nhất và **chỉ cải thiện nhẹ so với bản đồ 0** (0,6–3,5 %; ở B = 256 có 4/12
  cấu hình CI chạm 1). Sau phép đổi thang đo tốt nhất có thể triển khai, mô hình cộng tính theo ô gần như không dự đoán được hiệu ứng xoá trên `del_sq`.
  Đây là giới hạn của objective/D, không riêng họ đường nào.
- Ví dụ minh hoạ (cùng B = 1024, mô tả): incumbent spi α1 N32 so với spi α10 N64 — L ở β = 1: 13,19 so với 18,61 (cách biệt 5,4, cỡ ΔJ xác nhận của
  BUDGET); ở β riêng ảnh: 0,821 so với 0,858; ở β chung ngoài mẫu: 0,969 so với 0,976. **Phần lớn cách biệt J thô là thang đo.** C1/C2 vẫn đúng như
  phát biểu về objective đã khoá; không được diễn giải là "hướng attribution trung thực hơn".

**D2 — metric có dấu / không dấu.** Infidelity dùng tổng **có dấu** theo kênh trong ô; metric ngoài đã khoá xếp hạng theo $\sum_c|A_c|$ (không dấu).
Mâu thuẫn J ↔ metric ngoài không phải mâu thuẫn logic (đo hai tính chất khác nhau của cùng bản đồ). InsDel nền blur lệch nền với attribution nền
đen: `insdel_signed_blur` **bị loại** khỏi mọi diễn giải; chẩn đoán InsDel duy nhất được đọc là nền đen (`insdel_{abs,signed}_black`), cùng
`iou_*`, `energy_*` (DIAG_LOCK). Kết luận H-SU vòng 1 về InsDel bị loại.

**D3 — so sánh cùng exploration** (spi / ln / cmix cùng h, N = 32, cùng K): mô tả; khớp h không phải khớp J.

**D4 — completeness** (`PATHFAM_FINAL_AUDIT.md` §2b–2d; chi tiết `COMPLETENESS_SLOPE.md`). Sai số có dấu có CI chứa 0 ở cả 12 cấu hình tại K = 32
(55/60 ô cấu hình × K; 5 ngoại lệ ở K ≤ 16, 4 âm 1 dương, mỗi ngoại lệ ở một cấu hình khác nhau, P(≥ 5 | ngẫu nhiên) = 0,18) — cầu phương không chệch như M7.1–7.2. TB|err| giảm 2,4–4,2× từ
K = 2 đến 32 (lý thuyết 4×); sai số tương đối ở K = 32 là 6–11 %. Độ dốc TB|err| theo log K: −0,29 đến −0,52 (N = 64: chứa −½; 7/10 cấu hình N = 32:
phẳng hơn, CI loại −½). Phân rã chính xác RMS + log G: một phần do đuôi nặng ở K hữu hạn (N = 64, ln α4, spi α4), một phần ở profile RMS với hình
dạng gần chuẩn (cmix α0,3, cmix α1, ln α1, spi α0,3); độ lệch riêng ảnh b̂² có CI chứa 0 ở cả 12 cấu hình và giá trị điểm chỉ giải thích một phần nhỏ.
**Không gọi là lỗi cài đặt.** Luật $K^{-1/2}$ của sai số tuyệt đối được trình bày là tiệm cận. Completeness không nói gì về $V_1$ hay J (M7.1).

**D5 — đối chiếu với dự báo lý thuyết** ($\rho=2V_e/V_1$ so với H6.3; smooth so với DP cùng D theo H4.2; ln/cmix so với spi cùng h theo H3.2,
H4.1): báo như mô tả, dù khớp hay không khớp; khớp ≠ kiểm định (mạng thật không thoả (Poly$_p$)). Một quan sát mô tả từ D1: ở B = 1024, $1-\cos^2$ (β riêng ảnh) tăng đơn điệu theo α trong cả ba họ N = 32 — α = 0,3: 0,808–0,815; α = 1: 0,821–0,826;
α = 4: 0,832–0,838 — và cao hơn ở spi α = 10 N64 (0,858), smooth (0,874). Cùng chiều với M3.4 (h lớn → gần Shapley theo khối, phù hợp với trò chơi
xoá tập). Khác biệt nhỏ, CI chồng lấn, không có phép so cặp, trên ảnh test đã dùng: chỉ là giả thuyết cho nghiên cứu sau.

## 4. Giới hạn

| # | Giới hạn | Ảnh hưởng tới |
|---|---|---|
| L1 | J thô bị thang đo chi phối (≥ 91 % của L ở β = 1 là lệch thang đo, M6.5); β chung ngoài mẫu chỉ cho L = 0,965–0,994 (bản đồ 0: 1) | diễn giải C1, C2 |
| L2 | Một mô hình (ResNet-50, IMAGENET1K_V2), một tập ảnh (ImageNet-Segmentation, `gtsegs_ijcv`), một nền (đen) | khả năng suy rộng |
| L3 | Một họ D chính (`del_sq`, 512 + 512 ô); 7 D khác chỉ mô tả | phụ thuộc định nghĩa infidelity |
| L4 | M = 1 `rand`; N ∈ {32, 64}; K ≤ 32 (B ≤ 1024 cho kiểm định); 48 ảnh test (BUDGET), 48–128 theo lực (PATHFAM) | phạm vi ngân sách, độ chính xác |
| L5 | Lý thuyết cho đa thức / mô hình trơn; ReLU là tuyến tính từng mảnh; bậc ≥ 3 của phương sai chỉ có bằng chứng số | cầu nối lý thuyết → thực nghiệm |
| L6 | Các họ khớp theo D (một số); ln khớp D bằng hiệu chuẩn Monte Carlo (σ tất định, seed 10); smooth không đạt vùng α = 1 | so sánh họ |
| L7 | Nhiều phân tích hậu kiểm trên cùng ảnh test (signdiag, signdiag2, final-audit) | mọi chẩn đoán chỉ mô tả |
| L8 | Metric ngoài không bao giờ dùng để chọn; xếp hạng không dấu; InsDel nền blur lệch nền | không có tuyên bố về "chất lượng giải thích" nói chung |
| L9 | Độ dốc completeness tuyệt đối −0,29…−0,52 ở K ≤ 32; tính toán float32; độ lệch riêng ảnh chưa xác lập, nguồn (nếu có) chưa xác định | D4 |
| L10 | Lịch sử E10 / COORD_MIX trước đây không tái lập được từ artifact | chỉ trích dẫn như bối cảnh |

## 5. Hướng dẫn câu chữ

| Nên viết | Không viết |
|---|---|
| "theo objective J đã khoá" | "trung thực hơn" (không kèm điều kiện) |
| "không kém hơn quá 10 % J_rev" | "tương đương", "tốt ngang" |
| "Dirichlet là một lựa chọn thiết kế" | "Dirichlet tối ưu" / "Dirichlet không có vai trò" |
| "chẩn đoán hậu kiểm cho thấy …" | "chúng tôi chứng minh thực nghiệm …" cho D1–D5 |
| "β riêng ảnh (chẩn đoán, không triển khai được)" | báo L ở β riêng ảnh như một phương pháp |
| "phù hợp với Định lý …" | "xác nhận Định lý …" bằng dữ liệu ReLU |
| "độ dốc completeness lệch −½ theo hướng phù hợp với …" | "lỗi cài đặt / lỗi số học" |
