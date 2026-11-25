# Vì sao độ dốc completeness tuyệt đối trên CNN ở K = 2–32 có thể khác −½ dù mô phỏng CPU khớp

Phạm vi: hậu kiểm, mô tả. Dữ liệu: `signdiag2.csv` (ảnh / cấu hình test đã khoá, bản đồ lồng nhau K ∈ {2, 4, 8, 16, 32}). Không GPU, không mở
lại test, không chọn lại. Lý thuyết: `PATH_FAMILIES_THEORY.md` §7. Phân tích số: `pathfam-final-audit` §2b–2d (chạy lại lệnh này trên CPU để
có §2c, §2d mới).

## 1. Kết luận (từ dữ liệu thật, `PATHFAM_FINAL_AUDIT.md` 2026-11-25)

1. **Không chệch: đạt.** Sai số completeness có dấu có CI 95 % chứa 0 ở 55/60 ô (12 cấu hình × 5 mức K) và ở **cả 12 cấu hình tại K = 32**.
   5 ô không chứa 0 (kỳ vọng 3 nếu độc lập; P(≥ 5) = 0,18), 4 âm 1 dương, chỉ ở K ≤ 16, không có cấu hình nào lệch ở nhiều K.
2. **Sai số tuyệt đối giảm theo K**: TB|err| ở K = 32 nhỏ hơn K = 2 từ 2,4× đến 4,2× (lý thuyết 4×); sai số tương đối TB|err|/(|ΔF| + τ) ở
   K = 32 là 0,057–0,114 (τ = trung vị |ΔF| = 6,56).
3. **Độ dốc TB|err| theo log K: −0,29 đến −0,52.** Hai cấu hình N = 64 (spi α = 0,1; α = 10) có CI chứa −½; 7/10 cấu hình N = 32 có CI loại −½,
   tất cả theo hướng **phẳng hơn**.
4. **Phân rã** (độ dốc TB|·| = độ dốc RMS + độ dốc log G) cho thấy độ lệch **không cùng một cơ chế** ở mọi cấu hình (§8). Với N = 64, khớp −½ là do
   hai độ lệch triệt tiêu nhau (đuôi nặng ở K nhỏ làm phẳng, RMS ước lượng dốc hơn), không phải bằng chứng hành vi chuẩn. Với 4 cấu hình N = 32,
   độ lệch nằm ở mức RMS với hình dạng gần chuẩn — chỉ có thể do độ lệch riêng ảnh hoặc do biến thiên lấy mẫu của profile MSE.
5. **Độ lệch riêng ảnh chưa được xác lập**: b̂² có CI chứa 0 ở cả 12 cấu hình; giá trị điểm chỉ giải thích được độ dốc RMS khoảng −0,44…−0,50,
   trong khi quan sát −0,27…−0,39 ở nhóm này. Không có bằng chứng — và không tuyên bố — lỗi cài đặt hay lỗi số học.
6. Theo Mệnh đề 7.1, completeness chỉ đo **nhiễu cầu phương**; mọi điều trên không nói gì về $V_1$, $J$ hay chất lượng sampler.

## 2. Cái gì được đo

$e_K=\sum_i\bar A_{K,i}-\Delta F$ với $\bar A_K$ là trung bình $K/2$ cặp `rev` đầu tiên (lồng nhau: $K=32$ chứa các cặp của $K=2,\dots,16$).
Trong mã: lịch trình của cặp $j$ dùng khoá $(\text{tag},\text{cấu hình},\text{ảnh},j)$, điểm cầu phương dùng khoá $(\dots,j,w)$ với $w$ phân biệt hai
thành viên (`budget_opt.pair_maps`) ⇒ các cặp độc lập, hai thành viên có $U$ độc lập — đúng giả thiết (G). Độ dốc = OLS của $\log\overline{|e_K|}$
(trung bình qua ảnh) theo $\log K$; CI bootstrap theo ảnh (giữ nguyên cả 5 mức $K$ của mỗi ảnh nên tương quan do lồng nhau được tính đúng).

## 3. Bốn cơ chế

**M1 — hình dạng phân phối ở K hữu hạn (Mệnh đề 7.3).** Đồng nhất chính xác: độ dốc $\log\mathbb E|e_K|$ = độ dốc RMS + độ dốc $\log G_K$,
$G_K=\mathbb E|e_K|/\text{RMS}_K$. $G_K\to\sqrt{2/\pi}=0{,}798$ với tốc độ $\le\rho_3/(\sigma^3\sqrt{K/2})$. Sai số một đường là tổng $N$ số hạng
$h_n(U_n)-\int h_n$; khi biến thiên của đạo hàm hướng tập trung vào vài đoạn (đầu ra đổi nhanh trên một đoạn ngắn của đường — điển hình khi
mạng bão hoà), phân phối sai số cặp có đuôi nặng/dạng "hiếm khi một đoạn trội": $G$ nhỏ ở $K=2$ và độ dốc TB|·| **phẳng hơn** $-\tfrac12$. Ngược
lại phân phối đuôi nhẹ cho độ dốc **dốc hơn**. Toy trơn ($d=16$, tanh) và MLP ReLU nhỏ cho sai số gần chuẩn (độ nhọn 3,14), nên không tái hiện
M1. Việc CNN có đuôi nặng hơn là **giả thuyết**, đo được trực tiếp bằng $G_2$, $G_{32}$ và độ dốc $\log G$ trong §2c.

**M2 — độ lệch riêng ảnh $b_i$ (Mệnh đề 7.2).** $\mathbb Ee_K^2=b_i^2+2\sigma_i^2/K$: $b_i\ne0$ làm sai số chững lại ở $K$ lớn và làm phẳng cả
độ dốc RMS lẫn TB|·|, kể cả khi TB sai số có dấu qua ảnh bằng 0 (các $b_i$ trái dấu triệt tiêu). Với số học chính xác và (F1′), $b_i=0$ (M7.1).
Nguồn khả dĩ nằm ngoài mô hình: số học float32 trong lượt tiến/lùi (gradient làm tròn không còn là trường gradient chính xác), $\Delta F$ tính bằng
lượt tiến ở hai đầu mút còn tích phân dùng gradient tại điểm trong, khác biệt nhỏ về dtype/batch giữa hai phép tính. **Chưa có bằng chứng trực
tiếp cho nguồn nào**; §2c chỉ ước lượng độ lớn $\overline{b^2}$ (không chệch, nhờ khôi phục khối cặp rời nhau từ bản đồ lồng nhau).

**M3 — nhiễu lấy mẫu, độ không đồng nhất giữa ảnh, chệch Jensen (Hệ quả 7.4).** Thang sai số $\sigma_i$ thay đổi nhiều bậc giữa ảnh (theo $|\Delta F|$
và hình học đường) ⇒ trung bình qua 128 ảnh bị vài ảnh chi phối; SD của độ dốc lớn; $\log$ của trung bình mẫu chệch xuống mạnh nhất ở $K=2$
(một cặp) và khi đuôi nặng ⇒ thêm xu hướng làm phẳng; CI bootstrap percentile có thể phủ thiếu.

**M4 — thiết kế lồng nhau.** Không đổi kỳ vọng (M7.2); chỉ làm các mức $K$ tương quan. Bootstrap theo ảnh đã tính điều này. Không phải nguồn
lệch hệ thống.

## 4. Mô phỏng ở đúng thiết kế (CPU, đã chạy; 128 ảnh, 16 cặp lồng nhau, SD $\log\sigma_i=1$, 300 lần lặp)

| sai số cặp | độ lệch riêng ảnh SD$(b_i)/\sigma_i$ | E độ dốc TB\|err\| (SD) | E độ dốc RMS (SD) |
|---|---|---|---|
| chuẩn | 0 | −0,498 (0,055) | −0,501 (0,114) |
| đều (đuôi nhẹ) | 0 | −0,527 (0,049) | −0,505 (0,096) |
| Rademacher (rất nhẹ) | 0 | −0,560 (0,048) | −0,518 (0,088) |
| t₃ (đuôi nặng) | 0 | −0,443 (0,068) | −0,471 (0,148) |
| "đoạn trội" (5 %/đường) | 0 | −0,304 (0,098) | −0,406 (0,181) |
| chuẩn | 0,1 | −0,473 (0,057) | −0,475 (0,111) |
| chuẩn | 0,2 | −0,417 (0,056) | −0,419 (0,108) |
| chuẩn | 0,4 | −0,296 (0,051) | −0,298 (0,100) |

Độ phủ của CI percentile 95 % cho $-\tfrac12$ (cùng thiết kế, $b=0$): chuẩn 0,95 (TB|·|) / 0,91 (RMS); t₃ 0,74 / 0,85; "đoạn trội" 0,38 / 0,78.
Đọc: (a) ngay cả khi mọi thứ đúng và sai số chuẩn, SD của độ dốc ≈ 0,05 — một độ dốc quan sát $-0{,}45$ hay $-0{,}55$ không có gì bất thường;
(b) chỉ hình dạng phân phối đã có thể đẩy độ dốc TB|·| tới $-0{,}30\ldots-0{,}56$ với $b=0$; (c) $b_i$ cỡ 20 % của $\sigma_i$ đã cho $-0{,}42$.
Bảng §2d của `pathfam-final-audit` lặp lại mô phỏng này với SD $\log\sigma_i$ ước lượng từ dữ liệu thật.

## 5. Quy tắc đọc §2c (đặt trước khi chạy lại trên dữ liệu thật)

| quan sát trong §2c | diễn giải được phép |
|---|---|
| CI độ dốc RMS chứa $-\tfrac12$; CI độ dốc $\log G$ không chứa 0 | khác biệt của TB\|·\| do hình dạng phân phối hữu hạn K (M1) |
| CI $\overline{b^2}$ > 0; độ dốc RMS > $-\tfrac12$ | phù hợp với độ lệch riêng ảnh (M2); báo $\sqrt{\overline{b^2}}$ tương đối với $|\Delta F|$; nguồn chưa xác định |
| mọi CI chứa giá trị null; độ dốc nằm trong ±2 SD của §2d | phù hợp với nhiễu lấy mẫu (M3) |
| độ dốc RMS dốc hơn $-\tfrac12$, CI không chứa $-\tfrac12$ | không thể do M1/M2 trong kỳ vọng (M7.2) ⇒ nhiễu lấy mẫu, hoặc vi phạm độc lập giữa các cặp — trường hợp DUY NHẤT gợi ý kiểm tra cài đặt (khoá ngẫu nhiên); mã hiện tại đã kiểm khoá ở §2 |
| TB sai số có dấu có CI không chứa 0 (§2b) | mâu thuẫn với không chệch ⇒ có $b$ chung dấu; cần kiểm tra trực tiếp trước khi diễn giải |

## 6. Có thể viết gì trong paper

> *The `rand` quadrature is unbiased: the signed completeness error has a 95 % CI containing 0 for all 12 configurations at K = 32 (55/60
> configuration–K cells overall). The absolute error decreases with K (2.4–4.2× from K = 2 to 32, relative error 6–11 % at K = 32), but the fitted
> slope of mean |error| over K = 2–32 ranges from −0.29 to −0.52, flatter than the asymptotic −½ for most N = 32 configurations. An exact
> decomposition into an RMS part and a distribution-shape part (Prop. 7.3) shows that finite-K heavy tails explain the deviation for some
> configurations, while for others it lies in the RMS profile; per-image offsets are not distinguishable from zero. We therefore treat the
> K^{−1/2} law for absolute error as asymptotic and do not interpret the deviation as an implementation error. Completeness measures quadrature
> noise only and carries no information about path-sampling variance or J (Prop. 7.1).*

Không được: "lỗi cài đặt", "lỗi số học", "độ lệch riêng ảnh gây ra …" (chưa xác lập), hay dùng completeness để kết luận về $J$/sampler.

## 7. Kiểm tra bổ sung (tuỳ chọn, CPU, trên `signdiag2.csv` có sẵn)

- **§2e (đã cài, chưa chạy):** $\overline{\hat m_{ic}\hat m_{ic'}}$ ($c\ne c'$) ước lượng **không chệch** $\mathbb E[b_{ic}b_{ic'}]$ mà không cần trừ phương sai,
  vì nhiễu của hai cấu hình độc lập. Mạnh hơn b̂² nhiều (dùng 12 cấu hình trên cùng 128 ảnh). Dương rõ ⇒ độ lệch theo ảnh chung cho mọi họ
  (nguồn khả dĩ: phép tính ΔF / số học đầu mút — vẫn cần kiểm tra trực tiếp); ≈ 0 ⇒ không có độ lệch chung, độ phẳng do lấy mẫu / đuôi nặng.
  Chạy: `bash scripts/run.sh pathfam-final-audit`. Không đổi kết luận nào ở §6 nếu ≈ 0; nếu dương rõ, thêm một câu mô tả, vẫn không gọi là lỗi.
- Sau này (không làm trong vòng này): tính lại ΔF và vài đường ở float64 trên CPU; lưu sai số từng đường để có $G_1$ trực tiếp; mở rộng dải K.

## 8. Kết quả chi tiết trên dữ liệu thật (§2b–2d)

G tham chiếu "chuẩn + không đồng nhất" $=0{,}798\,e^{-s^2/2}$ với $s$ = SD log σ_i (G gộp qua ảnh thấp hơn 0,798 ngay cả khi sai số chuẩn; mức này
không đổi theo K nên không ảnh hưởng độ dốc). RMS dự báo = độ dốc của $\tfrac12\log(\hat b^2+2\hat\sigma^2/K)$ từ giá trị điểm.

| cấu hình | độ dốc TB\|err\| [CI] | = RMS [CI] | + log G [CI] | G₂ / tham chiếu | b̂²/MSE₃₂ | RMS dự báo từ b̂² | nhóm |
|---|---|---|---|---|---|---|---|
| cmix α0,3 | −0,305 [−0,396; −0,216] | −0,292 [−0,409; −0,175] | −0,013 | 0,72 / 0,72 | 0,23 | −0,46 | A |
| cmix α1 | −0,327 [−0,428; −0,227] | −0,328 [−0,453; −0,206] | 0,001 | 0,68 / 0,69 | 0,30 | −0,44 | A |
| ln α1 | −0,360 [−0,456; −0,265] | −0,393 [−0,490; −0,274] | 0,033 | 0,65 / 0,67 | 0,04 | −0,49 | A |
| spi α0,3 | −0,291 [−0,388; −0,193] | −0,266 [−0,385; −0,144] | −0,026 | 0,71 / 0,70 | 0,27 | −0,45 | A |
| spi α0,1 N64 | −0,519 [−0,638; −0,379] | −0,667 [−0,838; −0,423] | 0,148 [0,019; 0,255] | 0,38 / 0,66 | 0,17 | −0,47 | B |
| spi α10 N64 | −0,487 [−0,596; −0,373] | −0,650 [−0,778; −0,408] | 0,162 [0,016; 0,220] | 0,49 / 0,73 | < 0 | −0,50 | B |
| ln α4 | −0,412 [−0,532; −0,285] | −0,506 [−0,660; −0,296] | 0,094 [−0,015; 0,154] | 0,53 / 0,71 | 0,15 | −0,47 | B (sát) |
| spi α4 | −0,348 [−0,444; −0,248] | −0,475 [−0,607; −0,275] | 0,126 [−0,001; 0,193] | 0,51 / 0,69 | 0,23 | −0,46 | B (sát) |
| ln α0,3 | −0,331 [−0,452; −0,207] | −0,423 [−0,612; −0,158] | 0,092 [−0,072; 0,181] | 0,50 / 0,66 | 0,04 | −0,49 | C |
| spi α1 (incumbent) | −0,334 [−0,451; −0,210] | −0,388 [−0,543; −0,202] | 0,054 [−0,036; 0,120] | 0,57 / 0,66 | < 0 | −0,50 | C |
| cmix α4 | −0,375 [−0,476; −0,274] | −0,395 [−0,535; −0,241] | 0,020 | 0,66 / 0,71 | 0,19 | −0,47 | C |
| smooth δ1 | −0,424 [−0,504; −0,343] | −0,445 [−0,538; −0,350] | 0,021 | 0,69 / 0,64 | ≈ 0 | −0,50 | C |

**Nhóm A — độ lệch ở mức RMS, hình dạng gần chuẩn** (CI độ dốc RMS loại −½; log G ≈ 0; G₂ = 97–102 % tham chiếu). Theo M7.2, trong kỳ vọng chỉ
độ lệch riêng ảnh làm phẳng RMS; nhưng b̂² điểm chỉ dự báo −0,44…−0,49 (cận trên CI của b̂² cho phép tới khoảng −0,3, nên không loại được).
Phần còn lại phù hợp với biến thiên lấy mẫu của profile MSE: MSE ở K = 2 dựa trên **một** cặp mỗi ảnh, gộp qua 128 ảnh có thang khác nhau.
**Nhóm B — hình dạng ở K hữu hạn** (log G > 0, G₂ = 57–75 % tham chiếu): đuôi nặng ở K nhỏ làm phẳng TB|·| (M7.3). Với hai cấu hình N = 64, độ dốc
RMS ước lượng lại dốc hơn −½ (CI vẫn chứa −½; không thể dốc hơn trong kỳ vọng theo M7.2 ⇒ lấy mẫu), hai độ lệch triệt tiêu ⇒ TB|·| ≈ −½.
**Nhóm C — không phân biệt được**: các CI chứa cả giá trị null của RMS lẫn log G.

**So với mô phỏng §2d** (cùng thiết kế, s = 0,54): sai số chuẩn không lệch cho TB|·| −0,504 (SD 0,040), RMS −0,506 (SD 0,052). Các độ dốc N = 32
quan sát cách đó 2–5 SD — **không** phù hợp với "chuẩn, không lệch". Mô hình "đoạn trội" cho TB|·| −0,322 (SD 0,070), RMS −0,475 (SD 0,119); chuẩn có
SD(b_i)/σ_i = 0,2 cho −0,425: các độ dốc quan sát nằm trong ±2 SD của hai mô hình này. Lưu ý: 12 cấu hình dùng **chung 128 ảnh** nên không phải 12 lần
lặp độc lập — vài ảnh có sai số lớn có thể làm phẳng đồng thời nhiều cấu hình. Không suy luận gì về khác biệt giữa họ hay giữa N từ bảng này
(N = 64 trùng với α khác; mô tả).

Phân loại nhóm là mô tả, không hiệu chỉnh đa kiểm định, đặt theo quy tắc §5 trước khi xem số liệu.
