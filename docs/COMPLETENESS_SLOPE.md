# Vì sao độ dốc completeness tuyệt đối trên CNN ở K = 2–32 có thể khác −½ dù mô phỏng CPU khớp

Phạm vi: hậu kiểm, mô tả. Dữ liệu: `signdiag2.csv` (ảnh / cấu hình test đã khoá, bản đồ lồng nhau K ∈ {2, 4, 8, 16, 32}). Không GPU, không mở
lại test, không chọn lại. Lý thuyết: `PATH_FAMILIES_THEORY.md` §7. Phân tích số: `pathfam-final-audit` §2b–2d (chạy lại lệnh này trên CPU để
có §2c, §2d mới).

## 1. Kết luận ngắn

Độ dốc $-\tfrac12$ của **trung bình sai số tuyệt đối** theo $\log K$ là phát biểu **tiệm cận**. Thứ được đảm bảo chính xác ở mọi $K$ là
(i) sai số có dấu có kỳ vọng 0 và (ii) $\mathbb Ee_K^2=b^2+2\sigma^2/K$. Với 128 ảnh và dải $K$ chỉ 1,2 thập phân, độ dốc quan sát có thể lệch
$-\tfrac12$ vì ba cơ chế không liên quan tới lỗi cài đặt: hình dạng phân phối sai số ở $K$ nhỏ (M1), độ lệch riêng ảnh chưa xác định nguồn (M2),
và nhiễu lấy mẫu/chệch Jensen khi thang sai số rất khác nhau giữa ảnh (M3). Mô phỏng CPU khớp $-\tfrac12$ vì chúng nằm đúng trong vùng mà cả ba
cơ chế đều yếu (nhiều lần lặp, sai số gần chuẩn, thang đồng nhất). Không có bằng chứng — và không được tuyên bố — rằng độ lệch là lỗi cài đặt.

Điều quan trọng hơn cho paper: theo Mệnh đề 7.1, $\sum_iA_i(Z)=\Delta F$ với **mọi** đường, nên sai số completeness chỉ đo **nhiễu cầu phương**,
không đo phương sai đường của họ $Q$. Độ dốc completeness không nói gì về $V_1$, $J$ hay chất lượng sampler.

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

Được: "Cầu phương `rand` không chệch (sai số completeness có dấu có CI chứa 0 ở mọi K) và MSE giảm theo K; luật $K^{-1/2}$ của sai số tuyệt
đối là tiệm cận và ở K ≤ 32 độ dốc quan sát [giá trị, CI] lệch $-\tfrac12$ theo hướng phù hợp với [M1/M2/M3 theo §2c]." Không được: "lỗi cài
đặt", "lỗi số học" (khi chưa có kiểm tra trực tiếp), hay dùng completeness để kết luận về $J$ hoặc sampler.

## 7. Kiểm tra bổ sung có thể làm sau (không làm trong vòng này)

Tính lại $\Delta F$ và vài đường ở float64 trên CPU cho một số ảnh (tách M2 do số học); lưu sai số từng đường (cho $G_1$ trực tiếp); mở rộng dải K.
Mọi kiểm tra như vậy là hậu kiểm, không đổi endpoint.
