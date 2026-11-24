# PATHFAM_METRIC_AUDIT — định nghĩa chính xác các metric có dấu / không dấu (đối chiếu mã)

Không đổi endpoint. Tài liệu này chỉ xác định metric nào đo cái gì, chiều tốt, và chỗ nào không nhất quán với attribution.

## 1. Định nghĩa theo mã

Attribution A ∈ ℝ^{C×H×W} được tính theo đường từ **nền đen** x′ = 0 (`IEnv.x0 = make_base(x, "black")`), nên ΣA ≈ F(x) − F(x′) và dấu của
A_c(p) nói pixel-kênh đó đẩy logit lớp đích lên (+) hay xuống (−) **khi đi từ đen tới x**.

| Metric | Mã | Bản đồ dùng để xếp hạng / đo | Chiều tốt |
|---|---|---|---|
| `insdel_abs` (= `insdel_blur` đã khoá) | `SM.insdel_full(..., signed=False)` | m(p) = Σ_c \|A_c(p)\|, giảm dần; không phá hoà | lớn |
| `insdel_signed` | `SM.insdel_full(..., signed=True)` | m(p) = Σ_c A_c(p) (có dấu), giảm dần: dương lớn nhất trước, âm cuối cùng | lớn |
| `iou_abs` | top-k theo Σ_c\|A_c\|, k = diện tích mask, phá hoà bằng hoán vị cố định | IoU(top-k, mask) | lớn |
| `iou_pos` | top-k theo Σ_c A_c (có dấu), cùng k, cùng hoán vị | IoU(top-k dương nhất, mask) | lớn |
| `energy_abs` | Σ_{p∈mask} Σ_c\|A_c\| / Σ_p Σ_c\|A_c\| | tỷ lệ khối lượng không dấu trong mask | lớn |
| `energy_pos` | Σ_{p∈mask} max(Σ_c A_c, 0) / Σ_p max(Σ_c A_c, 0) | phần DƯƠNG lấy SAU khi cộng kênh (không phải dương theo từng kênh) | lớn |

InsDel: `diff = AUC_ins − AUC_del` của xác suất softmax lớp đích; chèn = đi từ nền tới x theo thứ tự xếp hạng, xoá = đi từ x tới nền.
Với `signed=True`, pixel có bằng chứng âm được chèn / xoá sau cùng.

## 2. Phát hiện: nền của InsDel khác nền của attribution

InsDel đã khoá dùng nền **blur**. Hiệu ứng bậc nhất của việc thay pixel p từ nền b sang x là g(p)·(x(p) − b(p)), còn attribution đo
g(p)·(x(p) − 0). Hai dấu khác nhau ở mọi pixel tối hơn vùng lân cận (x < b) khi g > 0. Hệ quả:
- `insdel_signed` với nền blur **không có diễn giải nhất quán**: một bản đồ hoàn toàn trung thực theo nền đen có thể cho diff ≈ 0 hoặc âm.
  Điều này giải thích giá trị trung bình gần 0 / âm quan sát được ở vòng 1, và có nghĩa so sánh H-SU ở vòng 1 bị nhiễu bởi lệch nền đối với
  InsDel (không ảnh hưởng `iou_*`, `energy_*` vì chúng không dùng nền).
- `insdel_abs` với nền blur cũng chịu lệch này, nhưng nhẹ hơn vì chỉ dùng độ lớn.

Ví dụ kiểm được (CPU, `tests/test_proof_signdiag.py::test_insdel_baseline_mismatch`): mô hình tuyến tính F = Σ w·x, A = w ⊙ x là IG chính
xác theo nền đen, w có cả âm và dương:

| nền InsDel | không dấu | có dấu |
|---|---|---|
| đen (cùng nền với A) | −0,34 | **+0,64** |
| blur | −0,10 | −0,11 |

Với nền nhất quán, xếp hạng có dấu đúng chiều (+0,64) còn không dấu sai chiều (−0,34) vì đưa bằng chứng âm lên sớm; với nền blur cả hai
đều mất ý nghĩa.

## 3. Hệ quả cho chẩn đoán (không đổi endpoint)

- Endpoint ngoài đã khoá (`insdel_blur`, không dấu) giữ nguyên.
- Vòng 2 (`pathfam-signdiag2`) báo InsDel theo cả hai nền: `insdel_{abs,signed}_{blur,black}`. Phiên bản nhất quán để so có dấu / không
  dấu là nền đen. Kết luận H-SU của vòng 1 về InsDel được coi là **không diễn giải được**; về IoU / năng lượng vẫn đọc được.
- Hoà: InsDel xếp hạng bằng `argsort` không phá hoà; IoU dùng hoán vị phá hoà cố định. Với bản đồ có nhiều pixel bằng 0 tuyệt đối (không
  xảy ra với attribution liên tục) điều này mới có ảnh hưởng.
