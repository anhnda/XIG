# attr_probe

Bộ code mới, độc lập với `spi_study`, để kiểm giả thuyết:

> Phương pháp attribution tốt ít phụ thuộc vào các chi tiết tuỳ ý của quá trình giải thích (lịch trình
> trạng thái, trạng thái trung gian bất thường, cách chia credit), nhưng vẫn nhạy với thay đổi chức năng thật.

Chưa chạy lần nào. Bắt đầu bằng `bash scripts/run.sh test`, sau đó `bash scripts/run.sh debug`.

## Thí nghiệm

| | Câu hỏi | Thiết kế |
|---|---|---|
| **A** | Cải thiện đến từ **đường đi** hay từ **quy tắc chia credit**? | 4 đường (IG, SPI α=10, Guided IG, BlurIG) × 2 quy tắc (canonical, IDGI) trên **cùng các trạng thái**. Báo cáo chất lượng (E_mask, IoU, insdel blur/đen, ROAD), hiệu ứng quy tắc theo từng đường, hiệu ứng đường theo từng quy tắc, và độ tương đồng giữa mọi cặp bản đồ (không lấy IG làm tâm) |
| **B** | Phương pháp nào bị một gradient bất thường đã biết kéo lệch? | Cài bẫy H vào hàm mục tiêu: H = 0 dọc một đường cong nhưng ∇H ≠ 0 trên đó. Quét độ rộng ống ρ, tần số không gian của hướng u, và hình dạng (quanh đường IG làm đối chứng dương, hoặc một đường cong mà không phương pháp nào đi theo). Đo R = ‖A(F+H) − A(F)‖/‖A(F)‖ và kiểm H có vô hại với hành vi mô hình không |
| **C** | Phương pháp có giữ được tín hiệu thật không? | Cài S(x) = w·Σ_{i∈R} x_i với ô R (trong hoặc ngoài vật), dấu ±, tổng đóng góp ±0,5·\|D\|. Đo retention, sai số so với đóng góp đúng, phần lan ra ngoài R, và đúng dấu |
| **Sanity** | Bản đồ có đổi khi mô hình bị phá không? | Khởi tạo lại tích luỹ fc → layer4 → layer3 → layer2 (theo Adebayo et al.). Spearman, cos, top-k so với bản gốc |
| **D** | Kết quả có lặp lại không? | Cùng các thí nghiệm trên 50 ảnh mới, config khoá trước (`--split confirm`) |

## Đọc kết quả B và C cho đúng

Tích phân thường (canonical) là **tuyến tính theo hàm**: A(F + H) = A(F) + A(H). Hệ quả:
- Với IG, SPI và BlurIG theo quy tắc canonical, kết quả của B **không phụ thuộc ResNet**, chỉ phụ thuộc hình học giữa đường đi và bẫy.
- Với C, retention = 1 theo cấu tạo; sai khác chỉ do rời rạc hoá. Đây là đối chứng kiểm code đúng.
- Thông tin thật của B và C nằm ở các phương pháp **phụ thuộc mô hình**: Guided IG (cả hai quy tắc) và mọi biến thể IDGI.

Riêng với SPI ở B có một **dự đoán định lượng**. Bẫy quanh đường IG có độ rộng r = ρ‖Δx‖. Trên đường SPI, ‖w‖² ≈ h·t(1−t)·‖Δx‖², nên phần SPI bị bẫy ảnh hưởng, so với IG, là a(ρ) = 2∫ sin²(πt)·exp(−h t(1−t)/(2ρ²)) dt. Cột "dự đoán" trong báo cáo B đặt cạnh cột "R / thay đổi của IG" để so trực tiếp.

## Các lựa chọn thiết kế

- **Cùng trạng thái cho hai quy tắc.** Mỗi đường trả về x_0..x_N, f(x_k) và g_k = ∂f/∂x tại x_k. Từ đó tính:
  - canonical = Σ g_k ⊙ (x_{k+1} − x_k)
  - idgi = Σ g_k²/‖g_k‖² · (f(x_{k+1}) − f(x_k))

  Như vậy hiệu ứng của quy tắc tách được khỏi hiệu ứng của đường. Riemann trái có chệch ở các bước nhảy của SPI; báo cáo A có cột residual.
- **Guided IG** cài lại theo thuật toán tham chiếu của thư viện PAIR `saliency`: 128 bước, fraction 0,25, max_dist 0,02, phân vị tính trên mọi phần tử.
- **BlurIG** dùng điểm đầu là ảnh mờ σ = 50, không ép về ảnh đen. Completeness tính theo f(x) − f(blur(x)).
- **Bẫy định nghĩa trong không gian ảnh**, với u ⟂ Δx. Không phải chia cho Δx (pixel đen có Δx = 0), nên áp được cho mọi đường, kể cả BlurIG và Guided IG. Với κ = 2τ‖A_IG‖/‖u⊙Δx‖, thay đổi của IG đúng bằng τ‖A_IG‖. Cột `ig_formula_err` kiểm điều này.
- **Common random numbers.** Đường SPI dùng cùng seed cho F, F + H, F + S và mô hình bị phá, nên hiệu giữa chúng gần như không mang nhiễu MC. Phần còn lại được khử bằng phương sai của hiệu các sub-estimate ghép cặp.
- **Khử nhiễu.** Mọi đại lượng bậc hai của SPI (E_mask, năng lượng, cos, R) đều trừ phương sai MC theo phần tử.
- **Ảnh.** Loại 200 ảnh của báo cáo gốc và 50 ảnh của đợt Cửa 1/1B (`local.exclude_index_files`). Phần còn lại chia thành pilot (20), confirm (50) và other (10, dùng để kiểm H vô hại).

## Chạy

```bash
pip install -r requirements.txt
# sửa local: trong config.yaml (đường dẫn gtsegs_ijcv.mat, các file index cũ)
bash scripts/run.sh test      # CPU, < 1 phút
bash scripts/run.sh debug     # 2 ảnh, ít bước, mọi thí nghiệm → runs_probe_debug/pilot/report.md
bash scripts/run.sh info      # danh sách ảnh, số eval mỗi biến thể
bash scripts/run.sh a         # mỗi lệnh in ước tính thời gian trước khi chạy
bash scripts/run.sh b
bash scripts/run.sh c
bash scripts/run.sh sanity
REASON="..." bash scripts/run.sh lock
bash scripts/run.sh confirm
```

Chạy tiếp được: các ảnh và biến thể đã có trong `*.jsonl` được bỏ qua. Attribution gốc của mỗi ảnh được cache trong `cache/`, dùng chung cho A, B, C và sanity.

**Ước tính thời gian ở 670 eval/s theo lô** (Guided IG chạy tuần tự, giả định khoảng 40 eval/s): khoảng 6–7 s cho mỗi ảnh mỗi biến thể mô hình.

| | Số ảnh | Số biến thể | Thời gian |
|---|---|---|---|
| A | 20 | 1 | khoảng 3 phút, cộng metric đầy đủ khoảng 3 phút |
| B | 6 | 20 bẫy | khoảng 13 phút |
| C | 10 | 4 | khoảng 5 phút |
| sanity | 10 | 4 | khoảng 5 phút |

Tổng pilot khoảng 30 phút. Con số chính xác do lệnh in ra sau khi đo trên máy thật.

## Đầu ra (`runs_probe[_debug]/<split>/`)

`a_quality.jsonl`, `a_pairs.jsonl`, `b_traps.jsonl`, `c_signal.jsonl`, `sanity.jsonl`, `report.md`, `run_info.json` (hash config, danh sách ảnh), `cache/`.

## Cấu trúc

```
probe/config.py       đọc config, chế độ debug, khoá cho bước xác nhận, seed
probe/data.py         gtsegs_ijcv, chia pilot / confirm / other, loại ảnh đã dùng
probe/model.py        ResNet-50, f = logit + bẫy + tín hiệu, khởi tạo lại theo tầng
probe/integrate.py    f và ∂f/∂x tại các trạng thái; quy tắc canonical và IDGI
probe/paths.py        IG, SPI (Dirichlet log-space), Guided IG, BlurIG
probe/traps.py        bẫy H, hướng u/q, κ theo τ, dự đoán suy giảm cho SPI, tín hiệu S, chọn ô
probe/metrics.py      E_mask, năng lượng, IoU, cos khử nhiễu, Spearman, top-k, insdel, ROAD
probe/experiments.py  A, B, C, sanity, ước tính thời gian
probe/report.py       report.md
tests/test_probe.py   test trên CNN ReLU nhỏ: công thức bẫy, completeness, các đường, tín hiệu
```

## Giới hạn đã biết

- H = 0 tại x′ và x, nhưng H vẫn có thể đổi hành vi mô hình ở nơi khác. Các cột H_noise, H_other chỉ kiểm trên một tập nhỏ đã định trước (x có nhiễu, 10 ảnh khác). Với ρ = ∞, H không còn cục bộ và có thể khác 0 đáng kể trên ảnh khác: đó là bẫy có ý nghĩa chức năng, không phải nhiễu.
- Không có DiffIG: cần một diffusion model đã huấn luyện.
- Một mô hình (ResNet-50 V2), một baseline (ảnh đen), một bộ dữ liệu.
