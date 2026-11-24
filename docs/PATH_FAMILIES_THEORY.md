# PATH_FAMILIES_THEORY — Phase 2: lý thuyết ở mức phân phối đường ngẫu nhiên theo feature

Câu hỏi: những thống kê nào của phân phối đường Q — moment, độ tập trung, tương quan giữa feature, cấu trúc kích hoạt — quyết định
attribution kỳ vọng μ_Q và phương sai, và do đó J? Không giả định Dirichlet. Mỗi phát biểu được gắn nhãn:
**[Định lý]** chứng minh đầy đủ; **[Hệ quả]** suy trực tiếp; **[Kiểm số]** xác nhận bằng tích phân chính xác trên CPU
(`bash scripts/run.sh pathfam-theory`, `pathfam-audit`, `tests/test_path_families.py`); **[Giả thuyết]** cần dữ liệu.

## 0. Thiết lập

Feature i = 1..d (pixel × kênh). Lịch trình Z: [0, 1] → [0, 1]^d, mỗi tọa độ không giảm, Z(0) = 0, Z(1) = 1, biến phân bị chặn (đường lưới
nội suy tuyến tính thoả). Đường γ = x′ + Δ ⊙ Z, Δ = x − x′. Attribution theo đường
  A_i(Z) = Δ_i ∫₀¹ ∂_iF(γ(t)) dZ_i(t)   (tích phân Riemann–Stieltjes).
Q là phân phối của Z; μ_Q = E_Q A(Z). Ước lượng: K đường, mỗi đường cầu phương `rand` (N đoạn, M điểm): E[Â | Z] = A(Z) (đã chứng minh ở P0).
Với chi phí N·M gradient / đường và ngân sách B: K = B/(NM) và
  J_D(Q, B) = L_D(μ_Q) + (NM/B) · σ²_D(Q),   σ²_D(Q) = [tr P_D (Cov_Q A + E_Q Var_U Â) P_Dᵀ] / ‖ΔF_D‖²   (`indep`).
Ký hiệu S_{a,i}(Z) = ∫₀¹ Z(t)^a dZ_i(t) với a là đa chỉ số trên các feature (Z^a = Π_j Z_j^{a_j}) — "moment Stieltjes" của đường.

## 1. Thống kê đủ cho μ (mọi Q, kể cả phụ thuộc)

**[Định lý 1]** Nếu F là đa thức bậc ≤ p thì A_i(Z) là tổ hợp affine của {S_{a,i}(Z) : |a| ≤ p − 1} với hệ số chỉ phụ thuộc F, x′, Δ. Do đó
μ_Q chỉ phụ thuộc Q qua M_{a,i}(Q) = E_Q S_{a,i}, và Cov_Q(A) chỉ phụ thuộc Q qua các moment bậc hai E[S_{a,i} S_{b,l}]. Ngược lại, nếu
M_{a,i}(Q) ≠ M_{a,i}(Q′) với một |a| ≤ p − 1 thì có F bậc ≤ p, x′, Δ sao cho μ_Q ≠ μ_{Q′}.

*Chứng minh.* ∂_iF(x′ + Δ⊙Z) là đa thức bậc ≤ p − 1 theo Z; khai triển và tích phân từng đơn thức theo dZ_i. Chiều ngược: x′ = 0, Δ = 1,
F = x^{a + e_i} cho ∂_iF = (a_i + 1) x^a nên A_i = (a_i + 1) S_{a,i}. ∎

*Nhận xét.* S_{a,i} là các tọa độ của chữ ký (signature) cấp |a| + 1 đã đối xứng hoá phần tiền tố (tích xáo trộn của các tích phân cấp 1),
nên μ_Q là một hàm tuyến tính của **chữ ký kỳ vọng** của Z — nhưng chỉ của tập con "tiền tố đối xứng", nhỏ hơn toàn bộ chữ ký.

## 2. Bậc 2: diện tích Lévy, bất biến khi đảo, và cặp `rev`

Đặt 𝔄_ij(Z) = ½∫(Z_j dZ_i − Z_i dZ_j) (diện tích Lévy của cặp tọa độ). Vì ∫Z_j dZ_i + ∫Z_i dZ_j = Z_iZ_j|₀¹ = 1, ta có S_{e_j,i} = ½ + 𝔄_ij.

**[Định lý 2]** Với F bậc 2, Hessian H, **theo từng đường**:  A_i(Z) − IG_i = Δ_i Σ_j H_ij Δ_j 𝔄_ij(Z).

*Chứng minh.* ∂_iF(γ) = c_i + Σ_j H_ij Δ_j Z_j với c_i không phụ thuộc Z; ∫dZ_i = 1, ∫Z_j dZ_i = ½ + 𝔄_ij, ∫Z_i dZ_i = ½; IG ứng với 𝔄 = 0. ∎

**[Hệ quả 2a]** μ_Q = IG với mọi F bậc 2 ⇔ E_Q 𝔄_ij = 0 với mọi i ≠ j.

**[Hệ quả 2b]** Đặt RZ(t) = 1 − Z(1 − t). Đổi biến s = 1 − t cho ∫RZ_j dRZ_i = 1 − ∫Z_j dZ_i, tức 𝔄(RZ) = −𝔄(Z). Vậy nếu Q bất biến khi đảo
(RZ ~ Z) thì E 𝔄 = 0 và μ_Q = IG cho mọi F bậc 2 — **không cần độc lập giữa feature và không cần E Z = t**. Cả 4 họ dp, ln, cmix, smooth
đều bất biến khi đảo (Phase 0 §2.2); họ phụ thuộc (patch, hier, LINE) cũng vậy nếu lịch trình nhóm bất biến khi đảo.

**[Định lý 2c — `rev` chính xác ở bậc 2]** Với F bậc 2 và MỌI Q: A(Z) + A(RZ) = 2·IG theo từng đường. Do đó trung bình cặp `rev` không có
phương sai đường ở bậc 2; V_e (phương sai của trung bình cặp) chỉ chứa phần bậc ≥ 3 của mô hình cộng nhiễu cầu phương.

**[Kiểm số]** (`pathfam-theory` T2 + T5, d = 4, N = 32, 10⁵ đường): sai lệch từng đường của Định lý 2 và 2c ≤ 2·10⁻¹⁵ cho dp, cmix, ln, smooth,
họ PHỤ THUỘC `group` (feature trong nhóm dùng chung lịch trình) và họ KHÔNG bất biến `power` (Z_i = t^{a_i}); |z| của E 𝔄 ≤ 1,5 cho mọi họ
bất biến; `power` có |E 𝔄| = 0,36 và ‖E A − IG‖ = 0,35 (khớp Hệ quả 2a).

## 3. Feature độc lập: moment biên quyết định μ

**[Định lý 3]** Nếu các tọa độ của Z độc lập, với m_{j,r}(t) = E Z_j(t)^r:
  M_{a,i}(Q) = (1/(a_i + 1)) ∫₀¹ Π_{j≠i} m_{j,a_j}(t) d m_{i,a_i+1}(t).
Do đó với F bậc ≤ p, μ_Q chỉ phụ thuộc các **đường moment biên** m_{j,r}, r ≤ p.

*Chứng minh.* Z_i^{a_i} dZ_i = d(Z_i^{a_i+1})/(a_i + 1) (Z_i liên tục tuyệt đối). Điều kiện theo các tọa độ khác, G(t) = Π_{j≠i} Z_j(t)^{a_j}
không âm, không giảm, bị chặn; tích phân từng phần cho E[∫G dW | G] = ∫ G d(E W) với W = Z_i^{a_i+1} độc lập với G; rồi lấy kỳ vọng theo G
(Fubini): E G(t) = Π m_{j,a_j}(t). ∎

Với v(t) = Var Z(t), κ₃, κ₄ là cumulant bậc 3, 4 của Z(t), D = ∫ v, cùng phân phối cho mọi feature và E Z = t:

**[Hệ quả 3a]** bậc ≤ 2: μ = IG. Bậc 3: μ chỉ phụ thuộc Q qua **D** (các số hạng: 1/3, 1/3 + D, 1/3 − D/2, 1/3).
**[Hệ quả 3b]** bậc 4: qua D, ∫ t v và ∫ κ₃; nếu bất biến khi đảo (v đối xứng, κ₃ phản đối xứng quanh ½) thì **chỉ qua D**
(bảy mẫu chỉ số liệt kê trong PATH_FAMILIES_AUDIT.md §5.1).
**[Hệ quả 3c]** bậc 5: xuất hiện ∫ v², ∫ t² v, ∫ t κ₃, ∫ κ₄ — không còn xác định bởi D, kể cả khi khớp toàn bộ hàm hiệp phương sai.
**[Kiểm số]** bậc 3, 4: E A_f − A_IG = D_f · a₁ với cùng a₁ cho 6 họ / tham số, |z| ≤ 2,5. Bậc 5 có chủ đích F = x₀(x₁ − ½)⁴: dp α = 1 và
cmix λ = ½ (cùng D, cùng c(s, t)) khác nhau z = 31; ln khác dp z = −9,6.

*Diễn giải kích hoạt.* Khi h → 1, Z_i(t) → 1{t ≥ U_i} với U_i độc lập đều: S_{e_j,i} = 1{U_j < U_i} (thứ tự kích hoạt), và A là giá trị
Shapley của trò chơi "bật feature theo thứ tự ngẫu nhiên" quanh x′. Trên lưới N đoạn, các feature nhảy trong cùng một đoạn đi theo đường
thẳng với nhau ⇒ "Shapley theo khối" với N khối. h nội suy giữa IG (h = 0) và Shapley theo khối (h = 1); D đo vị trí trên trục đó.

## 4. Phương sai: cần nhiều hơn D

**[Định lý 4]** Feature độc lập, cùng phân phối, E Z = t, F bậc 2. Với c(s, t) = Cov(Z(s), Z(t)), Γ = ∬ c, Λ = ∬ c d²c (độ đo
Lebesgue–Stieltjes của ∂²c/∂s∂t), các moment bậc hai của {S_{e_j,i}} là hàm song tuyến tính của c:
  Var S_{e_j,i} = 2Γ + Λ;  Cov(S_{e_j,i}, S_{e_k,i}) = Cov(S_{e_j,i}, S_{e_j,l}) = Γ;  Cov(S_{e_j,i}, S_{e_i,l}) = −Γ;  S_{e_i,j} = 1 − S_{e_j,i};
các cặp bốn chỉ số khác nhau: 0. Do đó Cov_Q(A) bậc 2 chỉ phụ thuộc Q qua **hàm hiệp phương sai c** — một hàm hai biến, không phải số D.

*Chứng minh (Var).* E(∫Z_j dZ_i)² = ∬ E[Z_j(s)Z_j(t)] d²E[Z_i(s)Z_i(t)] = ∬ (st + c) d²(st + c); khai triển bốn số hạng: ¼, Γ, Γ (tích phân từng
phần hai lần, c triệt tiêu ở biên), Λ. Các hiệp phương sai còn lại tính tương tự. ∎

| Họ | c(s, t), s ≤ t | Γ | Λ | Var S_{e_j,i} |
|---|---|---|---|---|
| dp lưới (α), cmix (λ = h) | h·s(1 − t) | h/12 | h²/12 | h/6 + h²/12 |
| smooth (δ) | δ² s(1 − s) t(1 − t) | δ²/36 | 0 | δ²/18 = 5h/18 ở cùng D |
| IG | 0 | 0 | 0 | 0 |
| block-Shapley (h = 1) | s(1 − t) | 1/12 | 1/12 | 1/4 |

**[Kiểm số]** (`pathfam-theory` T4a, 4·10⁵ đường): dp, cmix ở h = ½ lệch −1,9 % / −2,0 % (N = 32) và −0,5 % / −1,1 % (N = 128) so với
0,1042 — lệch giảm theo N (c của đường nội suy khác c liên tục); smooth δ = 1 lệch −0,1 % so với 1/18.

**[Hệ quả 4a]** dp và cmix ở cùng h có cùng c ⇒ cùng μ tới bậc 4 và cùng phương sai bậc 2; khác biệt phương sai giữa chúng chỉ đến từ bậc
≥ 3. **[Kiểm số]** F bậc 3: cùng μ (|z| = 1,4) nhưng phương sai khác (|z| = 71).
**[Hệ quả 4b]** Ở cùng D, smooth có phương sai bậc 2 lớn hơn DP khi h nhỏ (5h/18 so với h/6 + h²/12; tại h = 0,091: 0,0253 so với 0,0159).

## 5. Ngân sách: biên lệch–chi phí–phương sai

**[Mệnh đề 5]** Với hai cấu hình θ₁, θ₂: J(θ₁, B) − J(θ₂, B) = [L(μ₁) − L(μ₂)] + [N₁M₁σ²₁ − N₂M₂σ²₂]/B. Hiệu này tuyến tính theo 1/B nên
**đổi dấu nhiều nhất một lần** khi B thay đổi, tại B* = (N₂M₂σ²₂ − N₁M₁σ²₁)/(L(μ₁) − L(μ₂)) khi B* > 0. Tiêu chí so sánh họ đường là cặp
(L(μ_Q), N·M·σ²_Q) — "lệch" và "chi phí-phương sai mỗi gradient"; thứ hạng theo ngân sách (BUDGET, PATHFAM) là hệ quả trực tiếp.
(Đẳng thức đúng chính xác cho J_indep, coi B liên tục — K thực tế phải nguyên; với `rev` thay σ² bằng 2V_e và K chẵn.)

## 6. Tóm tắt: thống kê nào quyết định cái gì

| Đại lượng | Bậc tương tác của F | Thống kê đủ của Q | Trạng thái |
|---|---|---|---|
| μ_Q | 2 | E 𝔄_ij (diện tích Lévy kỳ vọng); = 0 nếu bất biến khi đảo | Định lý |
| μ_Q | ≤ p, mọi Q | moment Stieltjes M_{a,i}, \|a\| ≤ p − 1 | Định lý (cần và đủ) |
| μ_Q | 3 (độc lập, E Z = t) | D | Định lý |
| μ_Q | 4 (+ bất biến khi đảo) | D | Định lý |
| μ_Q | 5 | thêm ∫ v², ∫ κ₄, … | Định lý + phản ví dụ số |
| Cov_Q A | 2 (độc lập) | hàm hiệp phương sai c(s, t) | Định lý |
| Cov_Q A | ≥ 3 | moment bậc 4 của đường | Kiểm số |
| V_e (`rev`) | 2 | 0 với mọi Q | Định lý |
| thứ hạng theo B | — | (L(μ), N·M·σ²) | Mệnh đề |

## 7. Giả thuyết kiểm được trên dữ liệu PATHFAM đã có (không cần GPU; `pathfam-diag`, `pathfam-report`)

- **H-T1** Ở cùng h, |ΔL(μ)| của ln / cmix so với spi nhỏ so với |ΔV₁| (Hệ quả 3a–3b, 4a). Bảng "cùng h" trong INFID_PATHFAM_REPORT.md.
- **H-T2** smooth δ = 0,674 so với SPI α = 10 N = 64 (cùng D): L(μ) gần nhau, V₁ của smooth lớn hơn (Hệ quả 4b). `pathfam-diag` §4.
  Lưu ý: hai cấu hình khác N (32 / 64) nên so V₁ mỗi đường, không so J.
- **H-T3** ρ = 2V_e/V₁ ≈ 1 quan sát được ở mọi họ nghĩa là phần bậc 2 của phương sai đường nhỏ so với phần bậc ≥ 3 cộng nhiễu cầu phương
  (Định lý 2c). Kiểm thêm bằng τ² (nhiễu cầu phương) từ `quad-check` cũ.

## 8. J và metric ngoài (mục 3 của yêu cầu)

Sự kiện từ mã (đã đối chiếu): infidelity dùng A **có dấu**, cộng qua mọi kênh trong từng ô che, chuẩn hoá theo ‖ΔF_S‖² của cùng ảnh và cùng
bộ ô cho mọi cấu hình — không có lệch chuẩn hoá giữa họ. Metric ngoài xếp hạng pixel theo Σ_c |A_c| (`metrics.amap`, `insdel_full(signed=False)`):
bỏ dấu, bất biến với phép nhân dương ⇒ chỉ phụ thuộc **thứ hạng độ lớn**. Completeness chính xác cho từng đường với tích phân chính xác,
và không chệch với `rand` — giống nhau giữa các họ, nên thang đo tổng không phải nguồn khác biệt; khác biệt nằm ở phân bố khối lượng có dấu
giữa các pixel (triệt tiêu dương/âm trong cùng ô không ảnh hưởng infidelity nhưng làm đổi thứ hạng Σ|A|). Hai thuộc tính khác nhau của cùng
bản đồ ⇒ mâu thuẫn J ↔ metric ngoài không phải mâu thuẫn logic.

`pathfam-diag` đo: Kendall τ theo ảnh giữa −J (mức B) / −L(μ) (mức Ē, loại phương sai K) và từng metric ngoài; xu hướng theo h; smooth so với
incumbent và so với SPI α = 10 cùng D (tách hiệu ứng họ khỏi hiệu ứng exploration). Chưa đo được từ dữ liệu hiện có: tỷ lệ triệt tiêu
‖A‖₁/|ΣA|, khối lượng âm, InsDel có dấu, completeness thực nghiệm — cần lưu thêm từ bản đồ ở lần chạy sau (không thêm gradient).

## 9. Giới hạn

- ReLU làm F tuyến tính từng mảnh, không phải đa thức: μ phụ thuộc Q qua phân phối thời điểm đường cắt các mặt kink, không được Định lý 3
  bao phủ. Lý thuyết đa thức áp dụng cho xấp xỉ Taylor cục bộ hoặc mô hình trơn (GELU, softplus); khoảng cách giữa hai trường hợp là giả
  thuyết thực nghiệm.
- Định lý 4 chỉ cho bậc 2 và feature độc lập; họ phụ thuộc cần hiệp phương sai chéo giữa feature.
- N hữu hạn: mọi họ được định nghĩa trên lưới; c của đường nội suy khác c liên tục O(1/N) (thấy trong T4a).
- Phương sai cầu phương E Var_U Â phụ thuộc hình học đường và mô hình; chưa có cận theo thống kê của Q.
- Không có kết quả nào chứng minh Dirichlet tối ưu; các định lý cho thấy những gì quyết định μ và V đều là thống kê chung (moment Stieltjes,
  D, c, cumulant), không gắn với một họ cụ thể.
