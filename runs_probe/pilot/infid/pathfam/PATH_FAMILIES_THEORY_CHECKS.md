# PATH_FAMILIES Phase 2 — kiểm số các định lý ở mức phân phối đường (CPU)

Tích phân đường chính xác trên đường nội suy tuyến tính (Gauss–Legendre), không mô hình. Định lý và chứng minh: docs/PATH_FAMILIES_THEORY.md.

## T2 + T5 — F bậc 2: đồng nhất thức diện tích Lévy (từng đường), bất biến khi đảo ⇒ μ = IG, cặp rev chính xác

`group`: các feature trong nhóm dùng chung một lịch trình dp (phụ thuộc, vẫn bất biến khi đảo). `power`: Z_i = t^{a_i} (a = 0,5; 2; 1; 3) — không bất biến khi đảo ⇒ E 𝔄 ≠ 0 ⇒ μ ≠ IG. Hai cột đầu phải ở mức sai số máy (~1e-13).

| họ | phụ thuộc giữa feature | bất biến khi đảo | max |A − IG − Δ(H∘𝔄)Δ| (từng đường) | max |A(Z) + A(RZ) − 2 IG| (từng đường) | max |E 𝔄_ij| | max |z| của E 𝔄 (≈ 0 nếu bất biến) | ‖E A − IG‖ |
|---|---|---|---|---|---|---|---|
| dp | ✗ | ✓ | 1.55e-15 | 1.78e-15 | 9.77e-04 | 0.967 | 0.002 |
| cmix | ✗ | ✓ | 1.22e-15 | 1.78e-15 | 0.001 | 1.201 | 0.001 |
| ln | ✗ | ✓ | 1.67e-15 | 1.78e-15 | 0.001 | 1.454 | 0.002 |
| smooth | ✗ | ✓ | 5.55e-16 | 8.88e-16 | 7.43e-04 | 0.996 | 0.001 |
| group | ✓ | ✓ | 1.94e-15 | 1.78e-15 | 1.88e-04 | 0.186 | 4.32e-04 |
| power | ✗ | ✗ | 5.41e-16 | 8.88e-16 | 0.357 | tất định | 0.346 |

## T4a — phương sai bậc 2 chỉ phụ thuộc hàm hiệp phương sai c(s, t)

Var ∫Z₁dZ₀ (hai feature độc lập, N = 32). Lý thuyết liên tục: dp và cmix (cùng c = h(min − st)) h/6 + h²/12; smooth δ²/18. Lệch nhỏ còn lại là do rời rạc hoá N = 32 (c của đường nội suy ≠ c liên tục) và nhiễu MC. ln khớp D nhưng c khác ⇒ giá trị khác.

| họ | N | h | Var ∫Z₁dZ₀ (MC) | SE | lý thuyết (liên tục) | lệch tương đối | D (nội suy) | D liên tục |
|---|---|---|---|---|---|---|---|---|
| dp | 32 | 0.500 | 0.102 | 2.28e-04 | 0.104 | -0.019 | 0.081 | 0.083 |
| cmix | 32 | 0.500 | 0.102 | 2.28e-04 | 0.104 | -0.020 | 0.081 | 0.083 |
| ln | 32 | 0.500 | 0.102 | 2.29e-04 | nan | nan | 0.081 | 0.083 |
| smooth | 32 | 0.200 | 0.055 | 1.24e-04 | 0.056 | -0.001 | 0.033 | 0.033 |
| dp | 128 | 0.500 | 0.104 | 4.63e-04 | 0.104 | -0.005 | 0.083 | 0.083 |
| cmix | 128 | 0.500 | 0.103 | 4.61e-04 | 0.104 | -0.011 | 0.082 | 0.083 |

## T4b — F bậc 3: cùng μ, khác phương sai

| so sánh | max |z| E A (≈ 0: cùng μ) | max |z| Var A (lớn: khác phương sai) | Var A dp | Var A cmix |
|---|---|---|---|---|
| cmix λ=½ − dp α=1, F bậc 3 | 1.448 | 70.8 | 0.8211, 0.0429, 0.6239 | 0.8214, 0.0593, 0.6389 |

