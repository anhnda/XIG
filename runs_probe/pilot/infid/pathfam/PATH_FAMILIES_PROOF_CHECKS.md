# PATH_FAMILIES — kiểm toán chứng minh Phase 2 (symbolic + Monte Carlo, CPU)

Tài liệu: docs/PATH_FAMILIES_THEORY.md, docs/PATH_FAMILIES_THEORY_AUDIT.md.

## S1 — Hệ quả 3a / 3b: từng mẫu chỉ số, tính trực tiếp trên họ warp LỆCH (không bất biến khi đảo, κ₃ ≠ 0)

Z_j = t + ½ s_j t(1 − t)², s ∈ {2 (xác suất 1/3), −1 (2/3)}: E Z = t, v(t) không đối xứng, κ₃ ≠ 0 ⇒ mọi số hạng ∫tv, ∫κ₃ của công thức đều khác 0, nên phép kiểm không suy biến.

| mẫu | trực tiếp | công thức | hiệu (phải = 0) |
|---|---|---|---|
| bậc 3: j ≠ k, cả hai ≠ i | 1/3 | 1/3 | 0 |
| bậc 3: j = k ≠ i | 71/210 | 71/210 | 0 |
| bậc 3: j = i ≠ k | 139/420 | 139/420 | 0 |
| bậc 3: j = k = i | 1/3 | 1/3 | 0 |
| bậc 4: j, k, l khác nhau, ≠ i | 1/4 | 1/4 | 0 |
| bậc 4: j = k ≠ l, ≠ i | 141/560 | 141/560 | 0 |
| bậc 4: j = k = l ≠ i | 859/3360 | 859/3360 | 0 |
| bậc 4: một = i, hai còn lại khác nhau | 139/560 | 139/560 | 0 |
| bậc 4: một = i, hai còn lại bằng nhau | 1/4 | 1/4 | 0 |
| bậc 4: hai = i | 2501/10080 | 2501/10080 | 0 |
| bậc 4: ba = i | 1/4 | 1/4 | 0 |
| tham số họ | D = 1/210, ∫tv = 1/560, ∫κ₃ = 1/3360 | D/2 = 1/420 | — |

## S2 — bất biến khi đảo ⇒ ∫tv = D/2, ∫κ₃ = 0 (họ warp đối xứng φ = t(1 − t), s = ±1)

| ∫ t v − D/2 | ∫ κ₃ |
|---|---|
| 0 | 0 |

## S3 — bậc 5 phụ thuộc κ₄ và ∫v² (họ warp lệch)

| E∫Z_j⁴dZ_i trực tiếp | ∫(t⁴ + 6t²v + 4tκ₃ + 3v² + κ₄) | hiệu | đóng góp κ₄ | đóng góp 3∫v² |
|---|---|---|---|---|
| 4931/24024 | 4931/24024 | 0 | -1/17160 | 1/8580 |

## S4 — Định lý 4: mọi kiểu chồng chỉ số (symbolic, họ hạng 2, Λ ≠ 0)

| kiểu | trực tiếp | định lý | hiệu |
|---|---|---|---|
| Var S_{e_j,i} | 1301/259200 | 1301/259200 | 0 |
| (j,i) & (k,i): chung integrator | 13/5184 | 13/5184 | 0 |
| (j,i) & (j,l): chung integrand | 13/5184 | 13/5184 | 0 |
| (j,i) & (i,l): integrator = integrand kia | -13/5184 | -13/5184 | 0 |
| (j,i) & (l,j): integrand = integrator kia | -13/5184 | -13/5184 | 0 |
| (j,i) & (i,j) | -1301/259200 | -1301/259200 | 0 |
| (j,i) & (k,l): rời nhau | 0 | 0 | 0 |
| Γ, Λ của họ | Γ = 13/5184 | Λ = 1/259200 | — |

## M1 — Định lý 4 trên dp lưới (h = ½, N = 64, Monte Carlo)

Γ̂ = Var ∫Z dt (đồng nhất thức ∬c = Var ∫Z dt, đúng cho mọi luật). Các hiệp phương sai phải bằng ±Γ̂ / 0 trong nhiễu MC; Γ̂, Λ̂ lệch nhẹ khỏi giá trị liên tục do rời rạc hoá.

| kiểu | MC | định lý (theo Γ̂) | lệch / độ lớn | SE thô ≈ |
|---|---|---|---|---|
| Var S_{e_j,i} | 0.103 | 0.103 | 0 | 4.62e-04 |
| (j,i) & (k,i) | 0.042 | 0.042 | 7.69e-04 | 4.62e-04 |
| (j,i) & (j,l) | 0.041 | 0.042 | -0.003 | 4.62e-04 |
| (j,i) & (i,l) | -0.042 | -0.042 | -7.69e-04 | 4.62e-04 |
| (j,i) & (l,j) | -0.041 | -0.042 | 0.003 | 4.62e-04 |
| (j,i) & (k,l) | 3.41e-04 | 0 | 0.003 | 4.62e-04 |
| Γ̂ = Var∫Z (lý thuyết liên tục h/12 = 0.04167) | 0.042 | 0.042 | -9.86e-04 | nan |
| Λ̂ = Var − 2Γ̂ (lý thuyết liên tục h²/12 = 0.02083) | 0.020 | 0.021 | -0.040 | nan |

## M2 — Mệnh đề Shapley: λ = 1, N = 4096, F bậc 3 ngẫu nhiên, d = 4

Đường được nén (bỏ đoạn đứng yên; sai khác so với lịch trình đầy đủ trên N = 16: 0.0e+00). Đụng độ (hai feature nhảy cùng đoạn ⇒ đi thẳng cùng nhau) gây lệch O(d²/N), nhỏ hơn nhiễu MC ở đây.

| feature | E A (MC) | Shapley chính xác | z | đụng độ kỳ vọng / đường |
|---|---|---|---|---|
| 0 | -1.382 | -1.385 | 0.483 | 0.001 |
| 1.000 | -1.885 | -1.877 | -1.218 | 0.001 |
| 2.000 | 1.570 | 1.574 | -1.467 | 0.001 |
| 3.000 | -3.130 | -3.140 | 1.555 | 0.001 |

