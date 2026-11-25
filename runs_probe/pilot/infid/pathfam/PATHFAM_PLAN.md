# PATH_FAMILIES — kế hoạch thí nghiệm cuối cùng + kiểm toán chi phí gradient (CPU, không mô hình)

Chi phí đo bằng chính pipeline `schedule → nodes → integrate(rand)` với hàm giả (đếm lượt forward + backward). `|A − w⊙Δx|` = sai số completeness với F tuyến tính (phải ≈ 0 với mọi họ: đường hợp lệ, cầu phương đúng).

| config | họ | N | h | K rev @B256 | K rev @B1024 | grad/đường (đo) | N·M | khớp | |A − w⊙Δx| (F tuyến tính) | đường dev | grad dev / ảnh | σ | λ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spi_a0.1_N32 | spi | 32 | 0.909 | 8 | 32 | 32 | 32 | ✓ | 4.44e-16 | 32 | 1024 | nan | nan |
| spi_a0.3_N32 | spi | 32 | 0.769 | 8 | 32 | 32 | 32 | ✓ | 4.44e-16 | 32 | 1024 | nan | nan |
| spi_a1_N32 | spi | 32 | 0.500 | 8 | 32 | 32 | 32 | ✓ | 1.78e-15 | 32 | 1024 | nan | nan |
| spi_a3_N32 | spi | 32 | 0.250 | 8 | 32 | 32 | 32 | ✓ | 4.44e-16 | 32 | 1024 | nan | nan |
| spi_a10_N32 | spi | 32 | 0.091 | 8 | 32 | 32 | 32 | ✓ | 4.44e-16 | 32 | 1024 | nan | nan |
| ln_a0.1_N32 | ln | 32 | 0.909 | 8 | 32 | 32 | 32 | ✓ | 4.44e-16 | 32 | 1024 | 23.6 | nan |
| ln_a0.3_N32 | ln | 32 | 0.769 | 8 | 32 | 32 | 32 | ✓ | 1.78e-15 | 32 | 1024 | 9.176 | nan |
| ln_a1_N32 | ln | 32 | 0.500 | 8 | 32 | 32 | 32 | ✓ | 8.88e-16 | 32 | 1024 | 3.991 | nan |
| ln_a3_N32 | ln | 32 | 0.250 | 8 | 32 | 32 | 32 | ✓ | 4.44e-16 | 32 | 1024 | 2.256 | nan |
| ln_a10_N32 | ln | 32 | 0.091 | 8 | 32 | 32 | 32 | ✓ | 4.44e-16 | 32 | 1024 | 1.366 | nan |
| cmix_a0.1_N32 | cmix | 32 | 0.909 | 8 | 32 | 32 | 32 | ✓ | 3.33e-16 | 32 | 1024 | nan | 0.909 |
| cmix_a0.3_N32 | cmix | 32 | 0.769 | 8 | 32 | 32 | 32 | ✓ | 3.33e-16 | 32 | 1024 | nan | 0.769 |
| cmix_a1_N32 | cmix | 32 | 0.500 | 8 | 32 | 32 | 32 | ✓ | 4.44e-16 | 32 | 1024 | nan | 0.500 |
| cmix_a3_N32 | cmix | 32 | 0.250 | 8 | 32 | 32 | 32 | ✓ | 4.44e-16 | 32 | 1024 | nan | 0.250 |
| cmix_a10_N32 | cmix | 32 | 0.091 | 8 | 32 | 32 | 32 | ✓ | 4.44e-16 | 32 | 1024 | nan | 0.091 |
| smooth_d0.5_N32 | smooth | 32 | 0.050 | 8 | 32 | 32 | 32 | ✓ | 4.44e-16 | 32 | 1024 | nan | nan |
| smooth_d0.674_N32 | smooth | 32 | 0.091 | 8 | 32 | 32 | 32 | ✓ | 4.44e-16 | 32 | 1024 | nan | nan |
| smooth_d1_N32 | smooth | 32 | 0.200 | 8 | 32 | 32 | 32 | ✓ | 4.44e-16 | 32 | 1024 | nan | nan |
| spi_a0.1_N64 | spi | 64 | 0.909 | 4 | 16 | 64 | 64 | ✓ | 4.44e-16 | 32 | 2048 | nan | nan |
| spi_a10_N64 | spi | 64 | 0.091 | 4 | 16 | 64 | 64 | ✓ | 4.44e-16 | 32 | 2048 | nan | nan |

**Dev**: 20 cấu hình, m = 16 cặp, 32 ảnh (fresh 1200–1231) ⇒ 22,528 grad / ảnh, tổng 720,896 grad (— (đặt pathfam.per_eval_seconds để quy ra phút)); so với BUDGET G1 524 288: 1.38×.

**Test** (cận trên): ≤ 6 nghiệm theo họ (N = 32) + incumbent + 2 đối chứng, m = 32 ⇒ ≤ 22,528 grad / ảnh; n ∈ [48, 128] (tính lúc khoá) ⇒ 1,081,344–2,883,584 grad (— (đặt pathfam.per_eval_seconds để quy ra phút) – — (đặt pathfam.per_eval_seconds để quy ra phút)), cộng metric ngoài (chỉ forward).

Trần GPU: dev 240 phút, test 300 phút.

