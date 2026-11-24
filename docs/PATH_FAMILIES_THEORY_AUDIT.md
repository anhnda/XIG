# PATH_FAMILIES_THEORY_AUDIT — kiểm toán chứng minh Phase 2

Mục đích: trước khi gọi phần lý thuyết là đóng góp, kiểm từng phát biểu của `PATH_FAMILIES_THEORY.md`: giả định có được nêu đủ không,
bước nào của chứng minh có thể hỏng, và kiểm độc lập bằng tính toán chính xác (sympy) hoặc Monte Carlo với tích phân đường chính xác.
Lệnh: `bash scripts/run.sh pathfam-proofcheck` (CPU, < 1 phút) → `PATH_FAMILIES_PROOF_CHECKS.md`; `pathfam-theory` → `PATH_FAMILIES_THEORY_CHECKS.md`.
Kiểm thử: `pytest tests/` 86 passed (gồm các kiểm symbolic và MC dưới đây).

## 1. Bảng kiểm toán

| Phát biểu | Giả định (đủ, đã nêu) | Bước dễ hỏng → cách xử lý | Kiểm độc lập | Kết quả |
|---|---|---|---|---|
| Định lý 1 (thống kê đủ, cần và đủ) | F đa thức bậc ≤ p; Z liên tục, đơn điệu | chiều "cần": phải được tự do chọn x′ = 0, Δ = 1 → đã ghi rõ | đại số trực tiếp | đúng |
| Định lý 2 (diện tích Lévy, từng đường) | F bậc 2; Z liên tục, BV | không có (đẳng thức đại số + Z_iZ_j\|₀¹ = 1) | MC từng đường, 6 họ gồm phụ thuộc / không bất biến | sai số ≤ 2·10⁻¹⁵ |
| Hệ quả 2b (bất biến khi đảo ⇒ μ = IG bậc 2) | RZ ~ Z | đổi biến s = 1 − t trong Stieltjes → kiểm 𝔄(RZ) = −𝔄(Z) | MC: \|z\| ≤ 1,5 (bất biến), `power` lệch rõ | đúng |
| Định lý 2c (`rev` chính xác bậc 2) | F bậc 2, mọi Q | không | MC từng đường | sai số ≤ 2·10⁻¹⁵ |
| Định lý 3 (độc lập ⇒ moment biên) | độc lập giữa feature; Z liên tục, đơn điệu | (a) quy tắc dây chuyền Z^s dZ chỉ cần liên tục — bản trước ghi "liên tục tuyệt đối", **đã sửa**; (b) Fubini + tích phân từng phần với G không âm, không giảm, bị chặn | sympy (S1) | đúng |
| Hệ quả 3a (bậc 3 qua D) | + E Z = t, cùng phân phối | phân phối khác nhau giữa feature → thay D bằng D_j, **đã ghi** | sympy: 4/4 mẫu hiệu = 0 trên họ KHÔNG bất biến | đúng |
| Hệ quả 3b (bậc 4: D, ∫tv, ∫κ₃; + đảo ⇒ D) | như trên | liệt kê đủ 7 mẫu chỉ số? → kiểm cả 7 | sympy: 7/7 hiệu = 0 với ∫tv ≠ D/2, ∫κ₃ ≠ 0 (không suy biến); S2: đảo ⇒ ∫tv = D/2, ∫κ₃ = 0 | đúng |
| Hệ quả 3c (bậc 5 phụ thuộc κ₄, ∫v²) | như trên | khai triển moment bậc 4 | sympy S3 (hiệu 0; đóng góp κ₄ ≠ 0); phản ví dụ số z = 31 | đúng |
| Mệnh đề 3d (giới hạn kích hoạt = Shapley) | khoảng nhảy rời nhau h.c.c., thứ tự là hoán vị đều | trên lưới có đụng độ → **đã ghi** "Shapley theo khối", và với ảnh thật (d ≫ N) đụng độ là chuẩn | MC λ = 1, N = 4096: \|z\| ≤ 1,6; nén đoạn đứng yên kiểm bằng 0,0 | đúng (giới hạn nêu rõ) |
| Định lý 4 (phương sai bậc 2 qua c) | độc lập, cùng phân phối, E Z = t, F bậc 2 | bản trước **thiếu** kiểu (j,i) & (l,j) = −Γ và (j,i) & (i,j) = −(2Γ + Λ) → **đã bổ sung**; Λ cần c có biến phân Vitali bị chặn (đúng với mọi họ đang xét) | sympy: 7/7 kiểu, họ hạng 2 có Λ ≠ 0; MC dp lưới (c có khối lượng đường chéo): khớp ±Γ̂ trong nhiễu | đúng |
| Bảng giá trị Định lý 4 (dp/cmix h/6 + h²/12, smooth δ²/18) | thời gian liên tục | đường lưới có c khác c liên tục O(1/N) | MC: lệch −1,9 % (N = 32) → −0,5 % (N = 128); smooth −0,1 % | đúng (với sai số rời rạc hoá nêu rõ) |
| Mệnh đề 5 (thứ hạng theo B đổi tối đa một lần) | J_indep, B liên tục | K phải nguyên (và chẵn với `rev`) → đã ghi | đại số | đúng |

## 2. Những gì KHÔNG được tuyên bố

- Không có định lý nào cho mạng ReLU thật (F tuyến tính từng mảnh): các định lý đa thức chỉ là mô hình cục bộ.
- Định lý 4 chỉ cho bậc 2 và feature độc lập; phương sai bậc ≥ 3 chỉ có bằng chứng số (dp ≠ cmix cùng μ, z = 71).
- Phương sai cầu phương (quy tắc `rand`) chưa có cận theo thống kê của Q.
- Không có kết quả tối ưu: lý thuyết nói thống kê nào quyết định μ và V, không nói họ nào là tốt nhất.

## 3. Sửa đổi trong THEORY.md sau kiểm toán

1. Giả định "liên tục tuyệt đối" → "liên tục" (đủ và đúng với đường lưới, kể cả bước nhảy nằm trong một đoạn).
2. Định lý 4: bổ sung hai kiểu chồng chỉ số còn thiếu; thêm đồng nhất thức Γ = Var ∫Z dt.
3. Hệ quả 3a–3b: ghi trường hợp phân phối khác nhau giữa feature (D_j).
4. "Diễn giải kích hoạt" nâng thành Mệnh đề 3d có chứng minh, kèm giới hạn đụng độ trên lưới.
