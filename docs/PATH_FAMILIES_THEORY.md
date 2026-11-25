# Stochastic feature-wise path attribution — lý thuyết ở mức phân phối đường

Bản định lý–chứng minh dùng cho paper. Đối tượng là **phân phối Q của lịch trình đường theo feature**, không phải một họ cụ thể;
Dirichlet (DP lưới), logistic-normal (`ln`), coordinate-mix (`cmix`), `smooth`, nhóm (`hier`/`patch`) chỉ là ví dụ.
Mọi phát biểu gắn nhãn:

- **Định lý / Mệnh đề / Bổ đề / Hệ quả** — chứng minh đầy đủ dưới các giả thiết được liệt kê (§2).
- **Kiểm số** — kiểm độc lập bằng tính toán chính xác (sympy) hoặc Monte Carlo với tích phân chính xác, CPU
  (`pathfam-theory`, `pathfam-proofcheck`, `pathfam-final-audit`, `tests/`). Kiểm số không thay chứng minh; nó chỉ bắt lỗi đại số.
- **Giới hạn** — điều không được tuyên bố.

Lịch sử kiểm toán chứng minh (các lỗi đã sửa): `docs/PATH_FAMILIES_THEORY_AUDIT.md`.

---

## 1. Thiết lập và ký hiệu

Mô hình $F:\mathbb R^d\to\mathbb R$, đầu vào $x$, nền $x'$, $\Delta=x-x'$, $\Delta F=F(x)-F(x')$. Feature $i=1,\dots,d$ (pixel × kênh).

**Lịch trình** $Z=(Z_1,\dots,Z_d):[0,1]\to[0,1]^d$ là một phần tử ngẫu nhiên với luật $Q$; đường $\gamma(t)=x'+\Delta\odot Z(t)$.

**Attribution theo đường** (tích phân Riemann–Stieltjes):
$$A_i(Z)=\Delta_i\int_0^1\partial_iF(\gamma(t))\,dZ_i(t),\qquad \mu_Q=\mathbb E_Q A(Z).$$
IG là trường hợp $Z_i(t)=t$ với mọi $i$.

**Moment Stieltjes.** Với đa chỉ số $a\in\mathbb N^d$, $Z^a=\prod_j Z_j^{a_j}$:
$\;S_{a,i}(Z)=\int_0^1 Z(t)^a\,dZ_i(t)$, $\;M_{a,i}(Q)=\mathbb E_Q S_{a,i}$.

**Diện tích Lévy** của cặp $(i,j)$: $\mathfrak A_{ij}(Z)=\tfrac12\int_0^1(Z_j\,dZ_i-Z_i\,dZ_j)$.

**Phép đảo** $(RZ)(t)=\mathbf 1-Z(1-t)$ (từng tọa độ).

**Thống kê biên** (khi các tọa độ cùng luật biên): $v(t)=\operatorname{Var}Z_i(t)$, $D=\int_0^1v$, $\kappa_3(t),\kappa_4(t)$ cumulant bậc 3, 4
của $Z_i(t)$, hàm hiệp phương sai $c(s,t)=\operatorname{Cov}(Z_i(s),Z_i(t))$, $\Gamma=\iint c$, $\Lambda=\iint c\,d^2c$
($d^2c$: độ đo Lebesgue–Stieltjes sinh bởi $\partial^2c/\partial s\partial t$).

**Ước lượng ngân sách hữu hạn.** Lưới đều $t_n=n/N$; đường nội suy tuyến tính $Z^N$ qua các nút $Z(t_n)$; trên đoạn $n$:
$\gamma_n(u)=\gamma(t_n)+u\,\delta\gamma_n$, $\delta\gamma_n=\gamma(t_{n+1})-\gamma(t_n)$. Quy tắc `rand` ($M=1$):
$$\hat A_i(Z,U)=\sum_{n=0}^{N-1}\partial_iF(\gamma_n(U_n))\,\delta\gamma_{n,i},\qquad U_n\overset{iid}{\sim}\mathcal U(0,1)\text{ (chung mọi feature)}.$$
Chi phí $N$ gradient/đường; ngân sách $B=KN$. Hai thiết kế: `indep` ($K$ đường độc lập) và `rev` ($K/2$ cặp $(Z,RZ)$ độc lập, $U$ độc lập
giữa hai thành viên).

**Infidelity.** Với một tập phép xoá $\mathcal D$ (các ô $S_1,\dots,S_m$): $P_{\mathcal D}a=(\sum_{i\in S_k}a_i)_k$, $\delta_{\mathcal D}$ là vectơ hiệu
đầu ra khi xoá từng ô,
$$L_{\mathcal D}(a)=\frac{\|P_{\mathcal D}a-\delta_{\mathcal D}\|^2}{\|\delta_{\mathcal D}\|^2},\qquad J_{\mathcal D}(Q,B)=\mathbb E\,L_{\mathcal D}(\bar A_K).$$

---

## 2. Giả thiết

| Mã | Giả thiết | Dùng ở |
|---|---|---|
| **(P)** | Mỗi $Z_i$ liên tục, không giảm, $Z_i(0)=0$, $Z_i(1)=1$ (h.c.c.) | mọi kết quả |
| **(F1)** | $F\in C^1$ trên hộp $[x',x]$ | B2, B5, P3.4, §7 |
| **(F1′)** | $F$ Lipschitz và trên mỗi đoạn thẳng của đường, tập $\{u: F$ không khả vi tại $\gamma_n(u)\}$ có độ đo 0 (gradient tự động lấy giá trị bị chặn bất kỳ ở đó) | thay (F1) cho mạng ReLU ở B5, §7 |
| **(Poly$_p$)** | $F$ là đa thức bậc $\le p$ | Đ1–Đ4 |
| **(I)** | Các tọa độ $Z_1,\dots,Z_d$ độc lập | Đ3, Đ4 |
| **(S)** | Bất biến khi đảo: $RZ\overset{d}{=}Z$ | H2.2, H3.2 |
| **(M)** | Lịch trình trung bình tuyến tính: $\mathbb E Z_i(t)=t$ | H3.1–H3.2, Đ4 |
| **(H)** | Mọi tọa độ cùng luật biên (nếu không: thay $D,v,\dots$ bằng $D_j,v_j,\dots$ của đúng feature) | H3.1–H3.2, Đ4 (dạng gọn) |
| **(E)** | Tích phân chính xác (không cầu phương) | §3–§5 |
| **(G)** | Lưới đều + quy tắc `rand` như §1; các đường/cặp độc lập | §6, §7 |

Ghi chú: (P) cho phép bước nhảy *bên trong* một đoạn lưới (đường nội suy vẫn liên tục); không cần liên tục tuyệt đối. Nội suy tuyến tính
bảo toàn (P); bảo toàn (I), (M) tại nút; bảo toàn (S) vì lưới đều đối xứng. Do đó mọi kết quả ở §3–§5 áp dụng cho **luật nội suy $Q^N$** —
đúng đối tượng mà ước lượng `rand` nhắm tới (B5).

---

## 3. Bổ đề nền

**Bổ đề B1 (quy tắc dây chuyền Stieltjes).** Nếu $Z_i$ liên tục, không giảm và $\varphi$ liên tục với nguyên hàm $\Phi$ thì
$\int_0^1\varphi(Z_i)\,dZ_i=\Phi(Z_i(1))-\Phi(Z_i(0))$. Nói riêng, $\int Z_i^a\,dZ_i=1/(a+1)$ theo từng đường.

*Chứng minh.* Với phân hoạch $\{\tau_k\}$, định lý giá trị trung bình cho $\Phi(Z_i(\tau_{k+1}))-\Phi(Z_i(\tau_k))=\varphi(\xi_k)\,[Z_i(\tau_{k+1})-Z_i(\tau_k)]$
với $\xi_k$ giữa $Z_i(\tau_k)$ và $Z_i(\tau_{k+1})$; do $Z_i$ liên tục, $\xi_k=Z_i(\tau_k^*)$ với $\tau_k^*\in[\tau_k,\tau_{k+1}]$ (định lý giá trị trung gian),
nên tổng telescoping là một tổng Riemann–Stieltjes; qua giới hạn. ∎
(Với bước nhảy thật, đẳng thức sai — lý do (P) đòi liên tục.)

**Bổ đề B2 (completeness theo từng đường).** Dưới (P), (F1): $\sum_iA_i(Z)=\Delta F$ với mọi $Z$.

*Chứng minh.* $\gamma$ liên tục, biến phân bị chặn; với $F\in C^1$, quy tắc dây chuyền cho tích phân Stieltjes theo đường BV liên tục
$F(\gamma(1))-F(\gamma(0))=\sum_i\int\partial_iF(\gamma)\,d\gamma_i$ (giá trị trung bình trên từng mảnh phân hoạch + liên tục đều của $\nabla F$). ∎

**Bổ đề B3 (tích phân từng phần).** Dưới (P): $S_{e_j,i}+S_{e_i,j}=1$, nên $S_{e_j,i}=\tfrac12+\mathfrak A_{ij}$.

*Chứng minh.* $\int Z_j\,dZ_i+\int Z_i\,dZ_j=[Z_iZ_j]_0^1=1$ (tích phân từng phần cho hai hàm liên tục BV). ∎

**Bổ đề B4 (phép đảo).** Dưới (P): $S_{a,i}(RZ)=\int_0^1\prod_j(1-Z_j(s))^{a_j}\,dZ_i(s)$. Nói riêng $S_{e_j,i}(RZ)=1-S_{e_j,i}(Z)$ và
$\mathfrak A_{ij}(RZ)=-\mathfrak A_{ij}(Z)$.

*Chứng minh.* Đổi biến $s=1-t$: $d[1-Z_i(1-t)]$ với $t$ tăng tương ứng $dZ_i(s)$ với $s$ tăng (hai dấu trừ triệt tiêu). Áp B3. ∎

**Bổ đề B5 (cầu phương `rand` không chệch).** Dưới (P), (F1) hoặc (F1′), (G): $\mathbb E[\hat A(Z,U)\mid Z]=A(Z^N)$ — attribution chính xác trên
đường nội suy.

*Chứng minh.* Trên đoạn $n$, $dZ^N_i=\delta\gamma_{n,i}/\Delta_i\,du$ nên đóng góp chính xác của đoạn là
$\delta\gamma_{n,i}\int_0^1\partial_iF(\gamma_n(u))\,du=\mathbb E_{U_n}[\partial_iF(\gamma_n(U_n))]\,\delta\gamma_{n,i}$. Dưới (F1′) tích phân được hiểu theo Lebesgue,
tập điểm không khả vi có độ đo 0 không ảnh hưởng. Việc dùng chung $U_n$ cho mọi feature không ảnh hưởng kỳ vọng. ∎

---

## 4. Attribution kỳ vọng

### Định lý 1 (thống kê đủ, cần và đủ)

Dưới (P), (Poly$_p$): $A_i(Z)$ là tổ hợp affine của $\{S_{a,i}(Z):|a|\le p-1\}$ với hệ số chỉ phụ thuộc $(F,x',\Delta)$. Do đó $\mu_Q$ phụ thuộc
$Q$ chỉ qua $\{M_{a,i}(Q):|a|\le p-1\}$, và $\operatorname{Cov}_Q(A)$ chỉ qua các moment bậc hai $\mathbb E[S_{a,i}S_{b,l}]$. Ngược lại, nếu
$M_{a,i}(Q)\ne M_{a,i}(Q')$ với một $|a|\le p-1$ thì tồn tại $F$ bậc $\le p$, $x'$, $\Delta$ với $\mu_Q\ne\mu_{Q'}$.

*Chứng minh.* $\partial_iF(x'+\Delta\odot Z)$ là đa thức bậc $\le p-1$ theo $Z$; khai triển thành đơn thức $Z^a$ và tích phân theo $dZ_i$
(tuyến tính). Chiều ngược: $x'=0$, $\Delta=\mathbf 1$, $F=x^{a+e_i}$ cho $\partial_iF=(a_i+1)x^a$, nên $A_i=(a_i+1)S_{a,i}$. ∎

*Nhận xét.* $S_{a,i}$ là các tọa độ "tiền tố đối xứng" của chữ ký (signature) cấp $|a|+1$ của $Z$; $\mu_Q$ là hàm tuyến tính của một tập con
của chữ ký kỳ vọng.

### Định lý 2 (bậc 2: diện tích Lévy, theo từng đường)

Dưới (P), (Poly$_2$) với Hessian $H$: $\;A_i(Z)-\mathrm{IG}_i=\Delta_i\sum_{j\ne i}H_{ij}\Delta_j\,\mathfrak A_{ij}(Z)$.

*Chứng minh.* $\partial_iF(\gamma)=c_i+\sum_jH_{ij}\Delta_jZ_j$ với $c_i$ không phụ thuộc $Z$; $\int dZ_i=1$; B1 cho $\int Z_i\,dZ_i=\tfrac12$;
B3 cho $\int Z_j\,dZ_i=\tfrac12+\mathfrak A_{ij}$. IG ứng với $\mathfrak A\equiv0$. ∎

**Hệ quả 2.1.** $\mu_Q=\mathrm{IG}$ với mọi $F$ bậc 2 $\iff\mathbb E_Q\mathfrak A_{ij}=0$ với mọi $i\ne j$.
(⇐ từ Đ2; ⇒ chọn $H$ chỉ có $H_{ij}=H_{ji}=1$, $\Delta=\mathbf 1$.)

**Hệ quả 2.2 (bất biến khi đảo).** Dưới (P), (S): $\mathbb E\mathfrak A_{ij}=0$, do đó $\mu_Q=\mathrm{IG}$ cho mọi $F$ bậc 2 —
**không cần (I), không cần (M)**. (B4: $\mathfrak A(RZ)=-\mathfrak A(Z)$; $|\mathfrak A|\le1$ khả tích.)

**Định lý 2.3 (cặp `rev` chính xác ở bậc 2).** Dưới (P), (Poly$_2$), với MỌI $Q$: $A(Z)+A(RZ)=2\,\mathrm{IG}$ theo từng đường.
*Chứng minh.* Đ2 + B4. ∎ Do đó trung bình cặp không có phương sai *đường* ở bậc 2 (xem H5.1).

**Kiểm số.** Sai lệch từng đường của Đ2 và Đ2.3 $\le2\cdot10^{-15}$ trên dp, cmix, ln, smooth, họ phụ thuộc `group`, họ không bất biến `power`;
`power` có $\|\mathbb E A-\mathrm{IG}\|=0{,}35$ (khớp H2.1).

### Định lý 3 (feature độc lập: moment biên quyết định μ)

Dưới (P), (I), với $m_{j,r}(t)=\mathbb E Z_j(t)^r$:
$$M_{a,i}(Q)=\frac{1}{a_i+1}\int_0^1\prod_{j\ne i}m_{j,a_j}(t)\,dm_{i,a_i+1}(t).$$
Do đó dưới (Poly$_p$), $\mu_Q$ chỉ phụ thuộc các **đường moment biên** $m_{j,r}$, $r\le p$.

*Chứng minh.* B1: $Z_i^{a_i}dZ_i=dW/(a_i+1)$ với $W=Z_i^{a_i+1}$ liên tục, không giảm. Đặt $G=\prod_{j\ne i}Z_j^{a_j}$: liên tục, không giảm, trong $[0,1]$,
độc lập với $W$ theo (I). Tích phân từng phần: $\int G\,dW=1-\int W\,dG$. Có điều kiện theo $G$, tích phân theo độ đo cố định $dG$ là phiếm hàm
tuyến tính dương, $0\le W\le1$, nên (hội tụ trội cho tổng Riemann–Stieltjes) $\mathbb E[\int W\,dG\mid G]=\int m_W\,dG$ với $m_W=\mathbb E W$
(liên tục theo hội tụ trội). Tích phân từng phần lần nữa: $\int m_W\,dG=1-\int G\,dm_W$. Lấy kỳ vọng theo $G$ (Fubini, $dm_W$ dương):
$\mathbb E\int G\,dW=\int\mathbb E G\,dm_W$, và $\mathbb E G(t)=\prod_{j\ne i}m_{j,a_j}(t)$ theo (I). ∎

**Hệ quả 3.1 (bậc 3 qua D).** Dưới (P), (I), (M), (H) và (Poly$_3$): $\mu_Q$ phụ thuộc $Q$ chỉ qua $D$. Bốn mẫu chỉ số ($j,k\ne i$, $j\ne k$):

| $S_{a,i}$ | $Z_jZ_k\,dZ_i$ | $Z_j^2\,dZ_i$ | $Z_iZ_j\,dZ_i$ | $Z_i^2\,dZ_i$ |
|---|---|---|---|---|
| $M_{a,i}$ | $1/3$ | $1/3+D$ | $1/3-D/2$ | $1/3$ |

*Chứng minh.* Đ3 với $m_1=t$, $m_2=t^2+v$, $v(0)=v(1)=0$; ví dụ $\tfrac12\int t\,d(t^2+v)=\tfrac13+\tfrac12[tv]_0^1-\tfrac12\int v$. ∎

**Hệ quả 3.2 (bậc 4).** Dưới (P), (I), (M), (H) và (Poly$_4$): $\mu_Q$ phụ thuộc $Q$ qua $D$, $\int tv$, $\int\kappa_3$. Thêm (S) thì $v(1-t)=v(t)$,
$\kappa_3(1-t)=-\kappa_3(t)$, nên $\int tv=D/2$, $\int\kappa_3=0$ và **chỉ còn D**. Bảy mẫu ($j,k,l$ khác nhau, khác $i$):

| mẫu | $Z_jZ_kZ_l$ | $Z_j^2Z_k$ | $Z_j^3$ | $Z_iZ_jZ_k$ | $Z_iZ_j^2$ | $Z_i^2Z_j$ | $Z_i^3$ |
|---|---|---|---|---|---|---|---|
| tổng quát | $\tfrac14$ | $\tfrac14+\int tv$ | $\tfrac14+3\!\int tv+\int\kappa_3$ | $\tfrac14-\int tv$ | $\tfrac14$ | $\tfrac14-\int tv-\tfrac13\int\kappa_3$ | $\tfrac14$ |
| + (S) | $\tfrac14$ | $\tfrac14+\tfrac D2$ | $\tfrac14+\tfrac{3D}2$ | $\tfrac14-\tfrac D2$ | $\tfrac14$ | $\tfrac14-\tfrac D2$ | $\tfrac14$ |

*Chứng minh.* Đ3 với $m_3=t^3+3tv+\kappa_3$; ví dụ $\tfrac13\int t\,dm_3=\tfrac13(1-\int m_3)$; mẫu $Z_iZ_j^2$ cho $\tfrac12\int m_2\,dm_2=\tfrac14$ nhờ (H). ∎
**Kiểm số.** sympy tính trực tiếp 4 + 7 mẫu trên một họ warp KHÔNG bất biến ($\int tv\ne D/2$, $\int\kappa_3\ne0$): hiệu với bảng bằng 0.

**Mệnh đề 3.3 (bậc 5: D không còn đủ).** Dưới (P), (I), (M), (H), tồn tại $F$ bậc 5 và hai luật cùng $D$, cùng hàm hiệp phương sai $c$, cùng
(S), cho $\mu$ khác nhau.

*Chứng minh (giải tích).* $F=x_0(x_1-\tfrac12)^4$, $x'=0$, $\Delta=\mathbf 1$: $\mathbb E A_0=\int_0^1\mathbb E(Z_1(t)-\tfrac12)^4dt$.
(a) DP liên tục $\alpha=1$: $Z(t)\sim\mathrm{Beta}(t,1-t)$, $v=t(1-t)/2$, $D=1/12$; moment $\mathbb E Z^k=\prod_{r<k}(t+r)/(1+r)$ cho
$\mathbb E A_0=13/360$. (b) cmix $\lambda=\tfrac12$ (với xác suất $\lambda$ nhảy trọn tại $T\sim\mathcal U(0,1)$, ngược lại $Z=t$): $v=\lambda t(1-t)$,
$D=1/12$, $c(s,t)=\lambda\,s(1-t)$ (giống DP, $s\le t$); $\mathbb E A_0=\lambda/16+(1-\lambda)/80=3/80=13{,}5/360$. IG: $1/80$. ∎
Số hạng phân biệt là $\int\mathbb E(Z-t)^4$, tức $\int(\kappa_4+3v^2)$ — thống kê đuôi/độ nhọn, không phải hiệp phương sai.

**Mệnh đề 3.4 (giới hạn kích hoạt = Shapley).** Giả sử (P), (F1) hoặc (F1′) và: mỗi tọa độ chỉ chuyển động trong một khoảng thời gian riêng,
các khoảng của các feature rời nhau h.c.c., và thứ tự kích hoạt là hoán vị đều. Khi đó với mọi $F$ (không cần đa thức), $\mathbb EA_i$ là giá trị
Shapley của trò chơi $v(S)=F(x'+\Delta\odot\mathbf 1_S)$.

*Chứng minh.* Trong khoảng của $i$ chỉ $Z_i$ chuyển động, nên $A_i=F(\text{sau})-F(\text{trước})$ (B2 trên đoạn) — đóng góp biên của $i$ vào
tập đã kích hoạt; lấy kỳ vọng theo hoán vị đều. ∎ Trên lưới $N$ đoạn, các feature kích hoạt cùng một đoạn đi cùng nhau ⇒ **Shapley theo khối**;
với ảnh thật ($d\approx1{,}5\cdot10^5\gg N$) đụng độ là chuẩn, nên cmix/DP ở $h\to1$ tiến tới Shapley theo $N$ khối, không phải Shapley.

---

## 5. Hiệp phương sai của attribution

### Định lý 4 (bậc 2: hiệp phương sai qua hàm $c$)

Dưới (P), (I), (M), (H), $c$ có biến phân Vitali bị chặn. Với $j\ne i$, $k\ne l$:

| chồng chỉ số giữa $(j,i)$ và $(k,l)$ | $\operatorname{Cov}(S_{e_j,i},S_{e_k,l})$ |
|---|---|
| trùng hoàn toàn $(k,l)=(j,i)$ | $2\Gamma+\Lambda$ |
| chung integrator $l=i$, $k\ne j$ | $\Gamma$ |
| chung integrand $k=j$, $l\ne i$ | $\Gamma$ |
| integrator của cái này là integrand của cái kia ($k=i$, $l\ne j$) hoặc ngược lại ($l=j$, $k\ne i$) | $-\Gamma$ |
| đảo cặp $(k,l)=(i,j)$ | $-(2\Gamma+\Lambda)$ |
| bốn chỉ số khác nhau | $0$ |

Do đó, với $F$ bậc 2, $\operatorname{Cov}_Q(A)=\big(\Delta_i\Delta_l\sum_{j,k}H_{ij}H_{lk}\Delta_j\Delta_k\operatorname{Cov}(S_{e_j,i},S_{e_k,l})\big)_{il}$
phụ thuộc $Q$ chỉ qua **hàm hai biến $c$** (qua $\Gamma$, $\Lambda$), không qua số $D$. Đồng nhất thức: $\Gamma=\operatorname{Var}\int_0^1Z_i(t)\,dt$.

*Chứng minh.* Phương sai: $\mathbb E(\int Z_j\,dZ_i)^2=\iint\mathbb E[Z_j(s)Z_j(t)]\,d^2\mathbb E[Z_i(s)Z_i(t)]=\iint(st+c)\,d^2(st+c)$; bốn số hạng
$\tfrac14,\Gamma,\Gamma,\Lambda$ (tích phân từng phần hai lần; $c=0$ khi $s$ hoặc $t\in\{0,1\}$ vì $Z(0),Z(1)$ tất định). Trừ $(\tfrac12)^2$.
Các kiểu còn lại: điều kiện theo feature chung, dùng (M) cho feature không chung và tích phân từng phần một lần (ví dụ $l=j$:
$\iint t\,\partial_tR_j=\int R_j(s,1)ds-\iint R_j=\tfrac14-\Gamma$ với $R_j=st+c$). Đảo cặp: B3. ∎

| Họ | $c(s,t)$, $s\le t$ | $\Gamma$ | $\Lambda$ | $\operatorname{Var}S_{e_j,i}$ |
|---|---|---|---|---|
| DP liên tục ($h=1/(\alpha+1)$), cmix ($\lambda=h$) | $h\,s(1-t)$ | $h/12$ | $h^2/12$ | $h/6+h^2/12$ |
| smooth ($\delta$, $D=\delta^2/30$) | $\delta^2 s(1-s)t(1-t)$ | $\delta^2/36$ | $0$ | $\delta^2/18$ |
| IG | 0 | 0 | 0 | 0 |

**Hệ quả 4.1.** DP và cmix cùng $h$ có cùng $c$ ⇒ cùng $\mu$ tới bậc 4 (H3.2) và cùng $\operatorname{Cov}A$ bậc 2; khác biệt chỉ đến từ bậc $\ge3$
(phương sai) hoặc $\ge5$ (kỳ vọng, M3.3).
**Hệ quả 4.2.** Ở cùng $D$ (DP: $D=h/6$; smooth: $D=\delta^2/30$, tức $\delta^2=5h$), smooth có $\operatorname{Var}S=5h/18$ so với $h/6+h^2/12$
của DP; hiệu $h/9-h^2/12>0$ với mọi $h<4/3$, nên smooth luôn có phương sai bậc 2 lớn hơn trong miền đơn điệu của nó ($\delta\le1\iff h\le1/5$).
Ví dụ $h=0{,}091$: $0{,}0253$ so với $0{,}0159$.
**Kiểm số.** MC $4\cdot10^5$ đường: DP/cmix lưới lệch $-1{,}9\%/-2{,}0\%$ ($N=32$) và $-0{,}5\%/-1{,}1\%$ ($N=128$) so với giá trị liên tục ở $h=\tfrac12$
(lệch do $c$ của đường nội suy, giảm theo $N$); smooth $-0{,}1\%$. sympy kiểm cả 7 kiểu trên một họ hạng 2 có $\Lambda\ne0$.

---

## 6. Ước lượng ngân sách hữu hạn

**Mệnh đề 6.1 (phân rã phương sai).** Dưới (G): $\operatorname{Cov}(\hat A)=\operatorname{Cov}_Q\big(A(Z^N)\big)+\mathbb E_Q\operatorname{Cov}_U(\hat A\mid Z)$
(luật tổng phương sai + B5): phần **đường** và phần **cầu phương**.

**Định lý 6.2 (đồng nhất thức J).** Dưới (G), với $\Sigma=\operatorname{Cov}(\hat A)$ (một đường) và $\Sigma_{\rm pair}=\operatorname{Cov}\big(\tfrac12[\hat A(Z,U)+\hat A(RZ,U')]\big)$:
$$J_{\rm indep}=L(\mu)+\frac{V_1}{K},\quad V_1=\frac{\operatorname{tr}P\Sigma P^\top}{\|\delta\|^2};\qquad J_{\rm rev}=L(\mu)+\frac{2V_e}{K},\quad V_e=\frac{\operatorname{tr}P\Sigma_{\rm pair}P^\top}{\|\delta\|^2},$$
với $\mu=\mu_{Q^N}$ (dưới (S), trung bình cặp cũng có kỳ vọng $\mu$). Đẳng thức chính xác, không tiệm cận.

*Chứng minh.* $\bar A_K$ là trung bình của $K$ (hoặc $K/2$) vectơ iid có kỳ vọng $\mu$; $\mathbb E\|P\bar A_K-\delta\|^2=\|P\mu-\delta\|^2+\operatorname{tr}P\operatorname{Cov}(\bar A_K)P^\top$. ∎

**Hệ quả 6.3 (`rev` ở bậc 2).** Dưới (Poly$_2$), (S), (G): $\Sigma_{\rm pair}=\tfrac12\mathbb E\operatorname{Cov}_U(\hat A\mid Z)$ — chỉ còn phần cầu phương (Đ2.3).
Do đó $\rho=2V_e/V_1=\dfrac{\operatorname{tr}P\,\mathbb E\operatorname{Cov}_U\,P^\top}{\operatorname{tr}P\Sigma P^\top}\le1$. Ở mô hình thật, $\rho\approx1$ nghĩa là phần
bậc 2 của phương sai đường nhỏ so với phần bậc $\ge3$ cộng phần cầu phương — một diễn giải, không phải kiểm định.

**Mệnh đề 6.4 (thứ hạng theo ngân sách đổi tối đa một lần).** Với hai cấu hình (chi phí $N_1,N_2$ gradient/đường),
$J(\theta_1,B)-J(\theta_2,B)=[L(\mu_1)-L(\mu_2)]+[N_1V_{1}^{(1)}-N_2V_1^{(2)}]/B$ là affine theo $1/B$ ⇒ đổi dấu nhiều nhất một lần, tại
$B^*=(N_2V^{(2)}-N_1V^{(1)})/(L(\mu_1)-L(\mu_2))$ nếu $B^*>0$. (Coi $B$ liên tục; $K$ thực phải nguyên, chẵn với `rev`; với `rev` thay $V_1$ bằng $2V_e$.)
Hai tọa độ so sánh họ đường: $\big(L(\mu_Q),\,N\cdot V(Q)\big)$ — "lệch" và "phương sai × chi phí mỗi đường".

**Mệnh đề 6.5 (thang đo và hướng).** Với $y=P a$, $d=\delta$, $r=\|y\|/\|d\|$, $\cos=\langle y,d\rangle/(\|y\|\|d\|)$:
$$L(\beta a)=\beta^2r^2-2\beta r\cos+1,\qquad L(a)=\underbrace{(r-\cos)^2}_{\text{lệch thang đo}}+\underbrace{1-\cos^2}_{\text{lệch hướng}},\qquad \min_\beta L=1-\cos^2\text{ tại }\beta^*=\cos/r.$$
Bản đồ 0 có $L=1$; $L(a)<1\iff r<2\cos$. Vì $1-\cos^2\le1$, **$L(a)-1\le(r-\cos)^2$**: mọi phần của $L$ vượt quá 1 là lệch thang đo.
*Chứng minh.* Khai triển $\|\beta y-d\|^2/\|d\|^2$. ∎ Áp dụng cho từng thực hiện $\bar A_K$ nên cũng cho $J$ (trung bình).
Hệ quả: so sánh $J$ thô giữa hai cấu hình trộn lẫn khác biệt về chuẩn của $P\bar A_K$ (gồm cả phần do phương sai, vì $\mathbb E\|P\bar A_K\|^2=\|P\mu\|^2+\operatorname{tr}P\operatorname{Cov}P^\top$)
và khác biệt về hướng.

---

## 7. Completeness của ước lượng: lý thuyết cho sai số tuyệt đối

Đặt $e=\sum_i\hat A_i-\Delta F$ cho một đường, $e_K$ cho ước lượng `rev` $K$ đường.

**Mệnh đề 7.1 (sai số completeness chỉ là nhiễu cầu phương).** Dưới (P), (F1) hoặc (F1′), (E) cho $\Delta F$ và số học chính xác, (G):
với $h_n(u)=\nabla F(\gamma_n(u))\cdot\delta\gamma_n$,
$$e=\sum_n\big[h_n(U_n)-\textstyle\int_0^1h_n\big],\qquad\mathbb E[e\mid Z]=0,\qquad\operatorname{Var}(e\mid Z)=\sum_n\operatorname{Var}h_n(U_n).$$
*Chứng minh.* $\int_0^1h_n=F(\gamma(t_{n+1}))-F(\gamma(t_n))$ (B2 trên đoạn; dưới (F1′), $F\circ\gamma_n$ Lipschitz nên liên tục tuyệt đối với đạo hàm
$h_n$ h.k.n.); cộng telescoping cho $\Delta F$. ∎
**Hệ quả quan trọng:** vì $\sum_iA_i(Z)=\Delta F$ với MỌI đường (B2), completeness **mù với phương sai đường** của $Q$; nó chỉ đo phần cầu phương.
Luật $K^{-1/2}$ của completeness là kiểm tra cho cầu phương/cài đặt, không phải cho $V_1$ hay $J$.

**Mệnh đề 7.2 (MSE chính xác).** Dưới (S), (G), các sai số cặp $\varepsilon_k=\tfrac12(e(Z_k,U_k)+e(RZ_k,U'_k))$ iid, trung bình 0, phương sai
$\sigma^2=\tfrac12\operatorname{Var}e$ (hai thành viên không tương quan vì $\mathbb E[e\mid Z]=0$ và $U\perp U'$). Nếu thêm một độ lệch riêng ảnh $b$
(mọi nguồn ngoài mô hình: số học dấu phẩy động, $\Delta F$ tính khác $F$ trong tích phân, …): $e_K=b+\frac2K\sum_{k\le K/2}\varepsilon_k$ và
$$\mathbb Ee_K^2=b^2+\frac{2\sigma^2}{K}\quad\text{(mọi hình dạng phân phối của }\varepsilon\text{)}.$$
Thiết kế lồng nhau (các $K$ dùng chung tiền tố của cùng một dãy cặp) không đổi kỳ vọng này; chỉ làm các mức $K$ tương quan.
Do đó độ dốc của $\tfrac12\log\mathbb Ee_K^2$ theo $\log K$ bằng $-\tfrac12$ khi và chỉ khi $b=0$, và nằm trong $(-\tfrac12,0)$ khi $b\ne0$ —
**không bao giờ dốc hơn $-\tfrac12$ trong kỳ vọng**.

**Mệnh đề 7.3 (sai số tuyệt đối ở K hữu hạn).** Đặt $G_K=\mathbb E|e_K|/\sqrt{\mathbb Ee_K^2}$ (tỷ số Geary). Khi đó, đồng nhất:
$$\log\mathbb E|e_K|=\tfrac12\log\mathbb Ee_K^2+\log G_K,$$
nên độ dốc của $\log\mathbb E|e_K|$ = độ dốc RMS + độ dốc $\log G_K$. Khi $b=0$ và $\rho_3=\mathbb E|\varepsilon|^3<\infty$, cận Wasserstein của định
lý giới hạn trung tâm (Goldstein 2010, hằng số 1; $|x|$ là hàm 1-Lipschitz) cho
$$\Big|G_K-\sqrt{2/\pi}\Big|\le\frac{\rho_3}{\sigma^3\sqrt{K/2}}.$$
Do đó $\mathbb E|e_K|\propto K^{-1/2}$ chỉ **tiệm cận**; ở $K$ nhỏ độ lệch của $G_K$ khỏi $\sqrt{2/\pi}=0{,}798$ có thể cỡ $\rho_3/\sigma^3$ (luôn $\ge1$,
lớn khi phân phối sai số cặp có đuôi nặng hay dạng "hiếm khi một đoạn trội"). Chiều điển hình: $G_1=\mathbb E|\varepsilon|/\sigma$ nhỏ hơn 0,798 với
phân phối nhọn/đuôi nặng (t$_3$: 0,64) và lớn hơn với phân phối đuôi nhẹ (đều: 0,87; Rademacher: 1); $G_K\to0{,}798$ nên độ dốc $\log\mathbb E|e_K|$
**phẳng hơn** $-\tfrac12$ khi đuôi nặng, **dốc hơn** khi đuôi nhẹ — kể cả khi $b=0$ và mọi thứ được cài đặt đúng.

**Hệ quả 7.4 (không đồng nhất giữa ảnh).** Nếu $b_i=0$ và sai số cặp của ảnh $i$ là $\sigma_i\varepsilon$ với cùng luật chuẩn hoá, thì trung bình qua
ảnh $\overline{|e_K|}$ có kỳ vọng $\bar\sigma\cdot\mathbb E|\bar\varepsilon_K|$: độ dốc kỳ vọng không đổi, nhưng phương sai lấy mẫu của độ dốc tăng mạnh theo
độ phân tán của $\sigma_i$ (ảnh có $\sigma_i$ lớn chiếm ưu thế); log của trung bình mẫu còn có chệch Jensen, lớn nhất ở $K=2$ (một cặp) và khi
đuôi nặng — cũng theo hướng làm phẳng độ dốc RMS ước lượng.

**Kiểm số / mô phỏng** (`pathfam-final-audit` §2a, §2d; `tests/test_proof_signdiag.py`): mô hình trơn $d=16$, $N=32$: độ dốc $-0{,}496$ / $-0{,}492$
(4000 lần lặp); MLP ReLU: độ nhọn sai số 3,14, độ dốc TB|·| $-0{,}497$, RMS $-0{,}500$. Ở đúng thiết kế dữ liệu thật (128 ảnh, 16 cặp lồng nhau,
SD $\log\sigma_i=1$): sai số chuẩn $-0{,}498$ (SD 0,055); đều $-0{,}527$; t$_3$ $-0{,}443$; "đoạn trội" $-0{,}304$; chuẩn có $b_i$ với
$\operatorname{SD}(b_i)/\sigma_i=0{,}2$: $-0{,}417$. Phân tích dữ liệu CNN: `docs/COMPLETENESS_SLOPE.md`.

---

## 8. Tổng hợp: giả thiết → kết luận

| Kết luận | Giả thiết | Loại |
|---|---|---|
| $\sum_iA_i(Z)=\Delta F$ theo từng đường | (P), (F1)/(F1′) | Bổ đề B2 |
| μ chỉ phụ thuộc moment Stieltjes $|a|\le p-1$ (cần và đủ) | (P), (Poly$_p$) | Định lý 1 |
| bậc 2: $A-\mathrm{IG}$ = tổ hợp diện tích Lévy, từng đường | (P), (Poly$_2$) | Định lý 2 |
| bậc 2: $\mu=\mathrm{IG}$ | (P), (S) — **không cần (I), (M)** | Hệ quả 2.2 |
| bậc 2: cặp `rev` = 2·IG từng đường | (P), mọi Q | Định lý 2.3 |
| μ qua moment biên | (P), (I) | Định lý 3 |
| bậc 3: μ qua D | + (M), (H) | Hệ quả 3.1 |
| bậc 4: μ qua D | + (S) | Hệ quả 3.2 |
| bậc 5: D không đủ (đuôi, $\int\mathbb E(Z-t)^4$) | phản ví dụ giải tích | Mệnh đề 3.3 |
| luật kích hoạt rời nhau → Shapley (khối trên lưới) | mọi F khả vi | Mệnh đề 3.4 |
| Cov A bậc 2 qua hàm $c(s,t)$ | (I), (M), (H), (Poly$_2$) | Định lý 4 |
| `rand` không chệch cho đường nội suy | (G) | Bổ đề B5 |
| $J=L(\mu)+V/K$ chính xác; rev: $2V_e/K$ | (G) | Định lý 6.2 |
| thứ hạng theo B đổi tối đa một lần | (G), B liên tục | Mệnh đề 6.4 |
| $L=(r-\cos)^2+\sin^2$; phần $>1$ là thang đo | — | Mệnh đề 6.5 |
| completeness chỉ đo cầu phương; MSE $=b^2+2\sigma^2/K$ | (G), số học chính xác cho $b=0$ | Mệnh đề 7.1–7.2 |
| TB\|err\| $\propto K^{-1/2}$ chỉ tiệm cận; lệch ở K nhỏ cỡ $\rho_3/\sigma^3$ | (G), $b=0$ | Mệnh đề 7.3 |

## 9. Giới hạn

1. **Mạng ReLU không phải đa thức.** Đ1–Đ4 mô tả mô hình đa thức (hoặc xấp xỉ Taylor cục bộ/mô hình trơn). Với $F$ tuyến tính từng mảnh, μ
   phụ thuộc $Q$ qua phân phối thời điểm đường cắt các mặt kink; chưa có định lý. Khoảng cách giữa hai trường hợp là câu hỏi thực nghiệm.
   B2, B5, M3.4, §6, §7 vẫn đúng dưới (F1′).
2. **Bậc cao và họ phụ thuộc.** Phương sai bậc $\ge3$ chỉ có bằng chứng số (DP và cmix cùng μ bậc 3 nhưng phương sai khác, $|z|=71$). Đ4 cần (I).
3. **Lưới hữu hạn.** Mọi họ được thực hiện trên lưới $N$; $c$ của đường nội suy khác $c$ liên tục $O(1/N)$. Các định lý áp dụng cho luật nội suy $Q^N$.
4. **Phương sai cầu phương** $\mathbb E\operatorname{Cov}_U$ phụ thuộc hình học đường và mô hình; chưa có cận theo thống kê của $Q$.
5. **Không có kết quả tối ưu.** Lý thuyết nói thống kê nào của $Q$ quyết định $\mu$ và phương sai (moment Stieltjes, diện tích Lévy, D, $c$, cumulant,
   luật kích hoạt), không nói họ nào tốt nhất; không có gì đặc biệt về Dirichlet.
6. **J là một mục tiêu cụ thể.** M6.5 cho thấy $J$ thô trộn thang đo và hướng; các định lý về μ/phương sai không nói gì về metric xếp hạng không dấu.

Tham khảo: L. Goldstein (2010), *Bounds on the constant in the mean central limit theorem*, Ann. Probab. 38(4), 1672–1689.
