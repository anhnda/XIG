# Tổng kết thảo luận SPI: Path Attribution, Permutation, Credit Allocation và Insertion–Deletion

*Tổng hợp từ `SPI_Discuss.md`, ưu tiên các thảo luận cuối và những kết quả toy model đã kiểm tra.*

## 1. Kết luận trung tâm

Cuộc thảo luận cuối cùng hội tụ về một câu hỏi quan trọng hơn việc chứng minh SPI tốt hơn IG:

**Với một mô hình $F$ cố định, baseline $b$ và input $x$, có nhiều cách phân bổ cùng một lượng output $F(x)-F(b)$. Vậy cách phân bổ nào tốt hơn, dựa trên tiêu chí gì, và tiêu chí đánh giá có thực sự phù hợp với cách phân bổ đó không?**

Có sáu kết luận chính:

1. **Path quyết định credit allocation.** Thay đổi đường đi, từ IG đồng bộ đến đường gấp khúc, đường cạnh hay đường ngẫu nhiên SPI, làm thay đổi bối cảnh mà mỗi feature nhận marginal contribution. Mọi đường hợp lệ có thể cùng thỏa completeness nhưng cho attribution khác nhau rất lớn.
2. **Insertion–Deletion chủ yếu đánh giá ranking, không đánh giá trực tiếp lượng credit.** Các vector attribution khác nhau có thể đạt cùng InsDel nếu tạo ra cùng thứ tự feature. InsDel còn phụ thuộc vào reference, cách masking và quy tắc xếp hạng signed hay absolute.
3. **Có sự tương hợp giữa đường cạnh/Shapley và hard-mask InsDel**, vì cả hai đều sử dụng các trạng thái feature được bật/tắt hoàn toàn. Tuy nhiên, đường càng sát cạnh không đồng nghĩa điểm càng cao; chính ranking cuối cùng mới quyết định.
4. **Thứ tự kích hoạt trên path có thể điều khiển attribution.** Với interaction dương, kích hoạt feature muộn thường làm tăng credit nó nhận từ interaction; với interaction âm, kết quả có thể ngược lại. Thứ tự tối ưu để *tính attribution* có thể trái với thứ tự tối ưu để *đánh giá InsDel*.
5. **Expectation và averaging không đảm bảo cải thiện InsDel.** Do ranking là phép toán phi tuyến, $S(\mathbb E[A])\neq\mathbb E[S(A)]$. Trên toy, trung bình nhiều đường có lúc cải thiện rõ, nhưng có lúc làm score tệ hơn cả IG.
6. **Guided IG có một điểm chung đáng chú ý với chiến lược kích hoạt feature nhỏ trước, feature lớn sau**, nhưng chỉ tương đồng về cơ chế sắp xếp lịch thay đổi tọa độ. Guided IG thực tế chọn theo độ lớn gradient tức thời, không chủ động tối ưu phân bổ interaction hay InsDel.

---

## 2. Hình học đường tích phân: từ IG đến các đường cạnh

Với một đường $\gamma$ nối baseline đến input, attribution của feature $i$ là:

$$
A_i^\gamma=\int_\gamma \partial_iF(z)\,dz_i.
$$

Thay đổi đường đi không chỉ là một kỹ thuật giảm nhiễu. Nó thay đổi **trạng thái của các feature khác tại thời điểm feature $i$ được cập nhật**, từ đó thay đổi credit được gán cho $i$.

| Kiểu đường | Cách cập nhật feature | Ý nghĩa phân bổ |
|---|---|---|
| **Straight-line IG** | Tất cả tiến đồng bộ | Đánh giá marginal gradient khi các feature cùng tăng |
| **Đường gấp khúc nội miền** | Feature tăng không đều | Đánh giá gradient ở những tổ hợp trạng thái trung gian khác IG |
| **Đường cạnh** | Mỗi feature được kích hoạt trọn vẹn theo một thứ tự | Credit là marginal contribution có điều kiện theo thứ tự đó |
| **Shapley** | Trung bình đều các hoán vị đường cạnh | Credit là trung bình trên mọi thứ tự bật feature |
| **SPI** | Các lịch kích hoạt ngẫu nhiên, có thể xen kẽ từng phần | Lấy trung bình credit trên một phân phối rộng hơn của các trạng thái feature |

Một ví dụ đơn giản là $F(u,v)=uv$, baseline $(0,0)$, input $(1,1)$.

- Nếu $u$ tăng trước $v$, credit là $(0,1)$.
- Nếu $v$ tăng trước $u$, credit là $(1,0)$.
- Nếu cả hai tăng đồng bộ theo IG, credit là $(0.5,0.5)$.

**Không có thuật toán nào phải tách interaction ra một cách tường minh.** Sự phân bổ xuất hiện tự nhiên từ đạo hàm từng feature dọc các đường khác nhau.

## 3. Toy Model 1: Cùng tổng output, credit có thể khác rất lớn

Toy đầu tiên:

$$
F_\lambda(u,v)=u+\lambda (u-v)u(1-u)v(1-v).
$$

Với $b=(0,0)$, $x=(1,1)$, $\lambda=-60$:

$$
F(x)-F(b)=1.
$$

| Path | Attribution $(A_u,A_v)$ |
|---|---|
| IG đường chéo | $(-1,2)$ |
| Gấp khúc qua $(0.2,0.8)$ | $(0.7568,0.2432)$ |
| Gấp khúc qua $(0.25,0.75)$ | $(0.34375,0.65625)$ |
| Đường cạnh | $(1,0)$ |

Tất cả có tổng bằng 1. Nhưng IG thậm chí cho $v$ credit dương lớn và $u$ credit âm, trong khi trên mọi đỉnh bật/tắt, $u$ đóng góp 1 và $v$ đóng góp 0.

Điểm đặc biệt là trên đường chéo:

$$F_\lambda(t,t)=t.$$

Output hoàn toàn tuyến tính, nhưng:

$$
A^{IG}=\left(1+\frac{\lambda}{30},-\frac{\lambda}{30}\right).
$$

Lý do là gradient có thể chứa thành phần pháp tuyến đối với đường chéo, dù path không rời khỏi đường chéo. Khi IG tách đạo hàm thành credit từng tọa độ, các phần này có thể tạo ra hai credit trái dấu rồi triệt tiêu trong tổng.

**Bài học:** Output mượt dọc path, completeness hay việc path nằm trên manifold đều không tự xác nhận cách phân bổ từng feature là tốt.

### Giới hạn của InsDel ngay trên toy 2 biến

Với InsDel có dấu $S=\mathrm{InsertionAUC}-\mathrm{DeletionAUC}$, AUC hình thang và hai feature:

$$
S_{u>v}=\frac{F(1,0)-F(0,1)}2,
\qquad
S_{v>u}=-S_{u>v}.
$$

Do đó, hai thứ tự luôn có cùng $|S|$. Lấy trị tuyệt đối của score sẽ khiến phép đánh giá không phân biệt được thứ tự đúng và thứ tự đảo ngược trên thiết lập này.

Điều này là lý do thảo luận chuyển sang toy ba biến.

## 4. Toy Model 2: Mối quan hệ giữa path, ranking và InsDel

Mô hình chính:

$$
F(u,v,w)=u+0.2v+0.5w+0.6uvw-60H(u,v)
$$

với

$$
H(u,v)=(u-v)u(1-u)v(1-v).
$$

Baseline $b=(0,0,0)$, input $x=(1,1,1)$, tổng output chênh lệch $2.3$.

Thành phần $H$ bằng 0 trên tất cả các cạnh và đỉnh nhị phân, nhưng vẫn ảnh hưởng rất lớn tới attribution trên những đường đi trong nội miền.

### So sánh các cách phân bổ

| Phương pháp | Attribution $(u,v,w)$ | Ranking theo $|A|$ | InsDel |
|---|---|---|---:|
| IG | $(-0.8,2.4,0.7)$ | $v>u>w$ | −0.200 |
| Gấp khúc qua $(0.2,0.8,0.5)$ | $(1.1008,0.5172,0.682)$ | $u>w>v$ | +0.533 |
| Cạnh $w\to v\to u$ | $(1.6,0.2,0.5)$ | $u>w>v$ | +0.533 |
| Shapley | $(1.2,0.4,0.7)$ | $u>w>v$ | +0.533 |

Các điểm InsDel trong bảng sử dụng reference 0, output thô, AUC hình thang và ranking theo $|A|$.

**Điểm quan trọng nhất:** Ba phương pháp sau có cách phân bổ credit rất khác nhau nhưng cùng InsDel, chỉ vì ranking cuối cùng giống nhau.

Do đó, InsDel không phân biệt được một feature nhận $0.5172$, $0.2$ hay $0.4$ credit nếu điều đó không làm đổi ranking.

### Tại sao InsDel có sự tương hợp với đường cạnh/Shapley?

Hard-mask InsDel đánh giá mô hình trên các trạng thái feature được thay hoàn toàn bằng input hoặc reference.

Đường cạnh và Shapley cũng dựa trên marginal contributions giữa các trạng thái bật/tắt hoàn toàn. Đây là một sự tương hợp về **tập các trạng thái mô hình được truy vấn**.

Nhưng tương hợp không đồng nghĩa đường cạnh luôn có InsDel cao hơn. Hai đường cạnh khác thứ tự có thể cho các ranking khác nhau, và do đó score khác nhau.

Ngoài ra, thành phần $H$ là ví dụ điển hình của một thành phần mô hình có thể thay đổi attribution mạnh nhưng vô hình đối với hard-mask InsDel trên các đỉnh.

## 5. Thứ tự permutation trên đường cạnh và hiệu năng

Khi tính đủ sáu permutation trong toy dương, interaction $0.6uvw$ được giao toàn bộ cho feature kích hoạt cuối cùng.

| Feature cuối trên path | Attribution | Ranking theo $|A|$ | InsDel |
|---|---|---|---:|
| $w$ | $(1,0.2,1.1)$ | $w>u>v$ | +0.200 |
| $v$ | $(1,0.8,0.5)$ | $u>v>w$ | +0.333 |
| $u$ | $(1.6,0.2,0.5)$ | $u>w>v$ | **+0.533** |

Hai permutation khác nhau có cùng feature cuối cho cùng attribution trong toy này.

Đáng chú ý, **thứ tự kích hoạt trên path không phải thứ tự ranking khi InsDel đánh giá**. Ví dụ, kích hoạt $u$ cuối có thể khiến $u$ nhận credit lớn nhất và được đưa vào đầu tiên trong Insertion.

### Quy luật interaction quyết định hướng dịch chuyển credit

Cho hai feature $i,j$, với bối cảnh các feature khác là $S$, interaction rời rạc:

$$
I_{ij}(S)=F(S\cup\{i,j\})-F(S\cup\{i\})-F(S\cup\{j\})+F(S).
$$

Khi đổi từ kích hoạt $i$ trước $j$ sang sau $j$, ta có:

$$
\boxed{A_i^{\mathrm{after}}-A_i^{\mathrm{before}}=I_{ij}(S)}.
$$

Đây là đẳng thức chính xác cho hai thứ tự kề nhau, giữ nguyên bối cảnh còn lại.

| Dấu interaction | Kích hoạt feature $i$ muộn hơn |
|---|---|
| Dương | Làm credit của $i$ tăng |
| Âm | Làm credit của $i$ giảm |
| Bằng 0 | Không làm đổi credit |

Với interaction dương và *increasing marginal returns* (supermodularity), feature vốn có đóng góp lớn có thể được giữ đến cuối để nhận thêm interaction, nhờ đó đạt ranking phù hợp InsDel hơn.

Nhưng đây là quy luật về **credit**, không phải một định lý trực tiếp về **score**. Score chỉ tăng nếu sự thay đổi credit cải thiện ranking theo giao thức đã chọn.

## 6. SPI, expectation và averaging: phát hiện quan trọng nhất

### 6.1. Random edge permutations và Shapley

Nếu lấy đều sáu thứ tự đường cạnh, mỗi feature có xác suất $1/3$ được kích hoạt cuối và nhận interaction $0.6uvw$.

Attribution kỳ vọng là:

$$
\mathbb E_\gamma[A^\gamma]=(1.2,0.4,0.7),
$$

chính là Shapley attribution của trò chơi bật/tắt.

Tuy nhiên, có hai cách lấy trung bình khác nhau:

| Đại lượng | InsDel |
|---|---:|
| Trung bình score từng path, $\mathbb E[S(A^\gamma)]$ | $0.356$ |
| Score của attribution trung bình, $S(\mathbb E[A^\gamma])$ | $0.533$ |

Lý do khác nhau là InsDel chứa phép sắp xếp feature, vốn phi tuyến và không giao hoán với kỳ vọng.

**Averaging không chỉ giảm phương sai của ước lượng. Nó có thể thay đổi ranking và do đó thay đổi chính score.**

### 6.2. SPI không đơn thuần random một permutation

Với SPI có lịch DP độc lập:

$$
\gamma(t)=(Z_u(t),Z_v(t),Z_w(t)).
$$

Ở $\alpha$ hữu hạn, feature có thể tăng xen kẽ từng phần; không nhất thiết có một permutation hoàn chỉnh.

Trong toy ba biến, attribution kỳ vọng được suy ra:

$$
\boxed{
\mathbb E[A_\alpha^{SPI}]
=(1.2-60C_\alpha,\ 0.4+60C_\alpha,\ 0.7)
}
$$

với

$$
C_\alpha=\frac{\alpha^3}{30(\alpha+1)^2(\alpha+2)}.
$$

Hai giới hạn:

$$
\alpha\to0:\quad (1.2,0.4,0.7),
\qquad
\alpha\to\infty:\quad (-0.8,2.4,0.7).
$$

Tức là trong toy này, thay $\alpha$ tạo ra các phân bổ liên tục nối Shapley với IG. Tuy nhiên, **InsDel không thay đổi liên tục theo attribution**, bởi score chỉ thay đổi khi ranking đổi.

### 6.3. Một $\alpha$ trung gian có thể tệ hơn cả IG

| $\alpha$ | InsDel của attribution kỳ vọng SPI (ranking theo $|A|$) |
|---|---:|
| $\alpha\to0$ | +0.533 |
| 1 | +0.533 |
| 2 | −0.200 |
| 3 | **−0.533** |
| 10 | **−0.533** |
| 100 | −0.200 |
| $\alpha\to\infty$ (IG) | −0.200 |

Đây là phản ví dụ trực tiếp cho ý tưởng “SPI random mạnh hơn thì tốt hơn” hoặc “SPI tệ nhất cũng cải thiện InsDel”.

Thậm chí, thảo luận còn thử tăng số path trung bình $K$:

| $K$ | $\alpha=1$ | $\alpha=3$ |
|---:|---:|---:|
| 1 | +0.230 | −0.179 |
| 4 | +0.329 | −0.303 |
| 16 | +0.476 | −0.453 |
| 64 | +0.528 | −0.527 |
| $\infty$ | +0.533 | −0.533 |

Các số hữu hạn $K$ là Monte Carlo trong thảo luận.

Với $\alpha=1$, averaging cải thiện InsDel. Nhưng với $\alpha=3$, averaging làm điểm tệ dần, hội tụ về −0.533.

**Kết luận:** Tăng số path làm ước lượng hội tụ về một allocation operator cụ thể, nhưng operator đó không được bảo đảm phù hợp với InsDel.

## 7. Nên ưu tiên feature nào trước, feature nào cuối?

Phần cuối thảo luận thử một phản ví dụ với interaction âm:

$$
F(u,v,w)=u+0.2v+0.5w-1.3uvw.
$$

Feature $u$ có đóng góp trực tiếp lớn nhất là 1.

Nếu kích hoạt $u$ trước:

$$
u\to v\to w,\quad A=(1,0.2,-0.8).
$$

Nếu giữ $u$ đến cuối:

$$
v\to w\to u,\quad A=(-0.3,0.2,0.5).
$$

Credit của $u$ đổi từ $+1$ thành $-0.3$, bởi $u$ khi kích hoạt cuối phải nhận toàn bộ interaction âm $-1.3$.

Với ranking theo $|A|$, InsDel tương ứng là $+0.533$ và $+0.200$.

Vậy chiến lược **“giữ feature mạnh nhất đến cuối” không đúng phổ quát**. Nó có thể hữu ích với interaction dương, nhưng phản tác dụng với interaction âm.

Cách phát biểu hợp lý hơn là:

**Thứ tự kích hoạt cần được lựa chọn dựa trên marginal contributions có điều kiện và tiêu chí phân bổ mong muốn, thay vì chỉ dựa trên độ lớn đóng góp riêng lẻ.**

Và nếu mục tiêu là faithfulness, việc trực tiếp chọn permutation để tối đa hóa InsDel có nguy cơ tạo ra một phương pháp được thiết kế để khớp metric, chứ chưa chắc giải thích credit đúng hơn.

## 8. Liên hệ với Guided IG

Đây là một hướng đáng chú ý ở đoạn cuối.

Guided IG ưu tiên cập nhật những tọa độ có $|\partial_iF|$ nhỏ tại trạng thái hiện tại. Nhìn từ toy có interaction dương, nó giống chiến lược cho feature nhỏ tiến trước, giữ feature lớn lại để nhận credit khi các feature khác đã được kích hoạt.

Trong phần cộng tính của toy, gradient tại baseline là:

$$
\nabla F(0,0,0)=(1,0.2,0.5).
$$

Nếu đi theo thứ tự:

$$v\to w\to u,$$

thì nhận:

$$A=(1.6,0.2,0.5),\quad S=+0.533.$$

Đây là mức InsDel cao nhất của toy dưới reference 0.

Tuy nhiên, có hai điểm phải phân biệt:

1. **Guided IG gốc lựa chọn dựa trên gradient tức thời dọc đường**, không phải marginal contribution cuối cùng. Khi có $H(u,v)$, gradient nội miền có thể thay đổi mạnh, nên đường thực sự của Guided IG không nhất thiết là $v\to w\to u$. Con số +0.533 ở đây là của đường cạnh giả định, không phải phép chạy Guided IG đầy đủ.
2. **Guided IG không đặt mục tiêu tối đa hóa InsDel hay phân bổ tương tác dương cho feature lớn.** Việc hai cơ chế có thể cho thứ tự tương tự chỉ gợi ý một giả thuyết cần kiểm tra.

Một ý tưởng tổng quát xuất hiện từ đây là **guided stochastic paths**: sinh các lịch ngẫu nhiên nhưng có thiên lệch thứ tự theo thông tin gradient hoặc marginal contribution, thay vì để các feature có lịch độc lập và đối xứng như SPI chuẩn.

Đây là hướng thiết kế có tiềm năng, nhưng chưa có kết quả trong thảo luận chứng minh nó cải thiện credit allocation tổng quát.

## 9. Quy tắc ranking trong evaluation: signed hay absolute?

Đây là một sửa đổi rất quan trọng ở những đoạn cuối, vì nhiều bảng toy trước đó mặc định dùng $|A_i|$ để ranking.

Ba phép toán cần phân biệt hoàn toàn:

| Quy tắc | Ý nghĩa |
|---|---|
| Sort theo $A_i$ giảm dần | Ưu tiên feature có credit dương lớn, hỗ trợ output |
| Sort theo $|A_i|$ giảm dần | Ưu tiên feature có tác động lớn, kể cả tác động âm |
| Lấy $|\mathrm{InsDel}|$ | Đổi dấu score âm thành dương, có thể che mất ranking ngược |

Trên toy interaction âm, chỉ thay quy tắc ranking có thể làm score đổi từ $+0.533$ thành $-0.200$, hoặc ngược lại.

Theo phần kiểm tra mã nguồn được ghi trong thảo luận, implementation InsDel chuẩn của RISE sử dụng `argsort` trên attribution được đưa vào, **không tự áp dụng `abs()`**. Nhưng các mô tả công bố của SPI, DiffIG và Guided IG không đủ để kết luận chắc chắn toàn bộ pipeline của từng paper đều ranking theo signed attribution nguyên bản.

Vì thế, kết luận thực nghiệm đã thống nhất là nên báo cả signed ranking và absolute ranking, với ý nghĩa khác nhau, và không nên dùng $|S|$ làm score chính nếu muốn phân biệt thứ tự hỗ trợ output với thứ tự gây tác động ngược.

## 10. Reference và evaluation–path alignment

Cùng một attribution cố định có thể đổi kết quả InsDel khi thay reference đánh giá.

Trên toy ba biến, với reference đen $(0,0,0)$, IG có InsDel −0.200 còn đường gấp khúc/cạnh/Shapley có +0.533.

Nhưng với reference không đồng nhất $(0.5,0.2,0.8)$, IG đạt +0.460 còn nhóm kia đạt +0.119. Phương pháp thắng bị đảo.

Đây là reference được chọn để xây dựng phản ví dụ, không đại diện trực tiếp cho blur hay mean trong ảnh thực.

Thảo luận vì vậy đề xuất phân biệt hai chiều của giao thức đánh giá:

| Chiều thay đổi | Ví dụ |
|---|---|
| **Replacement value** | Black, blur, mean, inpainting |
| **Replacement dynamics** | Thay toàn bộ feature một lần, thay dần liên tục, thay phối hợp nhiều feature |

Chỉ thử nhiều replacement value chưa đủ kiểm tra đầy đủ evaluation–path alignment. Cần thử cả các kiểu replacement dynamics khác nhau.

Khi nghiên cứu độ nhạy của metric, cần giữ attribution cố định và chỉ đổi evaluation reference. Nếu thay cả baseline dùng để tính attribution, đó là một thí nghiệm khác và phải báo riêng.

## 11. Những luận điểm quan trọng khác từ giai đoạn E1–E12

Các kết quả trước đoạn toy cũng hỗ trợ việc mở rộng câu hỏi khỏi riêng SPI/DP.

**DP không phải điều kiện cần để đạt hiệu năng tương đương SPI.** E10 thử smooth random warp và logistic-normal increments ở mức exploration được khớp với DP $\alpha=10$, và ghi nhận chất lượng tương đương trong biên kiểm định đã đăng ký. Điều này cho thấy cơ chế tổng quát hơn là phân phối những trạng thái trung gian được path khảo sát, không phải bản thân Dirichlet Process.

**Không phải loại randomness nào cũng hữu ích.** Nếu mọi feature cùng dùng một lịch ngẫu nhiên, đường chỉ là tái tham số hóa IG và attribution canonical không thay đổi. COORD_MIX, LINE mixture, lịch shared theo RGB hoặc spatial block cũng cho thấy mức variance biên bằng nhau không đảm bảo chất lượng bằng nhau.

**Exploration quá mạnh cũng có thể gây hại.** E12 trên ảnh ghi nhận $\alpha=1$ kém $\alpha=10$ ở các metric đã đăng ký, nhưng chỉ số phi tuyến $R_{\mathrm{nl}}$ không dự đoán được đáng tin cậy sự suy giảm. Chưa có quy tắc chọn $\alpha$ thích nghi được xác thực.

**Population operator và sai số tính toán cần được tách riêng.** Thay $\alpha$ làm thay đổi attribution kỳ vọng, không chỉ variance của Monte Carlo. Nhưng số path, số nút tích phân và sai số cầu phương cũng ảnh hưởng đến attribution thực nghiệm, nên không thể quy mọi chênh lệch score cho hình học đường đi.

**Manifold và tangent projection không tự giải quyết bài toán phân bổ.** Toy 1 cho thấy mọi attribution complete đều có thể khớp output trên manifold $u=v$, trong khi phép chiếu gradient lên tiếp tuyến lại có thể làm mất tính Dummy của feature attribution. Vì thế manifold chỉ là một góc nhìn bổ trợ; nó không tự cung cấp quy tắc chia credit duy nhất.

## 12. Kết luận nghiên cứu và hướng tiếp tục

Điểm quan trọng nhất hiện nay không còn là “SPI có tốt hơn IG không?”, mà là:

$$
\boxed{
\text{Path distribution}
\longrightarrow \text{Credit allocation}
\longrightarrow \text{Ranking}
\longrightarrow \text{Evaluation score}
}.
$$

Bốn tầng này cần được phân biệt. Đặc biệt, một phương pháp có InsDel cao có thể chỉ vì cách phân bổ credit của nó tạo ra ranking phù hợp hơn với chính giao thức intervention được dùng để đánh giá.

Nếu tiếp tục nghiên cứu, các thí nghiệm có giá trị nhất là:

1. **Chốt lại toàn bộ bảng toy theo signed ranking**, đồng thời báo $|A|$ như một đối chứng. Các kết luận dùng absolute ranking không nên được chuyển nguyên xi sang benchmark RISE.
2. **Tách permutation selection khỏi evaluation selection.** So sánh random uniform, guided ordering, reverse ordering và các hoán vị có điều kiện theo dấu interaction; đánh giá trên giao thức chưa dùng để chọn đường.
3. **Tách expectation và single-path effects.** Báo $S(\mathbb E[A])$, $\mathbb E[S(A)]$, độ ổn định ranking theo $K$, và kiểm tra cả các trường hợp averaging cải thiện lẫn làm hại.
4. **Thiết kế multi-reference, multi-dynamics evaluation.** Không chỉ thay black bằng blur mà còn thay cách tiến triển feature, nhằm kiểm tra giả thuyết evaluation–path alignment.
5. **Đối chiếu Guided IG với các đường cạnh giả định và guided stochastic paths thực sự.** Cần xác định phần lợi ích đến từ thứ tự gradient nhỏ trước, từ tương tác có dấu, hay đơn thuần từ ranking phù hợp với InsDel.

### Kết luận cuối cùng

**Hiện chưa có một quy tắc hoán vị, một mức $\alpha$, hay một phân phối đường duy nhất được chứng minh là cho credit allocation tốt nhất.** Nhưng thảo luận đã xác định được một cơ chế khá rõ: đường đi điều khiển bối cảnh marginal contribution của từng feature; kỳ vọng quyết định cách trung bình credit qua các bối cảnh; ranking chuyển sự phân bổ ấy thành thứ tự; và InsDel chỉ thưởng cho mức phù hợp giữa thứ tự đó với các trạng thái can thiệp mà nó đánh giá.

Đây là nền tảng hợp lý hơn để tìm nguyên lý chung giữa IG, SPI, Guided IG, Shapley và các path attribution khác, thay vì chỉ so phương pháp nào tạo heatmap sạch hơn hoặc thắng một metric cụ thể.
