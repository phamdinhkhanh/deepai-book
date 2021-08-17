---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.8.2
kernelspec:
  display_name: Python 3
  name: python3
---

# 7.1. Bài toán dạng tổng quát

+++ {"id": "BfwFydKH4Fx9"}

Bài toán tối ưu dạng tổng quát:

$$ \begin{eqnarray}
\mathbf{x}^* &=& \arg\min_{\mathbf{x}} f_0(\mathbf{x}) \\ 
\text{subject to:}~ && f_i(\mathbf{x}) \leq 0, ~~ i = 1, 2, \dots, m\\ 
&& h_j(\mathbf{x}) = 0, ~~ j = 1, 2, \dots, p  \tag{1}
\end{eqnarray}
$$

Phát biểu bằng lời: Tìm giá trị của $\mathbf{x}$ để tối thiểu hàm mục tiêu $f_0(\mathbf{x})$ thoả mãn hệ điều kiện ràng buộc đẳng thức và bất đẳng thức.

Các định nghĩa:

* $\mathbf{x} \in \mathbb{R}^{n}$ là _biến tối ưu_ (_optimization variable_)
* $f_0(\mathbf{x})$ là _hàm mục tiêu_ (_objective/cost/lost function_)
* $f_i(\mathbf{x}) : \mathbb{R}^{n} \mapsto \mathbb{R}$ _ràng buộc dạng bất đẳng thức_ (_inequality constraint function_). Trong đó $i = 1, 2, \dots, m$.
* $h_j(\mathbf{x})=0: \mathbb{R}^n \mapsto \mathbb{R}$ _ràng buộc dạng đẳng thức_ (_equality constraint function_). Trong đó $j = 1, 2, \dots, p$.
* Tập hợp tất cả các điểm mà các hàm mục tiêu và $\mathcal{D} = \bigcap_{i=0}^m \text{dom}f_i \cap \bigcap_{pj=1}^p \text{dom}h_i$: _tập xác định_ (_domain_). 

Một điểm $\mathbf{x} \in \mathcal{D}$ là _feasible_ nếu như nó thoả mãn các ràng buộc $f_i(\mathbf{x}) \geq 0$, $i = 1, \dots, m$ và $h_{i}(\mathbf{x}) = 0$, $j = 1, 2, \dots, p$. Bài toán tối ưu $(1)$ được gọi là _feasible_ nếu như tồn tại ít nhất một điểm _feasible_ point. Trong trường hợp không tồn tại bất kì một điểm _feasible_ nào thì ta nói rằng bài toán là _infeasible_. Tập hợp tất cả các điểm _feasible points_ được gọi là _feasible set_ hoặc _constraint set_.

_Nghiệm tối ưu_ (_optimal value_) $p^*$ của bài toán còn được định nghĩa theo _infimum_ là:

$$p^* = \inf \{ f_0(\mathbf{x}) | f_{i}(\mathbf{x}) \leq 0, ~ i = 1, \dots, m, h_i(\mathbf{x}) = 0, i = 1, \dots, p \}$$


+++ {"id": "XhRpzNlaU0rr"}

# 7.2. Các bài toán tối ưu thực tiễn.

+++ {"id": "Tsa2yP93VX9p"}

## 7.2.1. Bài toán vận chuyển

Tối ưu kho vận là bài toán có tính ứng dụng rất cao trong lớp các bài toán tối ưu. Vấn đề mà bài toán giải quyết liên là vấn đề mà nhiều doanh nghiệp đang gặp phải. Bài toán này sẽ giúp cho các doanh nghiệp tiết kiệm được chi phí vận chuyển hàng hoá nhờ phân bổ nguồn lực ở các kho một cách hợp lý đến các cửa hàng. Không chỉ giúp tối ưu về chi phí, bài toán vận chuyển còn có thể chuyển sang mục tiêu khác đó là tối ưu thời gian vận chuyển. Hãy cùng tìm hiểu bài toán này qua ví dụ bên dưới.

+++ {"id": "vOVhcd8a6ucC"}

### Bài toán

Một công ty sản xuất sữa cần giao sản phẩm tới các các cửa hàng tại Hà Nội (1800 thùng) và TP. HCM (2000 thùng). Công ty có sẵn trong kho tại Thái Bình (1000 thùng) và Khánh Hoà (3000) thùng. Chi phí vận chuyển từ kho tới của hàng được cho như bảng bên dưới:

![](https://imgur.com/6qTFdyp.png)

Hỏi công ty cần phân bổ hàng tại kho như thế nào để chi phí giao hàng là thấp nhất.

**Lời giải:**

![](https://imgur.com/KqFUQFk.png)

Giả định số lượng giao hàng là các biến được thể hiện trong ma trận:

$$\begin{bmatrix}
x_{11} & x_{12} \\
x_{21} & x_{22}
\end{bmatrix}$$

Bài toán tối ưu với hàm mục tiêu là chi phí giao hàng tối thiểu:

$$\mathbf{x}^{*} = \arg \min_{\mathbf{x}} f(\mathbf{x}) = \arg \min_{\mathbf{x}}50 x_{11} + 120 x_{12} + 150 x_{21} + 40 x_{22}$$

* Ràng buộc về số lượng:

Tại Hà Nội: $x_{11}+x_{21} = 1800$

Tại TP.HCM: $x_{12} + x_{22} = 2000$

* Ràng buộc số lượng hàng tại kho:

Tại Thái Bình: $x_{11}+x_{12} \leq 1000$

Tại Khánh Hoà: $x_{21} + x_{22} \leq 3000$

* Ràng buộc về số lượng dương: $x_{11}, x_{12}, x_{21}, x_{22} \geq 0$. Trên thực tế chúng ta cần các giá trị $x_{ij}$ phải là số nguyên không âm nhưng sẽ khiến cho bài toán khó giải. Để đơn giản hoá chúng ta chuyển sang $x_{ij} \geq 0$.

**Phát biểu bài toán:**

Bài toán trên có thể viết như sau:

$$ \begin{eqnarray}
\mathbf{x}^* &=& \arg\min_{\mathbf{x}} f(\mathbf{x}) = \arg \min_{\mathbf{x}}50 x_{11} + 120 x_{12} + 150 x_{21} + 40 x_{22}\\ 
\text{subject to:}~ && x_{11} + x_{12} \leq 1000\\ 
&& x_{21} + x_{22} \leq 3000\\ 
&& x_{11} + x_{21} = 1800\\ 
&& x_{12} + x_{22} = 2000\\
&& x_{11}, x_{12}, x_{21}, x_{22} \geq 0
\end{eqnarray}
$$

**Lưu ý:** Bài toán có cả hàm mục tiêu và điều kiện ràng buộc là những hàm tuyến tính thì được gọi là bài toán _qui hoạch tuyến tính_ (_Linear Program - LP_). Phát biểu và lời giải của bài toán này sẽ được trình bày ở phần tiếp theo.

+++ {"id": "Hus40xKwFmKr"}

### Bài toán qui hoạch tuyến tính (_Linear Programming LP_)

Dạng tổng quát của bài toán vận chuyển là bài toán qui hoạch tuyến tính có dạng như sau:

$$\begin{eqnarray}
\mathbf{x}^* &=& \arg\min_{\mathbf{x}} \mathbf{c}^{\intercal}\mathbf{x} + d \\ 
\text{subject to:}~ && \mathbf{Gx} \preceq \mathbf{h} \\ 
&& \mathbf{Ax} = \mathbf{b}
\end{eqnarray}
$$

Ta nhận thấy hàm mục tiêu của bài toán qui hoạch tuyến tính là một hàm có dạng tuyến tính. Ở điều kiện ràng buộc thì $\mathbf{G} \in \mathbb{R}^{m \times n}, \mathbf{A} \in \mathbb{R}^{p \times n}, \mathbf{h} \in \mathbb{R}^{m}, \mathbf{b} \in \mathbb{R}^{p}$.  Dòng thứ $i$ của ma trận $\mathbf{G}$ là $\mathbf{g}_i$ được xem như hệ số của điều kiện ràng buộc dạng bất đẳng thức $g_i(\mathbf{x}) = \mathbf{g}_{i}\mathbf{x} \leq h_i$. $h_i$ là giá trị phần tử thứ $i$ của véc tơ $\mathbf{h}$. Tương tự dòng thứ $i$ của ma trận $\mathbf{A}$ cũng là hệ số của hệ điều kiện ràng buộc dạng đẳng thức. Một điều kiện đẳng thức cũng có thể được chuyển hoá sang điều kiện bất đẳng thức nếu như chúng ta thay chúng bằng hai điều kiện: $\mathbf{a}_i \mathbf{x} \leq b_i$ và $\mathbf{a}_i \mathbf{x} \geq b_i$. Do đó một số tài liệu người ta chỉ có điều kiện bất đẳng thức.

Một số bài toán thực tế sẽ xuất hiện thêm điều kiện $x_{ij} \geq 0$. Khi đó chúng ta có thể đổi dấu của $x_{ij}$ để chuyển về điều kiện $-x_{ij} \leq 0$.

Đối với hàm mục tiêu thì véc tơ $\mathbf{c}$ được xem như véc tơ trọng số của biến mục tiêu $\mathbf{x}$. $d$ là một đại lượng vô hướng và không ảnh hưởng tới nghiệm.

Lời giải của bài toán qui hoạch tuyến tính (_LP_) có thể được giải qua cxopt như sau:

```{note}
solver.lp(c, G, h, A, b)
```

Quay trở lại bài toán vận chuyển:

$$\mathbf{c} = [50, 150, 120, 40]^{\intercal}$$ 
$$\mathbf{x} = [x_{11}, x_{12},  x_{21}, x_{22}]^{\intercal}$$

$$\mathbf{G} = \begin{bmatrix}
1 & 1 & 0 & 0 \\
0 & 0 & 1 & 1\\
-1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & -1 \\
\end{bmatrix}
$$

$$\mathbf{A} = \begin{bmatrix}
1 & 0 & 1 & 0\\
0 & 1 & 0 & 1
\end{bmatrix}$$

$$\mathbf{h} = [1000, 3000, 0, 0, 0, 0]^{\intercal}$$

$$\mathbf{b} = [1800, 2000]^{\intercal}$$

```{note}
Lưu ý ở code bên dưới khi sử dụng ma trận được khởi tạo từ hàm cvxopt.matrix thì chúng ta khai báo một list bao gồm các list bên trong mà mỗi một list bên trong là 1 véc tơ cột. Xem thêm tại [create matrices - cvopt](https://cvxopt.org/examples/tutorial/creating-matrices.html). Như vậy các ma trận và véc tơ được tạo thành từ cvxopt.matrix phải là transpose của numpy.
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: BKWYmAV5HZWb
outputId: f9c97e95-c710-4d50-bc45-870ee916688d
---
import pprint
from cvxopt import matrix, solvers
c = matrix([50., 150., 120., 40.])

# moi dong cua matrix la 1 list va list nay phai la 1 vec to cot 
G = matrix([[1., 0., -1., 0., 0., 0.],
            [1., 0., 0., -1., 0., 0.],
            [0., 1., 0., 0., -1., 0.],
            [0., 1., 0., 0., 0., -1.]])

h = matrix([1000., 3000., 0., 0., 0., 0.])

A = matrix([[1., 0.],
            [0., 1.],
            [1., 0.],
            [0., 1.]])

b = matrix([1800., 2000.])

solvers.options['show_progress'] = True
sol = solvers.lp(c, G, h, A, b)

print('Solution"')
print(sol['x'])
```

+++ {"id": "IlwjJJjRylu7"}

Lưu ý: ở trên chúng ta sử dụng hàm matrix của cvxopt để khởi tạo ma trận. Bên trong chúng ta khai báo một list bao gồm các list con mà mỗi một list con sẽ đại diện cho một véc tơ cột của ma trận (khác với numpy là mỗi một list là một véc tơ dòng). 

+++ {"id": "jQrqKu_gPH2f"}

Như vậy lời giải của bài toán là giao 1000 cuốn từ Thái Bình đi Hà Nội, giao 2000 cuốn từ Khánh Hoà đi Hồ Chí Minh và 800 cuốn từ Khánh Hoà đi Hà Nội. Ta có thể dễ dàng suy luận ra điều này trên thực tế.

+++ {"id": "7dIl562iVvk5"}

## 7.2.2. Bài toán chi phí sản xuất

Bài toán chi phí sản xuất là một bài toán có tính ứng dụng cao thường được ứng dụng vào quá trình sản xuất của các nhà máy. Bài toán này giúp cho nhà quản lý có thể tối đa hoá nguồn lực đầu vào để mang lại lợi nhuận là lớn nhất cho doanh nghiệp. Sau khi đọc xong mục này, bạn cũng có thể ứng dụng được bài toán chi phí sản xuất vào quản lý công xưởng và doanh nghiệp của mình.

+++ {"id": "r4AejempPf6r"}

### Bài toán

Một nhà máy sản xuất ra 4 loại sản phẩm là: 

* Đũa tre (5 triệu/tấn)
* Tăm tre (6 triệu/tấn)
* Chân hương (4.5 triệu/tấn)
* Xiên thịt (5.5 triệu/tấn)

Biết rằng để sản xuất được 1 tấn của lần lượt các loại sản phẩm trên thì nhà máy cần tiêu tốn nguyên liệu và nhân công như sau:

* Nguyên liệu và chi phí nguyên liệu:

![](https://imgur.com/8uvggql.png)

* Số giờ làm việc của nhân công:

![](https://imgur.com/cGPEIHE.png)

* Lợi nhuận bán ra của từng loại:

![](https://imgur.com/SQEx1CR.png)

Kế hoạch trong tháng tới nhà máy có 1000 giờ làm việc và tối đa 5 tấn tre, 0.2 tấn hương liệu và 10 tấn than sấy. Hỏi nhà máy cần lập kết hoạch sản xuất như thế nào để tối đa lợi nhuận.

+++ {"id": "ugjOG-nYWJld"}

### Lời giải

Giả định sản lượng sản xuất của mỗi loại lần lượt là $x_1, x_2, x_3, x_4$. Bài toán tối ưu lợi nhuận:

$$\mathbf{x} = \arg \max_{\mathbf{x}} f(\mathbf{x}) = 18 x_1 + 30 x_2 + 22 x_3 + 25 x_4$$

Tương đương với bài toán:

$$\mathbf{x} = \arg \min_{\mathbf{x}} f(\mathbf{x}) = -18 x_1 -30 x_2 -22 x_3 -25 x_4$$

* Ràng buộc về nguyên liệu:

Đối với Tre: $1.1 x_1 + 1.5 x_2 + 1.4 x_3 + 1.2 x_2 \leq 5$

Đối với Hương liệu: $0.03 x_1 + 0.06 x_2 + 0.04 x_3 + 0.04 x_4 \leq 0.2$

Đối với than sấy: $x_1 + 2 x_2 + 2.2 x_3 + 2.1 x_4 \leq 10$

* Ràng buộc về số giờ làm việc:

$200 x_1 + 300 x_2 + 220 x_3+ 240 x_4 \leq 1000$

* Ràng buộc về số lượng $x_1 , x_2, x_3, x_4 \geq 0$

Như vậy:

$$\mathbf{c} = [-18, -30, -22, -25]^{\intercal}$$

$$\mathbf{G} = \begin{bmatrix}
1.1 & 1.5 & 1.4 & 1.2 \\
0.03 & 0.06 & 0.04 & 0.04 \\
1 & 2 & 2.2 & 2.1 \\
-1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & -1 \\
\end{bmatrix}$$

$$\mathbf{h} = [5, 0.2, 10, 0, 0, 0, 0]^{\intercal}$$

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: vgkxO3V6brvf
outputId: 4715aa2c-1f2a-4383-e04c-99a8efcb08a1
---
import pprint
from cvxopt import matrix, solvers
c = matrix([-18., -30., -22., -25.])

# We have to transpose G
G = matrix([[1.1, 0.03, 1., -1., 0., 0., 0.],
            [1.5, 0.06, 2., 0., -1., 0., 0.],
            [1.4, 0.04, 2.2, 0., 0., -1., 0.],
            [1.2, 0.04, 2.1, 0., 0., 0., -1.]])

h = matrix([5., 0.2, 10., 0., 0., 0., 0.])

solvers.options['show_progress'] = True
sol = solvers.lp(c, G, h)

print('Solution"')
print(sol['x'])
```

+++ {"id": "d_c__w1idCIQ"}

Như vậy để tối đa hoá lợi nhuận chúng ta cần tập trung vào sản xuất toàn bộ là `xiên thịt` với số lượng tối đa là $4.17$ tấn.

+++ {"id": "wNQlYLUBWC4p"}

## 7.2.3. Bài toán đóng thùng

+++ {"id": "mUplUSrTzUrn"}

### Bài toán
Một người cần chở 100 $m^3$ hàng hoá về kho. Mỗi một lượt vận chuyển tốn 100 USD cả chiều đi lẫn về. Để vận chuyển thì người đó đóng một chiếc thùng chở hàng với chi phí để đóng là 20 USD/$m^2$. Hỏi người đó cần đóng thùng hàng với kích thước các cạnh như thế nào để tối ưu hoá chi phí vận chuyển. Trên thực tế có thể có thêm ràng buộc về chiều dài, rộng, cao của thùng chở hàng nhưng ở đây chúng ta để đơn giản chúng ta giả định rằng thùng chở hàng có thể có kích thước tuỳ ý.

+++ {"id": "af-2YkmsWboY"}

### Lời giải

Chúng ta giả định kích thước dài, rộng, cao của thùng hàng lần lượt là $x, y, z$ m. Như vậy thể tích của thùng hàng là $xyz$. Chúng ta có hai loại chi phí phải chi trả:

* Chi phí chuyên chở: Số lượt cần chuyên chở là $\frac{100}{xyz}$. Như vậy chi phí tổng cộng sẽ là: $\frac{100}{xyz}\times 100 = \frac{10000}{xyz}$.

* Chi phí đóng thùng: $20(xy + yz+ zx)$.

Tổng chi phí người đó phải trả:

$$\frac{10000}{xyz} + 20(xy + yz + zx)$$

Áp dụng BĐT cauchuy ta dễ dàng suy ra chi phí tốt thiểu mà người đó cần chuyên chở là:

$$\begin{eqnarray}\frac{10000}{xyz} + 20(xy + yz + zx) & = & 20 (\frac{250}{xyz} + \frac{250}{xyz} + xy + yz + zx) \\
& \geq & 20 (5 (\frac{250}{xyz}.\frac{250}{xyz} xy.yz.zx)^{\frac{1}{5}}) \\
& = & 100.250^{2/5}&
\end{eqnarray}$$

Đẳng thức xảy ra khi $x = y = z = 250^{1/5}$.

Lời giải trên là khá đơn giản và thực tế dường như việc giải bài toán đó là không dễ dàng. Sẽ như thế nào nếu chúng ta đưa thêm các điều kiện ràng buộc về các chiều $x, y, z$ của thùng container? Liệu có tồn tại lời giải dạng tổng quát cho bài toán trên không? Chúng ta cùng tìm hiểu bài toán dạng tổng quát của bài toán trên là _Geometric Programming_.

+++ {"id": "PcKGMQMHBhMo"}

# 7.3. Bài toán _Geometric Programming GP_

Trong phần này, chúng ta mô tả một nhóm các bài toán tối ưu hóa không lồi
ở dạng tự nhiên của chúng. Tuy nhiên, những bài toán này có thể được chuyển đổi thành bài toán tối ưu lồi, bằng cách thay đổi các biến số và chuyển đổi mục tiêu và các chức năng ràng buộc.

+++ {"id": "CSg-5cXM3iEA"}

## 7.3.1. Đơn thức (_monomials_) và đa thức (_posynomials_)

Một hàm $f: \mathbb{R}^n \mapsto \mathbb{R}$ với miền xác định $\text{dom} g = \mathbb{R}^{n}_{++}$ được định nghĩa là:

$$f(x) = c x_1^{a_1}x_2^{a_2}\dots x_n^{a_n}$$

với $c > 0$ và $a_i \in \mathbb{R}$ được gọi là hàm đơn thức (_monomial funciton_). Số mũ $a_i$ của đơn thức có thể là bất kì giá trị thực nào, bao gồm cả phân số và số âm, nhưng hệ số $c$ chỉ có thể dương (khái niệm _monomial_ ở đây có thể khác biệt so với định nghĩa trong đại số , khi mà số mũ bắt buộc phải là một số nguyên không âm). Tổng của các đơn thức là một hàm có dạng như bên dưới:

$$f(x) = \sum_{i=1}^K c_k x_1^{a_{1k}}x_2^{a_{2k}}\dots x_n^{a_{nk}}$$

ở đây $k > 0$, được gọi là (_đa thức_) _posynomial function_ với $K$ phần tử.

+++ {"id": "kMF5SIVF3vc0"}

## 7.3.2. Bài toán tối ưu dạng đa thức

Một bài toán tối ưu có dạng:

$$\begin{eqnarray}
\mathbf{x}   &=& \arg\min_{\mathbf{x}} f_0(\mathbf{x}) \\ 
\text{subject to:}~ && f_i(x) \leq 1,  ~~ i = 1, 2, \dots, m \\ 
                    && h_j(x) = 1, ~~ j = 1, 2, \dots, p
\end{eqnarray}
$$

Trong đó $f_0, f_1, \dots, f_m$ là các _đa thức_ và $h_1, h_2, \dots, h_p$ là các _đơn thức_ được gọi là bài toán _Geometric Programming_ (_GP_). Điều kiện $\mathbf{x} \succeq 0$ được ẩn đi.

**Ví dụ:**

$$
\begin{eqnarray}
    (x, y, z)    &=& \arg\min_{x, y, z} xz/y                          \\ 
\text{subject to:}~ && 1 \leq x \leq 3 \\ 
 && x^2 + 3y/z \leq \sqrt{y} \\ 
 && z/y^2 = x
\end{eqnarray}
$$

Có thể được viết thành dưới dạng _GP_

$$\begin{eqnarray}
    (x, y, z)    &=& \arg\min_{x, y, z} xy ^{-1}z                        \\ 
\text{subject to:}~ && x^{-1} \leq 1 \\ 
&& (1/3)x \leq 1 \\
&& x^2y^{-1/2} + 3y^{1/2}z^{-1} \leq 1 \\ 
&& x^{-1}y^{-2}z = 1
\end{eqnarray}$$

Bài toán này là non-convex vì hàm mục tiêu và các điều kiện ràng buộc đều không lồi

+++ {"id": "xT2BzsLw6gsR"}

## 7.3.3. Biến đổi GP về đạng convex

Nếu giữ nguyên các đơn thức thì rõ ràng điều kiện ràng buộc là không lồi. Nhưng nếu chúng ta đặt $y_k = log(x_k)$? Khi đó $f_k(\mathbf{x})$ trở thành:

$$\begin{eqnarray}f_k(\mathbf{x}) & = & \exp(\log c)\exp(a_{1k}\log(x_1)) \dots \exp(a_{nk} \log x_n)\\
& = & \exp(\mathbf{a}^{\intercal}\mathbf{y} + b)
\end{eqnarray}$$

Ở đây $b = \log c$. Lúc này ta thu được hàm $l(\mathbf{y}) = \exp(\mathbf{a}^{\intercal}\mathbf{y} + b)$ là một hàm lồi theo $\mathbf{y}$. Thật vậy:

$$\frac{\partial^{2}l(\mathbf{y})}{\partial^{2} y_i} = a_{ik}^2\exp(\mathbf{a}^{\intercal}\mathbf{y}+b) > 0$$

Hàm $\mathbf{a}^{\intercal}\mathbf{y} + \mathbf{b}$ còn được gọi là hàm affine, đây đồng thời cũng là một hàm lồi.

Như vậy hàm mục tiêu của bài toán tối ưu GP có thể được viết lại thành:

$$f_k(\mathbf{x}) = g(\mathbf{y}) = \sum_{k = 1}^K \exp(\mathbf{a}_k^{\intercal}\mathbf{y} + b_k)$$

Trong đó $\mathbf{a}_k = [a_{1k}, a_{2k}, \dots, a_{nk}]^{\intercal}$ chính là các véc tơ số mũ của đơn thức và $b_k = \log c_k$.

Như vậy lúc này hàm mục tiêu đã được biến đổi thành tổng của $\exp$ của các hàm affine. Do các hàm thành phần đều lồi nên tổng này là một hàm lồi.

Bài toán GP được viết lại thành:

$$
\begin{eqnarray}
    \mathbf{y}    &=& \arg\min_{\mathbf{y}} \sum_{k=1}^{K} \exp(\mathbf{a}_{0k}^{\intercal}\mathbf{y} + b_{k})                      \\ 
\text{subject to:}~ && \sum_{k=1}^{K_i} \exp(\mathbf{a}_{ik}^{\intercal}\mathbf{y} + b_{ik}) \leq 1, ~~, i = 1, \dots, m \\ 
&& \exp(\mathbf{g}_j^{\intercal}\mathbf{y} + h_j) = 1, ~ j= 1, \dots, p
\end{eqnarray}
$$

Tron đó $\mathbf{a}_{ik} \in \mathbb{R}^{n}, ~ \forall i = \overline{1, m}$ và $\mathbf{g}_j \in \mathbb{R}^{n}$.

Đây là có các hàm mục tiêu là lồi và các điều kiện ràng buộc cũng là lồi. Chúng ta có thể chuyển chúng sang bài toán dạng lồi bằng cách lấy logarith như sau:


**GP convex form:**

$$\begin{eqnarray}
 \text{minimize}_{\mathbf{y}} \tilde f(\mathbf{y}) & = &  \log[\sum_{i=1}^{K} \exp(\mathbf{a}_{0k}^{\intercal} \mathbf{y} + b_{0k})]\\
\text{subject to:}~ && \log \sum_{k=1}^{K_i} \exp(\mathbf{a}_{ik}^{\intercal}\mathbf{y} + b_{ik}) \leq 0, ~~, i = 1, \dots, m \\ 
&& \tilde {h}(\mathbf{y}) = \mathbf{g}_j^{\intercal}\mathbf{y} + h_j = 0, ~ j= 1, \dots, p
\end{eqnarray}$$

+++ {"id": "pHzJDGBZFHGE"}

## 7.3.4. Giải bài toán đóng thùng bằng CVXOPT

Quay trở lại bài toán đóng thùng, đây là một bài toán có hàm mục tiêu dạng đa thức: 

$$\begin{eqnarray}f(x, y, z) & = & \frac{10000}{xyz} + 20(xy + yz + zx) \\
& = & 10000 x^{-1}y^{-1}z^{-1} + 20xyz^{0} + 20 x^{0}yz + 20 xy^{0}z \\
\rightarrow \tilde f(x, y, z) & = & \log [ \exp(\log10000 - \log{x} - \log{y} - \log{z}) \\
& + & \exp(\log 20 + \log x + \log y) \\
& + & \exp(\log 20 + \log{y} + \log{z}) \\
& + & \exp(\log{20} + \log{x} + \log{z})]
\end{eqnarray}$$

và đồng thời bài toán này không có điều kiện ràng buộc. Chúng ta có thể tìm kiếm _optimal point_ bằng package [CVXOPT](https://cvxopt.org/userguide/solvers.html) theo hướng dẫn:

![](https://imgur.com/JjAY7fm.jpeg)

Trong đó $F$ là ma trận hệ số của $\log{x}, \log{y}, \log{z}$. Ta sẽ có:


$$F = \begin{bmatrix}
-1 & -1 & -1 \\
1 & 1 & 0 \\
0 & 1 & 1 \\
1 & 0 & 1
\end{bmatrix}$$
và 

$$\mathbf{g} = [\log 10000, \log 20, \log 20, \log 20]^{\intercal}$$

Thế vào hàm `solvers.gp()` bên dưới ta được:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: kJd_CIRjc2EC
outputId: 283a49ac-15dc-46c9-f02e-cfac891373d8
---
from cvxopt import matrix, solvers
from math import log, exp# gp
from numpy import array
import numpy as np

K = [4]
F = matrix([[-1., 1., 1., 0.],
            [-1., 1., 0., 1.],
            [-1., 0., 1., 1.]])

g = matrix([log(10000.), log(20.), log(20.), log(20.)])
solvers.options['show_progress'] = True
sol = solvers.gp(K, F, g)

print('Solution:')
print(np.exp(np.array(sol['x'])))

print('\nchecking sol^5')
print(np.exp(np.array(sol['x']))**5)
```

+++ {"id": "J7NhnwULbirT"}

Ta nhận thấy nghiệm của bài toán thu được là $x = y = z = (250)^{1/5}$

+++ {"id": "mvYUl7GKBx2k"}

# 7.4. Bài toán _Quadratic Programming_ (_QP_)

_Quadratic Programming_ là một bài toán tối ưu mà bạn sẽ bắt gặp khá nhiều trong machine learning. Chẳng hạn như bài toán tối ưu trong mô hình [SVM](https://phamdinhkhanh.github.io/deepai-book/ch_ml/index_SVM.html) có thể chuyển về dạng _Quadratic Programming_ hoặc bài toán hồi qui tuyến tính. Để hình dung trực quan về bài toán _Quadratic Programming_ chúng ta cùng lấy một ví dụ về bài toán tìm bờ gần nhất.

**Bài toán tìm bờ:** Một con thuyền đang đậu tại 1 điểm $\mathbf{x}$ ở ngoài khơi của một hòn đảo mà có các cạnh dạng _đa giác lồi_ (_polygon_). Tìm toạ độ của điểm gần nhất trên hòn đảo để tàu cập bến.

+++ {"id": "eMkMsNmS9C5j"}

## 7.4.1. Dạng _Quadratic Programming - QP_ 

+++ {"id": "_EiZX_878h0w"}


Bài toán $(1)$ được gọi là _quadratic program (QP)_ nếu như hàm mục tiêu là (convex) quadratic và các hàm ràng buộc là những hàm affine. Một bài toán _QP_ có thể được biểu diễn dưới dạng:

$$\begin{eqnarray}
&& \text{minimize}_{\mathbf{x}} ~~ \frac{1}{2} \mathbf{x}^{\intercal}\mathbf{P}\mathbf{x} + \mathbf{q}^{\intercal}\mathbf{x} + r \\
&& \text{subject to:}~ \mathbf{G}\mathbf{x} \preceq \mathbf{h}\\ 
&& ~~~~~~~~~~~~~~~~~~~\mathbf{A}\mathbf{x} = \mathbf{b}
\end{eqnarray}$$

Ở đây $\mathbf{P} \in \mathbf{S}^{n}_{+}$ là một ma trận bán xác định dương, $\mathbf{G} \in \mathbb{R}^{m \times n}$, $\mathbf{A} \in \mathbb{R}^{p \times n}$, $\mathbf{h} \in \mathbb{R}^{m}$, $\mathbf{b} \in \mathbb{R}^{p}$. Trong bài toán _QP_, chúng ta tối thiểu hoá một _convex quadratic function_ trên một đa diện lồi _polyhedron_.

![](https://imgur.com/DIm7mNa.png)

**Hình 1**: Hình ảnh mô phỏng _QP_. Tập _feasible set_ là $\mathcal{P}$, đây là một hình đa diện lồi được đổ bóng. Các đường đồng mức (_contour lines_) là tập hợp những giá trị của $x$ trả về cùng một giá trị hàm mục tiêu, chúng là một hàm lồi _convex quadratic_ được thể hiện bằng đường nét đứt. Điểm tối ưu $x^*$ là điểm tiếp xúc giữa đường đồng mức có giá trị nhỏ nhất với tập _feasible set_ $\mathcal{P}$.

+++ {"id": "XdeNIM2hIA4c"}

## 7.4.2.Giải bài toán _QP_ trên CVXOPT


![](https://i.imgur.com/rfj3GCe.jpeg)

**Hình 2**: Đồ thị minh hoạ cho bài toán tìm điểm cập bến gần nhất.

Quay trở lại bài toán tìm điểm cập bến, để cụ thể hoá chúng ta giả định rằng vị trí của tàu có toạ độ là $(6, 3)$ hòn đảo chính là tập _feasible set_ $\mathcal{P}$ được gạch chéo. Khi đó mục tiêu của chúng ta là tìm điểm $(x, y) \in \mathcal{P}$ sao cho khoảng cách từ điểm đó tới tàu là nhỏ nhất. Tức là tương đương với bài toán tối ưu:

$$\begin{eqnarray}
& \text{minimize}_{x, y} & ~~ (x - 6)^2 + (y-3)^2 \\
\text{subject to:} & 2x + y & \leq 8 \\
& x + 4y & \leq 16 \\
& x + y & \leq 5 \\
& x, y & \geq 0 
\end{eqnarray}$$


Hàm mục tiêu của bài toán trên tương đương với:

$$f(x, y) = x^2 + y^2 - 12x - 6 y + 45$$

Như vậy:

$$\mathbf{P} = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}$$

$$\mathbf{q} = [-12, -6]^{\intercal}$$

$r = 45$

$$\mathbf{G} = \begin{bmatrix}
2 & 1 \\
1 & 4 \\
1 & 1 \\
-1 & 0 \\
0 & -1
\end{bmatrix}$$

$$\mathbf{h} = [8, 16, 5, 0, 0]^{\intercal}$$

Chúng ta giải bài toán _QP_ trên CVXOPT như sau:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: W1U74j--KBni
outputId: a311aad0-23ab-46dc-b84c-aad3e850ec04
---
from cvxopt import matrix, solvers
P = matrix([[1., 0.], 
            [0., 1.]])
q = matrix([-12., -16.])
G = matrix([[2., 1., 1., -1., 0.], [1., 4., 1., 0., -1.]])
h = matrix([8., 16., 5., 0., 0])

solvers.options['show_progress'] = False
sol = solvers.qp(P, q, G, h)

print('Solution:')
print(sol['x'])
```

+++ {"id": "EzT6FvPMbqer"}

# 7.5. Tổng kết

Như vậy ở chương này chúng ta đã tìm hiểu về 3 dạng bài toán phổ biến trong tối ưu đó là bài toán: _Linear Programming_, _Geometric Programming_ và _Quadratic Programming_ kèm theo cách giải chúng trên package CVXOPT. Những lớp bài toán này không chỉ tồn tại trên lý thuyết mà trên thực tế chúng còn được áp dụng giải quyết các vấn đề trong tối ưu vận chuyển, tối ưu chi phí sản xuất. Khi nắm được những công cụ này, bạn có thể áp dụng chúng một cách hiệu quả vào các vấn đề của doanh nghiệp.

+++ {"id": "PysB8lmDUZXK"}

# 7.6. Bài tập

1. Một doanh nghiệp sản xuất trứng gà có 2 kho hàng tại _Cần Thơ_ (2 triệu quả) và _Hà Nội_ (3 triệu quả). Doanh nghiệp cần giao trứng gà của mình đến các cửa hàng tại các tỉnh _Khánh Hoà_ (0.5 triệu quả), _Bình Dương_ (1.2 triệu quả) và _Thanh Hoá_ (3.2 triệu quả) với chi phí vận chuyển (VNĐ/triệu quả) được cho ở bảng sau:

![](https://imgur.com/OGXgVaz.png)

Hỏi doanh nghiệp này cần giao hàng như thế nào để chi phí là thấp nhất mà vẫn đảm bảo giao đủ hàng.

2. Cần giao hàng với số lượng yêu cầu tương tự như bài toán trên nhưng mục tiêu là tối ưu chi phí thời gian. Biết mỗi container chỉ chở được 0.1 triệu quả/ chuyến và thời gian của các chuyến hàng cả đi lẫn về là:

![](https://imgur.com/753v92o.png)

3. Một doanh nghiệp sản xuất bánh với các loại bánh bao gồm `A, B, C, D` và chi phí nguyên liệu, số giờ lao động để sản xuất 1 tấn bánh và nguồn lực sẵn có như bên dưới:

![](https://imgur.com/jyPZbjd.png)

Hỏi doanh nghiệp này cần phải sản xuất mỗi loại bánh với số lượng bao nhiêu để tối đa hoá lợi nhuận.

4. Một người nông dân có một khu đất 20 ha và số ngày công làm việc sẵn có là 30 ngày. Hỏi người đó nên trồng loại rau, củ, quả nào để tối đa hoá lợi nhuận. Biết chi phí, lợi nhuận và nguồn lực được cho như bảng bên dưới:

![](https://imgur.com/5ZvcxEV.png)


+++ {"id": "tNgyFnSrcuYX"}

# 7.7. Tài liệu tham khảo

+++ {"id": "WhabF4KNknp5"}

1. [Chapter 4 - Convex Optimization , Boyd and Vandenberghe, Cambridge University Press, 2004](http://stanford.edu/~boyd/cvxbook/)

2. [Bài 17: Convex Optimization Problems - Machine Learing cơ bản](https://machinelearningcoban.com/2017/03/19/convexopt/#-geometric-programming)
