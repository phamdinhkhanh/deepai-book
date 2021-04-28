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

# 1. Giải tích tích phân

Trên thực tế, con người đã ứng dụng giải tích tích phân từ rất lâu trong đo đạc diện tích. Tính diện tích hình vuông, hình chữ nhật thì khá đơn giản nhưng làm sao để tính được diện tích của hình tròn? Người xưa đã biết cách chia nhỏ hình tròn thành những phần bằng nhau thông qua những lát cắt đi qua tâm. Sau đó xếp chúng lại theo hình răng cưa để thu được một hình _xấp xỉ_ chữ nhật, từ đó tìm ra công thức tính diện tích hình tròn. Số lượng các lát chia càng nhiều thì giá trị diện tích hình chữ nhật càng tiệm cận tới giá trị thật. Phương pháp như vậy được gọi là _vét cạn_. 

![](https://i.imgur.com/NuEZ7iQ.png)

Ngày nay _giải tích tích phân_ cho phép chúng ta tính diện tích hình tròn một cách nhẹ nhàng.

![](https://i.imgur.com/pcKqwdF.png)

Chẳng hạn để tính diện tích hình tròn thì từ phương trình hình tròn:

$$x^2+ y^2 = R^2$$

Ta sẽ tìm cách thể biểu diễn $y$ theo $x$ trên miền $[-R, R]$. Đối với nửa đường tròn nằm trên trục $y=0$:

$$y = f_{1}(x) = \sqrt{R^2-x^2}$$

và nửa đường tròn dưới trục $y=0$:

$$y = f_{2}(x) = -\sqrt{R^2-x^2}$$

Diện tích của hình tròn sẽ được tính thông qua công thức tích phân của hàm $y=f_{1}(x)$ và nhân đôi do đối xứng qua trục $y=0$:

$$\text{S} = 2\int_{-R}^{R} f(x) dx
$$

Ngoài tính diện tích hình tròn, _giải tích tích phân_ còn giúp ta tính được rất nhiều các hình phức tạp khác miễn là chúng ta biết được phương trình tường minh của chúng. Ngoài ra nó còn được dùng để  xác định phân phối xác suất của biến và tính toán xác suất của các sự kiện trên một miền xác định dựa trên phân phối xác suất. 

Qua các ví dụ trên chúng ta có thể thấy _giải tích tích phân_ đóng vai trò quan trọng như thế nào trong thực tiễn.

# 2. Giải tích vi phân

_giải tích vi phân_ cho phép ta hiểu được hàm số tốt hơn thông qua việc xác định tốc độ thay đổi của nó như thế nào tại một điểm dữ liệu. Thông qua đạo hàm chúng ta có thể khảo sát sự biến thiên của hàm số , tìm giá trị cực trị và tìm hướng di chuyển phù hợp để đi tới điểm cực trị địa phương. Đạo hàm của hàm $f(x) : \mathbb{R} \mapsto \mathbb{R}$ được tính theo công thức:

$$f'(x_0)= \lim_{x \rightarrow x_0}\frac{f(x)-f(x_0)}{x-x_0} \tag{1}$$

Một ứng dụng rất thực tế của đạo hàm đó là gia tốc trong vật lý khi nó cho ta biết vận tốc sẽ thay đổi thế nào trong một khoảng thời gian rất ngắn chẳng hạn như 1 giây. Khi bạn đi xe, vận tốc sẽ không đều tại mọi điểm thời gian mà tuỳ vào bạn tăng ga hay giảm ga mà vận tốc tương ứng sẽ tăng hoặc giảm. Gia tốc chính là đạo hàm của vận tốc theo thời gian.

Trong thực tiễn, chúng ta có thể tính đạo hàm cho bất kỳ một hàm số nào thông qua công thức lim ở phương trình $(1)$.

```{code-cell}
import torch

# Tính đạo hàm của vận tốc f(x) = x^2+2*x

def _derivative(x):
  f1 = x**2 + 2*x
  delta = 0.001 # giá trị delta ở mẫu
  x0 = x-delta
  f0 = x0**2 + 2*x0
  der = (f1-f0)/delta
  return der

_derivative(10)
```

## 2.1. Những đạo hàm cơ bản

Chúng ta có một bảng mẫu về đạo hàm của một số hàm có sẵn, đây là những công thức đạo hàm đã được chứng minh và chúng ta có thể sử dụng chúng làm cơ sở để tính những đạo hàm phức tạp hơn. Điều này cũng gần giống như để tính phép nhân hai chữ số chúng ta phải dựa trên bảng cửu chương:

| Phương trình     | Đạo hàm | Phương trình | Đạo hàm |
| ----------- | ----------- | ----------- | ----------- |
| $\frac{dx^a}{dx}$     | $ax^{a-1}$ | $\frac{de^x}{dx}$     | $e^{x}$ |
| $\frac{da^x}{dx}$     | $a^{x} \ln{a}$ | $\frac{d\ln x}{dx}$     | $\frac{1}{x}$ |
| $\frac{d \sin(x)}{dx}$     | $\cos x$ | $\frac{d \cos(x)}{dx}$     | $-\sin x$ |
| $\frac{d \tan(x)}{dx}$     | $1+\tan^{2}(x)$ | $\frac{d \arcsin(x)}{dx}$     | $\frac{1}{\sqrt{1-x^2}}$ |
| $\frac{d \arccos(x)}{dx}$     | $\frac{-1}{\sqrt{1-x^2}}$ | $\frac{d \arctan(x)}{dx}$     | $\frac{1}{1+x^2}$ |

Trong công thức $f(x) = x^2+2x$ ở trên ta thấy đạo hàm của hàm số  xấp xỉ 22. Đạo hàm của $f(x)$ cũng chính là $f'(x)=2x+2$ và cũng trả ra giá trị là 22. Như vậy công thức lim đã tính ra giá trị gần đúng của đạo hàm.

## 2.2. Các qui tắc đạo hàm

Ở trên là những đạo hàm của những hàm cơ bản . Ngoài ra chúng ta có các qui tắc đạo hàm như bên dưới để thực thi tính toán các đạo hàm của tổng, tích, phân số  và đạo hàm hàm hợp. Để rút gọn mình ký hiệu $\frac{d f(x)}{d x} \triangleq f'$ và  $\frac{d f(g(x))}{d g(x)} \triangleq f'(g)$:

* Tổng đạo hàm:

$$(f+g)' = f'+g'$$

* Đạo hàm của phân số:

$$(\frac{f}{g})' = \frac{f'g - fg'}{g^2}$$


* Đạo hàm tích (Product rule):

$$(fg)' = f'g + fg'$$

* Đạo hàm hàm hợp (Chain rule):

$$(f(g(x)))' = f_{g}'g'(x)$$

## 2.3. Khai triển taylor
Là một trong những công thức nền của giải tích vi phân, khai triển taylor cho phép ta tạo ra được biểu diễn xấp xỉ của nhiều công thức quan trọng trong giải tích như các biểu diễn hàm $\sin x, \cos x, e^{x}, \dots$ và rất nhiều các hàm số khác. Bạn cũng có thể dùng công thức taylor để chứng minh các hằng đẳng thức trong lượng giác và chứng minh các bất đẳng thức trong giải tích.

Trong công thức khai triển taylor chúng ta biểu diễn giá trị của hàm số thông qua các đạo hàm bậc cao của nó. Cụ thế nếu hàm $f(x): \mathbb{R} \mapsto \mathbb{R}$. Khi đó khai triển taylor của hàm $f(x)$ có dạng:

$$f(x) = f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + \dots + \frac{f^{(n)}(x_0)}{n!} (x-x_0)^n + O((x-x_0)^{n+1})$$

Thành phần cuối cùng là hàm $O((x-x_0)^{n+1})$ là một hàm big $O$ của $(x-x_0)^{n+1}$. Giá trị của hàm này có tỷ số so với $(x-x_0)^{n+1}$ là bị chặn. 
Tức là khi giá trị của $x$ là lân cận của $x_0$, luôn tồn tại các số dương $\epsilon$ và $M$ sao cho: 

$$|O((x-x_0)^{n+1})| < M (x-x_0)^{n+1},$$ 

$\forall x$ thoả mãn $\epsilon > |x-x_0| > 0$

Hay nói cách khác tỷ số giữa $|O((x-x_0)^{n+1})|$ và $(x-x_0)^{n+1}$ là bị chặn: 

$$\lim_{x \rightarrow x_0} \sup \frac{|O((x-x_0)^{n+1})|}{(x-x_0)^{n+1}} < M$$

Viết một cách ngắn gọn thì khai triển taylor có dạng:

$$f(x) = \sum_{i=0}^{+\infty} \frac{f^{(i)}(x_0)}{i!}(x-x_0)^i$$

Ngoài ra ta từ khai triển taylor cũng cho phép ta chứng minh được công thức đạo hàm ở $(1)$. Bởi nếu chỉ khai triển đến bậc nhất ta có:

$$f(x) = f(x_0) + f'(x_0)(x-x_0) + O((x-x_0)^2)$$

$$\begin{eqnarray}\lim_{x \rightarrow x_0} f'(x_0) & = & \lim_{x \rightarrow x_0} \frac{f(x)-f(x_0)}{x-x_0} + \lim_{x \rightarrow x_0} \frac{O(x-x_0)^{2}}{x-x_0} \\
& = & \lim_{x \rightarrow x_0} \frac{f(x)-f(x_0)}{x-x_0}
\end{eqnarray}$$

Dòng thứ 2 là vì 

$$\lim_{x \rightarrow x_0} |\frac{O(x-x_0)^2}{x-x_0}| = \lim_{x \rightarrow x_0} \frac{|O(x-x_0)^2|}{|x-x_0|} \leq \lim_{x \rightarrow x_0} \frac{M(x-x_0)^2}{|x-x_0|} = \lim_{x \rightarrow x_0} M|x-x_0| = 0$$

## 2.4. Đạo hàm riêng (partial derivative)

Đối với một hàm số có nhiều đầu vào thì đạo hàm riêng sẽ coi các chiều khác là hằng số và tính đạo hàm dựa trên chiều biến đổi. Chẳng hạn hàm $f(x_1, \dots, x_n)$ sẽ có đạo hàm riêng theo $x_i$ như sau:

$$\frac{\delta f(x_1, x_2 \dots, x_i, \dots, x_n)}{\delta x_i} = \lim_{\Delta \rightarrow 0} \frac{f(x_1, x_2 \dots, x_i+\Delta, \dots, x_n)- f(x_1, x_2 \dots, x_i, \dots, x_n)}{\Delta}$$

Ta coi như các chiều còn lại khác $x_i$ là hằng số và đạo hàm theo chỉ $x_i$.

Công thức đạo hàm riêng sẽ được sử dụng để tính gradient descent.

## 2.5. Gradient descent

Gradient là đạo hàm bậc nhất của một hàm số theo một véc tơ. Gradient descent là tên của phương pháp tối ưu theo vòng lặp dựa trên gradient nhằm tìm nghiệm tối ưu cục bộ của một hàm khả vi. Gradient descent một phương pháp thường xuyên được sử dụng trong huấn luyện và cập nhật hệ số của mạng nơ ron. Bởi chúng ta hình dung mạng neural network sẽ tìm cách tối ưu hàm loss function theo các véc tơ hệ số ở từng layer. Do đó chúng ta cần tính gradient của hàm loss function theo véc tơ hệ số.

$$\nabla_{\mathbf{w}} f(\mathbf{w}) = [\frac{\delta f(\mathbf{w})}{\delta  w_1}, \frac{\delta  f(\mathbf{w})}{\delta  w_2}, \dots, \frac{\delta  f(\mathbf{w})}{\delta  w_n} ]^{\intercal}
$$

Lưu ý đây là véc tơ cột vì có dấu chuyển vị. 

Trong véc tơ gradient thì mỗi thành phần $\frac{\delta f(w_i)}{\delta  w_i}$ là đạo hàm riêng của hàm $f(\mathbf{w})$ theo chiều $w_i$

## 2.6. Công thức đạo hàm vector-value

Có những véc tơ mà mỗi phần tử của nó là một hàm đối với biến $x$. Ví dụ $\mathbf{f}(x) = [f_1(x), f_2(x), \dots, f_n(x)]$. Khi đó đạo hàm của véc tơ cũng chính bằng đạo hàm của từng thành phần theo biến $x$.

$$\frac{d \mathbf{f}(x)}{d x} = [\frac{d f_1(x)}{dx}, \dots ,\frac{d f_n(x)}{dx}]$$

Lưu ý trong công thức trên hàm $f_i(x)$ là chỉ có một biến đầu vào $x$ nên ta sử dụng $d f_i(x)$ thay cho $\delta f_i(x)$. Đây là một qui ước mà bạn cần nhớ.

## 2.7. Đạo hàm vector-vector

Đạo hàm của một véc tơ hàm số $\mathbf{f(x)} = [f_1(x), \dots, f_n(x)]$ theo vector $\mathbf{x} = [x_1, \dots, x_m]$ chính là ma trận Jacobian.

$$\begin{eqnarray}
\nabla_{\mathbf{x}} \mathbf{f(x)} &\triangleq &
\left[
\begin{matrix}
    \frac{\partial f_1(\mathbf{x})}{\partial x_1} & \frac{\partial f_1(\mathbf{x})}{\partial x_2} & \dots & \frac{\partial f_1(\mathbf{x})}{\partial x_m} \\ 
    \frac{\partial f_2(\mathbf{x})}{\partial x_1} & \frac{\partial f_2(\mathbf{x})}{\partial x_2} & \dots & \frac{\partial f_2(\mathbf{x})}{\partial x_m} \\ 
    \vdots & \vdots & \ddots & \vdots \\ 
    \frac{\partial f_n(\mathbf{x})}{\partial x_1} & \frac{\partial f_n(\mathbf{x})}{\partial x_2} & \dots & \frac{\partial f_n(\mathbf{x})}{\partial x_m}
\end{matrix}
\right] \\
& = & 
\left[
\begin{matrix}
    \nabla f_1^{\intercal}(\mathbf{x}) & \nabla f_2^{\intercal}(\mathbf{x}) & \dots & \nabla f_n^{\intercal}(\mathbf{x})
\end{matrix} \right]
\end{eqnarray}$$

Ở công thức dòng thứ hai thì $\nabla f_i^{\intercal}(\mathbf{x})$ chính là một véc tơ dòng và là gradient descent của hàm số $f_i(\mathbf{x})$ theo véc tơ $\mathbf{x}$.

## 2.8. Lan truyền thuận (_feed forward_)

Mạng nơ ron network có kiến trúc gồm nhiều layers, mỗi layer bao gồm các unit nodes mà mỗi unit node đại diện cho một chiều đầu vào của dữ liệu tại layer đó.

![](https://imgur.com/m19Vtoi.png)

**Hình 1**: Kiến trúc mạng nơ ron với hai layers. $x_1, x_2, x_3$ là các biến đầu vào, $\mathbf{W}$ là ma trận hệ số có $\mathbf{w}_{ij}$ là kết nối từ input node thứ $j$ của layer trước tới output node thứ $j$ của layer liền sau. $b_2$ hệ số tự do không phụ thuộc vào dữ liệu. Giá trị được nhân với $b_2$ luôn cố định là $1$ và giá trị này được gọi là giá trị mở rộng. $f(h_i)$ chính là hàm activation để tạo ra tính phi tuyến cho mạng nơ ron. Đầu ra sau khi đi qua hàm activation là $u_i$ sẽ tiếp tục được dùng để tính phân phối xác suất dự báo ở sau cùng là $s$.

Lan truyền thuận là quá trình từ input tính ra phân phối xác suất ở output. Ta có thể tóm tắt các bước tính lan này theo tuần tự công thức:

1. Xuất phát từ input : $x$
2. Tính layer 1: $z = \mathbf{W}x + \mathbf{b}$
3. Tính activation: $\mathbf{h} = f(z)$
4. Tính output: $s = \mathbf{u}^{\intercal}\mathbf{h}$

Các bước tính toán này có thể được tóm lược trong một sơ đồ tính toán (_computational graph_).

![](https://imgur.com/98Lq2Xc.png)

Xuất phát từ $x$, các node $\odot$ là đại diện cho phép nhân và $\oplus$ đại diện cho phép cộng. Chiều của mũi tên thể hiện thứ tự tính toán.

## 2.9. Chain rule trong lan truyền ngược (_backpropagation_)

Quá trình backpropagation sẽ thực hiện tính toán gradient descent và cập nhật hệ số. backpropagation có thể coi như _linh hồn _ của quá trình huấn luyện các mạng nơ ron vì không có chúng, chúng ta không thể cập nhật hệ số mạng nơ ron (_weights_) để tiến tới các điểm cực trị địa phương. Công thức backpropagation sẽ thực hiện hai việc:

* Tính toán gradient descent tại mỗi layer theo chiều từ cuối trở về đầu. Vậy nên công thức này mới có tên gọi là lan truyền ngược.

* Cập nhật gradient descent tại mỗi layer theo giá trị được tính ở trên.

![](https://imgur.com/LWWkkPA.png)

Để tính được gradient descent trong backpropagation thì chúng ta dựa vào công thức chain rule. Chẳng hạn như trong sơ đồ trên, các node và mũi tên màu xanh dương sẽ đi ngược chiều với mũi tên màu đen là đại diện cho các gradient descent theo chain rule.

Giả sử ta xét qúa trình backpropagation tại một node cụ thể  có đầu vào là $\mathbf{z}$ và đầu ra là $\mathbf{h}$. $\mathbf{h} = f(\mathbf{z})$.

![](https://imgur.com/5b5GQuK.png)


Trong backpropagation, từ output $\mathbf{s}$, để tính đạo hàm $\frac{\delta s}{\delta z}$ ta sẽ không tính được trực tiếp mà phải thông qua các đạo hàm thành phần trong chain rule lần lượt là đạo hàm theo $\mathbf{h}$ và sau đó là theo $\mathbf{z}$:

$$\frac{\delta s}{\delta z} = \frac{\delta s}{\delta \mathbf{h}} \frac{\delta \mathbf{h}}{\delta z}$$

Những gradient descent được thực hiện ngay tại node đó gọi là _local gradient_, gradient descent ở phía sau node là _downstream gradient_ và phía trước node là _upstream gradient_. Công thức trên có thể gói gọn thành:

$$\text{downstream} = \text{upstream} \times \text{local}$$

## 2.10. Một số công thức đạo hàm đáng nhớ

$\mathbf{A}$ là một ma trận và $\mathbf{x}$ là một véc tơ. Khi đó

* Đạo hàm của tích ma trận với véc tơ:

$$\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$$

* Đạo hàm của tích véc tơ với ma trận:

$$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} = \mathbf{A}$$

* Đạo hàm của tích hai véc tơ

$$\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$$

* Đạo hàm của tích véc tơ nhân với ma trận và véc tơ (bạn sẽ gặp lại tích này trong phân tích suy biến).

$$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x} = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$$

Đây là những công thức quan trọng mà các bạn cần ghi nhớ.

# 3. Bài tập

Tính các đạo hàm sau:

1. $\frac{1}{1+e^x}$
2. $\ln (x^2+1)$
3. $\sqrt{x+1}+x$
4. $\frac{\sin(x)}{\sqrt{x+1}}$
5. Với $\mathbf{A}$ là ma trận, $\mathbf{w}$ là véc tơ. Tính: $\nabla_{\mathbf{w}} ||\mathbf{Aw}-\mathbf{y}||^{2}$.
6. $\nabla_{\mathbf{w}}^{2} ||\mathbf{Aw}-\mathbf{y}||^{2}$
7. $\nabla_{\mathbf{x}}\mathbf{a^{\intercal}\mathbf{x}^{\intercal}\mathbf{x}\mathbf{b}}$
8. Thực hiện khai triển taylor với lần lượt các hàm số $e^{x},
\sin(x), \cos(x)$

# 4. Tài liệu tham khảo

1. [standford - cs224n](https://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture04-neuralnets.pdf)

2. [d2l - đạo hàm riệng](https://d2l.aivivn.com/chapter_preliminaries/calculus_vn.html#dao-ham-rieng)

3. [matrix calculus - wiki](https://en.wikipedia.org/wiki/Matrix_calculus)

4. [math - machine learning cơ bản](https://machinelearningcoban.com/math/)
