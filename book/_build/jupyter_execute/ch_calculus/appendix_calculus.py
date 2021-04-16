# 1. Giải tích tích phân

Thực ra con người đã ứng dụng giải tích tích phân từ rất lâu trong đo đạc diện tích. Tính diện tích hình vuông, hình chữ nhật thì khá đơn giản nhưng làm sao để tính được diện tích của hình tròn? Người xưa đã biết cách chia nhỏ hình tròn thành những miếng bánh bằng nhau bằng những lát cắt đi qua tâm hình tròn. Sau đó xếp chúng lại theo hình răng cưa để thu được một hình chữ nhật. Dựa trên hình chữ nhật để tìm ra công thức tính diện tích hình tròn. Cách làm như vậy được gọi là phương pháp vét cạn.

![]()

Ngày nay việc với sự phát triển của _giải tích tích phân_ thì việc tính diện tích hình tròn không còn là một vấn đề quá khó. Từ phương trình hình tròn:

$$x^2+ y^2 = R^2$$

Ta sẽ tìm cách thể biểu diễn $y$ theo $x$ trên miền $[-R, R]$. Đối với nửa đường tròn nằm trên trục $y=0$:

$$y = f(x) = \sqrt{R^2-x^2}$$

và nửa đường tròn dưới trục $y=0$:

$$y = f(x) = -\sqrt{R^2-x^2}$$

Diện tích của hình tròn sẽ được tính thông qua công thức tích phân của hàm $y=f(x)$:

$$\text{area} = 2\int_{-R}^{R} f(x) dx
$$

Ngoài tính diện tích hình tròn, _giải tích tích phân_ còn giúp ta tính được rất nhiều các hình phức tạp khác miễn là chúng ta biết được phương trình tường minh của chúng. Ngoài ra nó còn được dùng để  xác định phân phối xác suất của biến và tính toán xác suất của các sự kiện trên một miền xác định dựa trên phân phối xác suất. 

Qua các ví dụ trên chúng ta có thể thấy _giải tích tích phân_ đóng vai trò quan trọng như thế nào trong thực tiễn.

# 2. Giải tích vi phân

_giải tích vi phân_ cho phép ta hiểu được hàm số tốt hơn thông qua việc xác định tốc độ thay đổi của nó như thế nào tại một điểm dữ liệu. Thông qua đạo hàm chúng ta có thể khảo sát sự biến thiên của hàm số , tìm giá trị cực trị và tìm hướng di chuyển phù hợp để đi tới điểm cực trị địa phương. Đạo hàm của hàm $f(x) : \mathbb{R} \mapsto \mathbb{R}$ được tính theo công thức:

$$f'(x_0)= \lim_{x \rightarrow x_0}\frac{f(x)-f(x_0)}{x-x_0} \tag{1}$$

Một ứng dụng rất thực tế của đạo hàm đó là gia tốc trong vật lý khi nó cho ta biết vận tốc sẽ thay đổi thế nào trong một khoảng thời gian rất ngắn chẳng hạn như 1 giây. Khi bạn đi xe, vận tốc sẽ không đều tại mọi điểm thời gian mà tuỳ vào bạn tăng ga hay giảm ga mà vận tốc tương ứng sẽ tăng hoặc giảm. Gia tốc chính là đạo hàm của vận tốc theo thời gian.

Trong thực tiễn, chúng ta có thể tính đạo hàm cho bất kỳ một hàm số nào thông qua công thức lim ở phương trình $(1)$.

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

## 2.1. Những đạo hàm cơ bản

Trên thực tế thì chúng ta có một bảng mẫu có sẵn của một số đạo hàm cơ bản, đây là những đạo hàm đã được chứng minh và chúng ta có thể sử dụng chúng như cơ sở để tính những đạo hàm khác:

| Phương trình      | Đạo hàm |
| ----------- | ----------- |
| $\frac{dx^a}{dx}$     | $ax^{a-1}$ |
| $\frac{de^x}{dx}$     | $e^{x}$ |
| $\frac{da^x}{dx}$     | $a^{x} \ln{a}$ |
| $\frac{d\ln x}{dx}$     | $\frac{1}{x}$ |
| $\frac{d \sin(x)}{dx}$     | $\cos x$ |
| $\frac{d \cos(x)}{dx}$     | $-\sin x$ |
| $\frac{d \tan(x)}{dx}$     | $1+\tan^{2}(x)$ |
| $\frac{d \arcsin(x)}{dx}$     | $\frac{1}{\sqrt{1-x^2}}$ |
| $\frac{d \arccos(x)}{dx}$     | $\frac{-1}{\sqrt{1-x^2}}$ |
| $\frac{d \arctan(x)}{dx}$     | $\frac{1}{1+x^2}$ |

Trong công thức $f(x) = x^2+2x$ ở trên ta thấy đạo hàm của hàm số  sấp xỉ 22. Đạo hàm của $f(x)$ cũng chính là $f'(x)=2x+2$ và cũng trả ra giá trị là 22. Như vậy công thức lim đã tính ra giá trị gần đúng của đạo hàm.

## 2.2. Các qui tắc đạo hàm

Ở trên là những đạo hàm của những hàm cơ bản . Ngoài ra chúng ta có các qui tắc đạo hàm như bên dưới để thực thi tính toán đạo hàm với những hàm phức tạp hơn. Để rút gọn mình ký hiệu $\frac{d f(x)}{d x} \triangleq f'$ và  $\frac{d f(g(x))}{d g(x)} \triangleq f'(g)$:

* Tổng đạo hàm:

$$(f+g)' = f'+g'$$

* Đạo hàm của phân số:

$$(\frac{f}{g})' = \frac{f'g - fg'}{g^2}$$


* Đạo hàm tích (Product rule):

$$(fg)' = f'g + fg'$$

* Đạo hàm hàm hợp (Chain rule):

$$(f(g(x)))' = f_{g}'g'(x)$$

## 2.3. Khai triển taylor
Là một trong những công thức nền của giải tích vi phân, khai triển taylor cho phép ta chứng minh được nhiều công thức quan trọng trong đó có công thức khai triển hàm $\sin x, \cos x, $e^{x}, \dots$ và rất nhiều các hàm số khác. Vậy khai triển taylor là gì?

Khai triển taylor là một công thức cho phép chúng ta tính toán giá trị của hàm số thông qua các giá trị đạo hàm bậc cao của nó. Cụ thế nếu hàm $f(x): \mathbb{R} \mapsto \mathbb{R}$. Khi đó khai triển taylor của hàm $f(x)$ có dạng:

$$f(x) = f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + \dots + \frac{f^{(n)}(x_0)}{n!} (x-x_0)^n + o((x-x_0)^{n})$$

Thành phần cuối cùng là hàm $o((x-x_0)^{n})$ là một hàm $o$ nhỏ của $(x-x_0)^n$. Giá trị của hàm này có tỷ số so với $(x-x_0)^{n}$ là bị chặn. 
Tức là với mọi giá trị $\epsilon > 0$ tuỳ ý (có thể rất nhỏ), luôn tồn tại một giá trị $N$ sao cho: 

$$|o((x-x_0)^n)| < \epsilon (x-x_0)^n, \forall x > N$$ 

Hay nói cách khác: $\lim_{x \rightarrow x_0} \frac{o((x-x_0)^n)}{(x-x_0)^n} = 0$. Viết một cách ngắn gọn thì khai triển taylor có dạng:

$$f(x) = \sum_{i=0}^{+\infty} \frac{f^{(i)}(x_0)}{i!}(x-x_0)^i$$

Ngoài ra ta từ khai triển taylor cũng cho phép ta chứng minh được công thức đạo hàm ở $(1)$. Bởi các nếu chỉ khai triển đến bậc nhất ta có:

$$f(x) = f(x_0) + f'(x_0)(x-x_0) + o(x-x_0)$$

$$\begin{eqnarray}\lim_{x \rightarrow x_0} f'(x_0) & = & \lim_{x \rightarrow x_0} \frac{f(x)-f(x_0)}{x-x_0} + \lim_{x \rightarrow x_0} \frac{o(x-x_0)}{x-x_0} \\
& = & \lim_{x \rightarrow x_0} \frac{f(x)-f(x_0)}{x-x_0}
\end{eqnarray}$$

Dòng thứ 2 là vì $\lim_{x \rightarrow x_0} \frac{o(x-x_0)}{x-x_0}=0$

## 2.4. Đạo hàm riêng

Đối với một hàm số có đầu vào nhiều thì đạo hàm riêng sẽ được tính dựa trên từng chiều của hàm số đó. Chẳng hạn hàm $f(x_1, \dots, x_n)$ sẽ có đạo hàm riêng theo $x_i$ như sau:

$$\frac{d f(x_1, x_2 \dots, x_i, \dots, x_n)}{d x_i} = \lim_{\Delta \rightarrow 0} \frac{f(x_1, x_2 \dots, x_i+\Delta, \dots, x_n)- f(x_1, x_2 \dots, x_i, \dots, x_n)}{\Delta}$$

Ta coi như các chiều còn lại khác $x_i$ là hằng số và đạo hàm theo chỉ $x_i$.

Công thức đạo hàm riêng sẽ được sử dụng để tính gradient descent.

## 2.5. gradient descent

Gradient descent là đạo hàm của một hàm số theo một véc tơ. Đây là một công thức thường xuyên được sử dụng trong huấn luyện và cập nhật hệ số của mạng nơ ron. Bởi chúng ta hình dung mạng neural network sẽ tìm cách tối ưu hàm loss function theo các véc tơ hệ số ở từng layer. Do đó chúng ta cần tính gradient descent của hàm loss function theo véc tơ hệ số.

$$\nabla_{\mathbf{w}} f(\mathbf{w}) = [\frac{d f(\mathbf{w})}{d w_1}, \frac{d f(\mathbf{w})}{d w_2}, \dots, \frac{d f(\mathbf{w})}{d w_n} ]^{\intercal}
$$

Lưu ý đây là véc tơ cột vì có dấu chuyển vị. 

Trong véc tơ gradient thì mỗi thành phần $\frac{d f(w_i)}{d w_i}$ là đạo hàm riêng của hàm $f(\mathbf{w})$ theo chiều $w_i$

## 2.3. Công thức đạo hàm vector-value

Có những véc tơ mà mỗi phần tử của nó là một hàm đối với biến $x$. Ví dụ $\mathbf{f}(x) = [f_1(x), f_2(x), \dots, f_n(x)]$. Khi đó đạo hàm của véc tơ cũng chính bằng đạo hàm của từng thành phần theo biến $x$.

$$\frac{d \mathbf{f}(x)}{d x} = [\frac{d f_1(x)}{dx}, \dots ,\frac{d f_n(x)}{dx}]$$

## 2.6. Đạo hàm vector-vector

Đạo hàm của một véc tơ hàm số $\mathbf{f(x)} = [f_1(x), \dots, f_n(x)]$ theo vector $\mathbf{x} = [x_1, \dots, x_m]$ chính là ma trận Jacobian.

$$\begin{eqnarray}
\nabla \mathbf{f(x)} &\triangleq &
\left[
\begin{matrix}
    \frac{\partial f_1(\mathbf{x})}{\partial x_1} & \frac{\partial f_2(\mathbf{x})}{\partial x_1} & \dots & \frac{\partial f_n(\mathbf{x})}{\partial x_1} \\ 
    \frac{\partial f_1(\mathbf{x})}{\partial x_2} & \frac{\partial f_2(\mathbf{x})}{\partial x_2} & \dots & \frac{\partial f_n(\mathbf{x})}{\partial x_2} \\ 
    \vdots & \vdots & \ddots & \vdots \\ 
    \frac{\partial f_1(\mathbf{x})}{\partial x_m} & \frac{\partial f_2(\mathbf{x})}{\partial x_m} & \dots & \frac{\partial f_n(\mathbf{x})}{\partial x_m}
\end{matrix}
\right] \\
& = & 
\left[
\begin{matrix}
    \nabla f_1(\mathbf{x}) & \nabla f_2(\mathbf{x}) & \dots & \nabla f_n(\mathbf{x})
\end{matrix} \right]
\end{eqnarray}$$

Ở công thức dòng thứ hai thì $\nabla f_i(\mathbf{x})$ chính là một véc tơ cột và là gradient descent của hàm số $f_i$ theo véc tơ $\mathbf{x}$.

## 2.7. Lan truyền thuận (_feed forward_)

Mạng nơ ron network có kiến trúc gồm nhiều layer, mỗi layer bao gồm các unit node có tác dụng tính toán ra một chiều của dữ liệu tại layer đó.

![](https://imgur.com/m19Vtoi.png)

**Hình 1**: Kiến trúc mạng nơ ron với hai layers. $x_1, x_2, x_3$ là các biến đầu vào, $\mathbf{W}$ là ma trận hệ số có $\mathbf{w}_{ij}$ là kết nối từ input thứ $j$ tới output là node thứ $j$. $b_2$ đại diện cho hệ số của giá trị mở rộng $1$. $f(h_i)$ chính là hàm số activation để tạo ra tính phi tuyến và tạo ra $u_i$. Những giá trị $u_i$ này được dùng để tính $s$.

Lan truyền thuận là quá trình từ input tính ra phân phối xác suất ở output. Ta có thể tóm tắt các bước tính lan này theo tuần tự công thức:

1. Xuất phát từ input : $x$
2. Tính layer 1: $z = \mathbf{W}x + \mathbf{b}$
3. Tính activation: $\mathbf{h} = f(z)$
4. Tính output: $s = \mathbf{u}^{\intercal}\mathbf{h}$

Các bước tính toán này có thể được tóm lược trong một sơ đồ.

![](https://imgur.com/98Lq2Xc.png)

Xuất phát từ $x$, các node $\odot$ là đại diện cho phép nhân và $\oplus$ đại diện cho phép cộng. Chiều của mũi tên thể hiện thứ tự tính toán.

## 2.8. Chain rule trong lan truyền ngược _backpropagation_

Quá trình backpropagation sẽ thực hiện tính toán gradient descent và cập nhật hệ số. Bước này nói một cách hoa mỹ có thể coi như _linh hồn _ của quá trình huấn luyện. Công thức lan truyền ngược backpropagation sẽ thực hiện hai việc:

* Tính toán gradient descent tại mỗi layers theo chiều từ cuối trở về đầu tiên. Vậy nên công thức này mới có tên gọi là backpropagation.

* Cập nhật gradient descent tại mỗi layer theo giá trị được tính ở trên.

![](https://imgur.com/LWWkkPA.png)

Chain rule trong backpropagation.

Công thức chain rule sẽ được ứng dụng rất nhiều trong backpropagation. Chẳng hạn như trong sơ đồ trên, các node và mũi tên màu xanh dương sẽ đi ngược chiều với mũi tên màu đen là đại diện cho backpropagation.

Giả sử ta xét qúa trình backpropagation tại một node cụ thể $\mathbf{h} = f(z)$.

![](https://imgur.com/5b5GQuK.png)


Để tính đạo hàm $\frac{\delta s}{\delta z}$ ta sẽ không tính được trực tiếp mà phải thông qua chain rule:
$$\frac{\delta s}{\delta z} = \frac{\delta s}{\delta \mathbf{h}} \frac{\delta \mathbf{h}}{\delta z}$$

Những gradient được thực hiện tại node đó gọi là _local gradient_, gradient ở phía liền sau node là _downstream gradient_ và phía trước node là _upstream gradient_. Công thức trên có thể gói gọn lại thành 

$$\text{downstream} = \text{upstream} \times \text{local}$$

## 2.9. Một số công thức đạo hàm đáng nhớ

$\mathbf{A}$ là một ma trận và $\mathbf{x}$ là một véc tơ. Khi đó

* Đạo hàm của tích ma trận với véc tơ:

$$\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$$

* Đạo hàm của tích véc tơ với ma trận:

$$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} = \mathbf{A}$$

* Đạo hàm của tích hai véc tơ

$$\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$$

* Đạo hàm của tích véc tơ nhân với ma trận và nhân với véc tơ đó nhưng chuyển vị.

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

https://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture04-neuralnets.pdf

https://d2l.aivivn.com/chapter_preliminaries/calculus_vn.html#dao-ham-rieng

https://en.wikipedia.org/wiki/Matrix_calculus

https://en.wikipedia.org/wiki/Derivative

https://machinelearningcoban.com/math/