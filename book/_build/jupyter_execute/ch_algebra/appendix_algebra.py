# 1. Đại số tuyến tính

## 1.1. Số vô hướng (scalar)

Trong cuộc sống hàng ngày chúng ta sẽ gặp rất nhiều số vô hướng (scalar). Giá trị của tiền nhà tháng này mà bạn phải trả cho chủ nhà là một số vô hướng. Bạn vừa thực hiện một bài kiểm tra toán, bạn được 9 điểm thì điểm số này là một số vô hướng,.... Tóm lại số vô hướng là một con số cụ thể. Số vô hướng sẽ khác với biến số vì biến số có thể nhận nhiều giá trị trong khi số vô hướng chỉ nhận một giá trị duy nhất. Ví dụ, khi biểu diễn giá nhà $y$ theo diện tích $x$ theo phương trình $y=20x + 200$ thì các số  vô hướng là $20, 200$ và các biến là $x, y$.

Để khởi tạo một số vô hướng, chúng ta sẽ sử dụng tensor

import torch

a = torch.tensor(20)
b = torch.tensor(200)
print("diện tích x = 50 --> giá nhà y = a*50+b = ", a*50+b)

Số vô hướng có thể được coi như hằng số trong một phương trình. Chúng ta có thể thực hiện các phép toán cộng/trừ/nhân/chia với số vô hướng như với hằng số.

## 1.2. Véc tơ

Véc tơ là một khái niệm cơ bản nhất của toán học. Chúng ta có thể coi véc tơ là một tập hợp nhiều giá trị của số vô hướng. Véc tơ thường biểu diễn một đại lượng cụ thể trên thực tiễn. Ví dụ như diện tích của các căn nhà là một véc tơ, số lượng phòng ngủ cũng là một véc tơ. Véc tơ có độ dài đặc trưng chính bằng số lượng các phần tử trong nó. Để khởi tạo một véc tơ, trong pytorch chúng ta bao các giá trị của nó trong dấu ngoặc vuông.

x = torch.tensor([1, 1.2, 1.5, 1.8, 2])
x

### 1.2.1. Các thuộc tính của véc tơ

Một véc tơ sẽ có độ dài và định dạng dữ liệu xác định. Ngoài ra nếu coi một biến số là một véc tơ thì trong thống kê mô tả chúng ta sẽ quan tâm tới tổng, trung bình, phương sai, giá trị lớn nhất, nhỏ nhất.

# Độ dài
print("length of vector: ", x.size()) # or len(x)

# Định dạng của véc tơ
print("vector type: ", x.dtype)

# Tổng của các phần tử 
print("sum of vector: ", x.sum())

# Trung bình các phần tử
print("mean of vector: ", x.mean())

# Giá trị nhỏ nhất
print("min of vector: ", x.min())

# Giá trị lớn nhất
print("max of vector: ", x.max())

### 1.2.2. Các phép tính trên véc tơ

Chúng ta có thể thực hiện các phép tính trên véc tơ như phép cộng, trừ, tích vô hướng, tích có hướng giữa hai véc tơ.. Lưu ý là chúng phải có cùng độ dài. Trong khuôn khổ cuốn sách này, các véc tơ sẽ được ký hiệu là một ký tự chữ thường in đậm như $\mathbf{x}, \mathbf{y}, \mathbf{z}$. Ngoài ra $\mathbf{x}\in \mathbb{R}^{n}$ là véc tơ số thực có độ dài $n$.

x = torch.tensor([1, 2, 1.5, 1.8, 1.9])
y = torch.tensor([1.1, 2.2, 1.2, 1.6, 1.7])
print("x + y: ", x + y)
print("x - y: ", x - y)
print("x * y: ", x * y)

Véc tơ có thể thực hiện các phép cộng, trừ, nhân, chia với một số vô hướng. Giá trị thu được là một véc tơ cùng kích thước mà mỗi phần tử của nó là kết quả được thực hiện trên từng phần tử của véc tơ với số vô hướng đó.

x = torch.tensor([1, 2, 1.5, 1.8, 1.9])
print("x + 5: ", x + 5)
print("x - 5: ", x - 5)
print("x * 5: ", x * 5)

## 1.3. Ma trận

Véc tơ là đại lượng một chiều nên nó chỉ có thể biểu diễn cho một biến. Trong trường hợp chúng ta cần biểu diễn cho nhiều biến thì sẽ cần tới đại lượng hai chiều là ma trận. Ma trận được ký hiệu bởi một chữ cái in đậm $\mathbf{A} \in \mathbb{R}^{m\times n}$ là một ma trận số thực có $m$ dòng và $n$ cột.

$$\begin{split}\mathbf{A}=\begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\ 
a_{21} & a_{22} & \cdots & a_{2n} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
a_{m1} & a_{m2} & \cdots & a_{mn} \\ 
\end{bmatrix}.\end{split}$$

Để xác định một phần tử bất kỳ thuộc dòng thứ $i$, cột thứ $j$ của ma trận $\mathbf{A}$ ta ký hiệu chúng là $\mathbf{A}_{ij}$. Véc tơ dòng thứ $i$ sẽ là $\mathbf{A}_{i:}$ và véc tơ cột thứ $j$ sẽ là $\mathbf{A}_{:j}$. Để đơn giản hoá ta qui ước $\mathbf{A}_{j}$ là véc tơ cột $j$ và $\mathbf{A}^{(i)}$ là véc tơ dòng $i$.

### 1.3.1. Các ma trận đặc biệt

* Ma trận vuông: Ma trận vuông là ma trận có số dòng bằng số cột. Ma trận vuông rất quan trọng vì khi tìm nghiệm cho hệ phương trình, từ ma trận vuông ta có thể chuyển sang ma trận tam giác. Ma trận vuông cũng là ma trận có thể tính được giá trị định thức. Tóm lại ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$ vuông nếu $m=n$. 

* Ma trận đơn vị: Là ma trận có đường chéo chính bằng 1, các phần tử còn lại bằng $0$. Ví dụ ma trận đơn vị kích thước $3 \times 3$ được ký hiệu là $\mathbf{I}_3$ có gía trị là:

$$\begin{split}\mathbf{I}_3=
\begin{bmatrix} 
1 & 0 & 0 \\ 
0 & 1 & 0 \\ 
0 & 0 & 1 
\end{bmatrix}
\end{split}$$

Tóm lại, $\mathbf{A}$ là ma trận đơn vị nếu nó là ma trận vuông và $a_{ij} = 1$ nếu $i=j$ và $a_{ij} = 0$ nếu $i \neq j$.

* Ma trận đường chéo: Là ma trận có các phần tử trên đường chéo chính khác 0 và các phần tử còn lại bằng 0. Ví dụ về ma trận đường chéo:


$$\begin{split}\mathbf{A}=
\begin{bmatrix} 
1 & 0 & 0 \\ 
0 & 2 & 0 \\ 
0 & 0 & 3 
\end{bmatrix}
\end{split}$$


* Ma trận chuyển vị: $\mathbf{B}$ là ma trận chuyển vị của $\mathbf{A}$ nếu $b_{ij} = a_{ji}$ với mọi $i, j$. Dễ hiểu hơn, tức là mọi dòng của ma trận $A$ sẽ là cột của ma trận $\mathbf{B}$. Ví dụ:

$$\begin{split}\mathbf{A}=
\begin{bmatrix} 
1 & 2 & 3 \\ 
3 & 2 & 1
\end{bmatrix}
\end{split}, \begin{split}\mathbf{B}=
\begin{bmatrix} 
1 & 3 \\
2 & 2 \\ 
3 & 1\end{bmatrix}
\end{split}$$


Ký hiệu chuyển vị của ma trận $\mathbf{A}$ là $\mathbf{A}^{\intercal}$

### 1.3.2. Các thuộc tính của ma trận

Một ma trận được đặc trưng bởi dòng và cột.

import torch
A = torch.tensor([[1, 2, 3], 
                  [3, 2, 1]])

# shape của matrix A
A.size()

### 1.3.3. Các phép tính trên ma trận

Hai ma trận có cùng kích thước chúng ta có thể thực hiện các phép cộng, trừ, tích hadamard (hoặc elementi-wise). Ma trận thu được cũng có cùng kích thước và các phần tử của nó được tính dựa trên các phần tử có cùng vị trí trên cả hai ma trận $\mathbf{A}$ và $\mathbf{B}$.

**Tích hadamard hoặc element-wise**

$$
\begin{split}\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}\end{split}
$$

Trên pytorch chúng ta có thể tính tích hadamard của hai ma trận đơn giản như sau:

import torch
A = torch.tensor([[1, 2, 3], 
                  [3, 2, 1]])

B = torch.tensor([[2, 1, 2], 
                  [1, 3, 0]])

A*B

Tương tự với các phép cộng và trừ

print("A-B: \n", A-B)
print("A+B: \n", A+B)

**Tích thông thường**: Tích thông thường giữa hai ma trận $\mathbf{A}$ có kích thước $m \times n$ và $\mathbf{B}$ có kích thước $n \times p$ là một ma trận có kích thước $m \times p$. Ma trận output $\mathbf{C}$ có giá trị tại phần tử $c_{ij} = \mathbf{A}^{(i)} \mathbf{B}_{j}$ (dòng thứ $i$ của ma trận $\mathbf{A}$ nhân với cột thứ $j$ của ma trận $\mathbf{B}$).

$$
\begin{split}\mathbf{A}_{m \times n} \mathbf{B}_{n \times p} =
\begin{bmatrix}
    \mathbf{A}^{(1)}  \mathbf{B}_{1} & \mathbf{A}^{(1)}  \mathbf{B}_{2} & \dots  & \mathbf{A}^{(1)}  \mathbf{B}_{p} \\
    \mathbf{A}^{(2)}  \mathbf{B}_{1} & \mathbf{A}^{(2)}  \mathbf{B}_{2} & \dots  & \mathbf{A}^{(2)}  \mathbf{B}_{p} \\
    \vdots & \vdots & \ddots & \vdots \\
    \mathbf{A}^{(m)}  \mathbf{B}_{1} & \mathbf{A}^{(m)}  \mathbf{B}_{2} & \dots  & \mathbf{A}^{(m)}  \mathbf{B}_{p} \\
\end{bmatrix}\end{split}_{m \times p}
$$

Chắc các bạn còn nhớ $\mathbf{A}^{(i)}$ là véc tơ dòng và $\mathbf{A}_{j}$ là véc tơ cột.

import torch
A = torch.tensor([[1, 2, 3], 
                  [3, 2, 1]])

B = torch.tensor([[2, 1], 
                  [1, 3],
                  [1, 1]])

A@B

### 1.3.4. Truy cập thành phần

Chúng ta có thể truy cập vào các thành phần của ma trận $\mathbf{A}$ dựa theo các chỉ số slice index. Chúng ta có thể tổng hợp kiến thức về truy cập thành phần trong bản sau:

| Cú pháp      | Mô tả |
| ----------- | ----------- |
| :n      | lấy n index đầu tiên từ [0, 1, ..., n-1]       |
| -n:   | lấy n index cuối cùng từ [len-n, ..., len-1]        |
| i:j   | lấy các index từ [i, i+1, ..., j-1]        |
| ::2   | lấy các index chẵn liên tiếp [0, 2, 4 ..., int(len/2)*2]        |
| ::k   | lấy các index cách đều và chia hết cho k một cách liên tiếp [0, k, 2k, ..., int(len/k)*k]        |
| :   | lấy toàn bộ index        |

import torch
A = torch.tensor([[1, 2, 3], 
                  [3, 2, 1],
                  [4, 2, 2]])

# Truy cập ma trận con từ 2 dòng đầu tiên và 2 cột đầu tiên.
A[:2, :2]

# Truy cập ma trận con từ 2 dòng cuối cùng và 2 cột đầu tiên
A[-2:, :2]

# Truy cập véc tơ con từ dòng thứ 2 và 2 cột cuối cùng.
print(A[2, -2:])

# Hoặc
A[2:3, -2:][0]

# Truy cập ma trận có các dòng chẵn
A[::2, :]

# Truy cập một index cụ thể ví dụ dòng 0, 2 của ma trận
A.index_select(0, torch.tensor([0, 2]))
# Trong công thức trên 0 là chiều mà ta sẽ lấy, tensor([0, 2]) là các index ta sẽ lấy từ chiều 0.

## 1.4. Tensor

Tensor là một định dạng đặc biệt được nghĩ ra bởi google. Nó tổng quát hơn so với ma trận vì có thể biểu diễn được các không gian với số chiều tuỳ ý. Chẳng hạn trong xử lý ảnh chúng ta có một bức ảnh với kích thước là $W \times H \times C$ lần lượt $W, H, C$ là chiều _width, height và channels_. Thông thường $C = 1$ hoặc $3$ tuỳ theo ảnh là ảnh xám hay ảnh màu. Trong huấn luyện mô hình phân loại ảnh thì các đầu vào được kết hợp theo mini-batch nên sẽ có thêm một chiều về batch_size. Do đó input có kích thước là $N \times W \times H \times C$.

### 1.4.1. Các thuộc tính của tensor

Một tensor được đặc trưng bởi kích thước các chiều, số lượng chiều, định dạng dữ liệu của tensor.

A = torch.tensor([[[1, 2, 3], 
                  [3, 2, 1]],
                  [[2, 1, 2], 
                  [1, 3, 0]]])

# Kích thước của tensor
print("shape of A: " , A.size())

# Số chiều 
print("total dim: ", A.ndim)

# Định dạng dữ liệu
print("dtype: ", A.dtype)

### 1.4.2. Các phép tính trên tensor

**Tích thông thường giữa 2 tensors**: Nếu tensor $\mathbf{A}$ có kích thước $m \times n \times p$ và tensor $\mathbf{B}$ có kích thước $n \times p \times q$ thì tích giữa chúng có kích thước là $m \times n \times q$. Trên python chúng ta sử dụng ký hiệu `@` để đại diện cho tích thông thường.

import torch

A = torch.randn([2, 3, 4])
B = torch.randn([2, 4, 2])

# Tích giữa 2 tensor
(A@B).size()

Ngoài ra chúng ta có thể tính **tích hadamard giữa 2 tensors** $\mathbf{A}$ và $\mathbf{B}$ được ký hiệu bằng dấu `*` như sau:

import torch

A = torch.randn([2, 3, 4])
B = torch.randn([2, 3, 4])

# Tích hadamard giữa 2 tensor
(A*B).size()

Chúng ta cũng có thể thực hiện các phép cộng, trừ giữa các tensor cùng kích thước.

# Phép cộng
(A+B).size()

# Phép trừ
(A-B).size()

**Truy cập thành phần**: Để truy cập vào một mảng thành phần của $\mathbf{A}$ chúng ta sẽ cần khai báo vị trí indices của chúng trên mỗi chiều của ma trận $\mathbf{A}$. Cách truy cập cũng hoàn toàn tương tự như đối với ma trận.

import torch

# Khởi tạo ma trận A kích thước m, n, p = 2, 3, 4
A = torch.randn([2, 3, 4])

# Truy cập ma trận đầu tiên 
A[:1, :, :]

# Truy cập ma trận đầu tiên và chỉ lấy dòng từ 1 tới 3
A[0][1:3, :]

# Truy cập tương ứng với các chiều m, n, p lần lượt index đầu tiên, 2 index đầu tiên, và index thứ 3.
A[:1, :2, 3]

## 1.5. Tích giữa một ma trận với véc tơ

Bản chất của phép nhân một ma trận với một véc tơ là một **phép biến hình**. Giả sử bạn có ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$ và véc tơ $\mathbf{x} \in \mathbb{R}^{n}$. Khi đó tích giữa ma trận $\mathbf{A}$ với véc tơ $\mathbf{x}$ là một véc tơ $\mathbf{y}$ có kích thước mới là $\mathbf{y} \in \mathbb{R}^{m}$.

$$\mathbf{A}\mathbf{x} =
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix} \mathbf{x} = \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{x} \\
\mathbf{a}^\top_{2} \mathbf{x} \\
\vdots \\
\mathbf{a}^\top_m \mathbf{x} \\
\end{bmatrix} = \begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_m
\end{bmatrix} = \mathbf{y}$$

Như vậy thông qua ma trận $\mathbf{A}$ chúng ta đã biến đổi véc tơ $\mathbf{x}$ từ không gian $n$ chiều sang véc tơ $\mathbf{y}$ trong không gian $m$ chiều. Đây là một định lý rất quan trọng vì bạn sẽ gặp nó thường xuyên trong mạng nơ ron để giảm chiều dữ liệu, trong phân tích suy biến, trong phép xoay ảnh và đặc biệt nhất là trong hồi qui tuyến tính.

Giả sử bạn đã biết được các biến đầu vào gồm: diện tích và số phòng ngủ như các dòng của ma trận bên dưới:

import torch

X = torch.tensor([[100, 120, 80, 90, 105, 95], 
                  [2, 3, 2, 2, 3, 2]])

Và hệ số hồi qui tương ứng với với chúng lần lượt là $\mathbf{w} = (10, 100)$. Khi đó giá nhà có thể được ước lượng bằng tích $\mathbf{y} = \mathbf{X}^{\top}\mathbf{w}$

w = torch.tensor([[10], [100]])
y = X.T@w
y

## 1.6. Tích vô hướng

Tích vô hướng giữa hai véc tơ $\mathbf{x}, \mathbf{y} \in \mathbb{R}^{d}$ có cùng kích thước là một số vô hướng được ký hiệu là $\langle \mathbf{x}, \mathbf{y} \rangle$ hoặc $\mathbf{x}^{\top}\mathbf{y}$ có công thức như sau:

$$\langle \mathbf{x}, \mathbf{y} \rangle = \sum_{i=1}^{d} x_i y_i$$

import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([2, 3, 4])
x.dot(y)

Tích vô hướng rất quan trọng. Đây cũng là lý do mà tôi phải tách chúng thành một mục riêng. Bạn có thể bắt gặp tích vô hướng rất nhiều trong machine learning. Bên dưới là một số tình huống thường gặp:

- Tích vô hướng có thể được sử dụng để tính giá trị ước lượng của phương trình hồi qui tuyến tính. Ví dụ nếu bạn biết giá nhà được biểu diễn theo diện tích $x_1$ và số phòng ngủ $x_2$ theo công thức: 

$$y=20x_1 + 10x_2+ 200$$

Thì một cách khái quát bạn có thể ước lượng $y$ theo tích vô hướng giữa véc tơ đầu vào $\mathbf{x}^{\top} = (x_1, x_2, 1)$ và véc tơ hệ số $\mathbf{w} = (20, 10, 200)$ như sau:
 
$$\hat{y} = \mathbf{x}^{\top}\mathbf{w}$$

- Tích vô hướng cũng được sử dụng để tính trung bình có trọng số của $\mathbf{x}$:

$$\bar{\mathbf{x}} = \sum_{i=1}^{n} x_i q_i= \mathbf{x}^{\top}\mathbf{q}$$
với $\sum_{i=1}^{n} q_i= 1$

- Ngoài ra, chắc hẳn bạn còn nhớ cách tính cos giữa hai véc tơ $\mathbf{x}$ và $\mathbf{y}$ sẽ bằng tích vô hướng giữa norm chuẩn bậc 2 giữa hai véc tơ này.

$$\cos({\mathbf{x}, \mathbf{y}}) = \mathbf{||x||_2}^{\top}\mathbf{||y||_2}
$$

Khái niệm về norm chuẩn bậc 2 cũng là một kiến thức rất quan trọng. Mình sẽ giúp các bạn tìm hiểu bên dưới.


## 1.7. Khái niệm chuẩn

Chuẩn là một khái niệm liên quan đến véc tơ. Hay nói chính xác hơn nó là một độ đo trên véc tơ để so sánh các véc tơ với nhau. Cụ thể hơn: 

$f(\mathbf{x})$ là một phép ánh xạ từ véc tơ sang một đại lượng vô hướng $\mathbb{R}^{d} \mapsto \mathbb{R}$ nếu nó thoả mãn các tính chất.

1. Tính chất co dãn: 

$$\alpha f(\mathbf{x}) = f(\alpha\mathbf{x})$$

Như vậy khi bạn phóng đại lên véc tơ $\alpha$ lần thì giá trị chuẩn của nó cũng phóng đại lên $\alpha$ lần.

2. Bất đẳng thức tam giác: 

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(
  \mathbf{y}
)$$

Nếu ta coi $\mathbf{x}$ như là véc tơ cạnh và $f(\mathbf{x})$ như là độ dài cạnh của một tam giác thì $f(\mathbf{x}), f(\mathbf{y})$ là độ dài của 2 cạnh bất kỳ và tổng của chúng sẽ lớn hơn độ dài cạnh còn lại $f(\mathbf{x} + \mathbf{y})$.

3. Tính chất không âm: 

$$f(\mathbf{x}) \geq 0, \forall \mathbf{x}$$

Tính chất này là hiển nhiên vì đã là độ đo thì không được âm.

Trong machine learning các bạn sẽ thường xuyên gặp một số chuẩn chính là chuẩn bậc 2 

$$L_{2} = \|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n \left|x_i \right|^2 }$$

import torch
x = torch.randn(10)
torch.norm(x, p=2)

Ta nhận thấy hàm MSE đo lường sai số giữa giá trị dự báo và thực tế trong phương trình hồi qui tuyến tính cũng là một dạng chuẩn bậc 2.

Chuẩn bậc 1:

$$L_{1} = \|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right| $$

print(torch.norm(x, p=1))
# hoặc 
torch.abs(x).sum()

Trong hồi qui tuyến tính thì chuẩn bậc 1 đo lường sai số tuyệt đối giữa giá trị dự báo và giá trị thực tế. Tuy nhiên nó ít được sử dụng hơn so với chuẩn bậc 2 như là một loss function vì giá trị của nó có đạo hàm không liên tục. Điều này dẫn tới việc huấn luyện mô hình không ổn định. Tuy nhiên nó cũng khá thường xuyên được sử dụng trong các mô hình deep learning chẳng hạn như GAN.

Cả hai chuẩn trên đều là trường hợp cụ thể của chuẩn bậc $p$ (ký hiệu $L_{p}$) tổng quát hơn có công thức như sau:

$$L_{p} = \|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}$$

Để cả 3 điều kiện về chuẩn được thoả mãn thì chúng ta cần có $p \geq 1$.

# chuẩn p bất kỳ >= 1, chẳng hạn p=1.5
torch.norm(x, p=1.5)

# 2. Tóm tắt

Như vậy qua chương này mình đã hướng dẫn cho các bạn các kiến thức bản nhất trong đại số tuyến tính. Bao gồm:

1. Các khái niệm: Số vô hướng, véc tơ, ma trận, tensor kèm theo thuộc tính của chúng.
2. Các phép tính cộng, trừ, nhân ma trận, nhân véc tơ, nhân ma trận với véc tơ.
3. Khái niệm về chuẩn và ý nghĩa của chúng trong vai trò một độ đo đối với véc tơ.

Đây là những kiến thức nền tảng nhưng rất quan trọng mà bạn đọc cần nắm vững trước khi học sâu về AI.

# 3. Bài tập

Một vài bài tập dưới đây sẽ giúp bạn ôn lại kiến thức tốt hơn:

1. Khởi tạo một số vô hướng, một véc tơ có độ dài là $3$ và một ma trận bất kỳ có kích thước là $2\times 3$ trên pytorch.
2. Tính tích giữa véc tơ và ma trận.
3. Tính tổng các dòng và tổng các cột của ma trận.
4. Chứng minh rằng nếu $\mathbf{A}$ là một ma trận vuông thì $\mathbf{A} + \mathbf{A}^{\top}$ là một ma trận đối xứng.
5. Cho $\mathbf{A}, \mathbf{B}, \mathbf{C}$ là ba ma trận có kích thước lần lượt là $m \times n$, $n \times p$ và $p \times q$ chứng minh rằng $\mathbf{ABC} = (\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$
6. $\mathbf{trace}$ của ma trận là tổng các phần tử nằm trên đường chéo chính ( phần tử mà có index dòng bằng cột). Chứng minh rằng: $\mathbf{trace(AB) = trace(BA)}$
7. Chứng minh: $\mathbf{A} \odot \mathbf{(B+C)} = \mathbf{A} \odot \mathbf{B} + \mathbf{A} \odot \mathbf{C}$
8. Chứng minh: $\mathbf{A} \odot (\mathbf{B} \odot \mathbf{C})= (\mathbf{A} \odot \mathbf{B}) \odot \mathbf{C}$