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

# 7,1. Hàm mất mát của SVM


+++ {"id": "BS8WHsU91Tlv"}

## 7.1.1. Góc nhìn từ hồi qui Logistic

Trong [hồi qui Logistic](https://phamdinhkhanh.github.io/deepai-book/ch_ml/classification.html) chúng ta đã làm quen với _hàm mất mát_ (_loss function_) dạng:

$$\mathcal{L}(\mathbf{w}) = \sum_{i=1}^{n} -[y_i\log(\hat{y_i}) + (1-y_i)\log(1-\hat{y}_i)]$$

Bản chất của hàm mất mát trong hồi qui Logistic là một _thước đo_ về sự tương quan giữa phân phối xác suất dự báo với _ground truth_.

Trong đó phân phối xác suất được ước tính dựa trên hàm `Sigmoid` theo công thức $\hat{y} = \sigma(z) = \frac{1}{1+e^{-z}}$.

Ta cũng biết rằng đường biên phân loại của hồi qui Logistic là một siêu phẳng có phương trình $\mathbf{w}^{\intercal}\mathbf{x}$.

$$
\begin{split}
y = \left\{
\begin{matrix}
1 \text{ if } \mathbf{w}^{\intercal}\mathbf{x} > 0 \\
0 \text{ if } \mathbf{w}^{\intercal}\mathbf{x} \leq 0
\end{matrix}
\right.\end{split}
$$

Tiếp theo chúng ta sẽ cùng phân tích _hàm mất mát_ của mô hình trong hai trường hợp $y=0$ và $y=1$:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 404
id: nLSUaiHyy4j_
outputId: fa20d563-fc97-4eea-d2a4-132e7949fa62
---
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize = (18, 6))
z = np.linspace(-3, 3, 100)

def sigmoid(z):
  return 1/(1+np.exp(-z))

y0 = -np.log(sigmoid(z)) # Trường hợp ground truth = 0
y1 = -np.log(1-sigmoid(z)) # Trường hợp ground truth = 1

# Hàm mất mát nếu ground truth = 0
ax[0].plot(z, y0)
ax[0].set_xlabel('z')
ax[0].set_ylabel('L(y, yhat)')
ax[0].set_title('y=0')

# Hàm mất mát nếu ground truth = 1
ax[1].plot(z, y1)
ax[1].set_xlabel('z')
ax[1].set_ylabel('L(y, yhat)')
ax[1].set_title('y=1')
plt.show()
```

+++ {"id": "0-IBcaF36CUW"}

Ta nhận thấy hình dạng của _hàm mất mát_ trong hai trường hợp tương ứng với $y=1$ và $y=0$ là trái ngược nhau:

* Đối với trường hợp nhãn $y = 0$: Khi giá trị của $z$ càng lớn thì hàm mất mát sẽ tiệm cận 0. Điều đó đồng nghĩa với mô hình sẽ phạt ít những trường hợp $z$ lớn và có nhãn 0. Những trường hợp này tương ứng với những điểm nằm cách xa đường biên phân chia.

* Đối với nhãn $y=1$ thì trái lại, mô hình có xu hướng phạt ít với những giá trị $z$ nhỏ. Khi đó những điểm này sẽ nằm cách xa đường biên về phía nửa mặt phẳng $y=1$.

Những phân tích ở trên là hợp lý vì ở các mức giá trị $z$ đủ lớn hoặc đủ nhỏ thì đều là các điểm nằm cách xa đường biên phân chia nên chúng ta có thể dễ dàng dự báo đúng nhãn cho chúng. Việc phạt những điểm này nếu phân loại sai không mang nhiều ý nghĩa bằng phạt những điểm nằm gần đường biên và được xem như là case khó (_hard case_). Thậm chí nếu phạt những điểm nằm xa đường biên một giá trị lớn dễ khiến xảy ra nguy cơ _quá khớp_ vì hầu hết những điểm đó đều là _ngoại lai_.

+++ {"id": "ybI-DYA11cPp"}

## 7.1.2. Từ Logistic tới SVM
Trong SVM chúng ta có một thay đổi đột phá đó là tìm cách xấp xỉ hàm mất mát dạng cross-entropy của Logistic bằng một hàm mà chỉ phạt những điểm ở gần đường biên thay vì phạt những điểm ở xa đường biên bằng cách đưa mức phạt về 0.

Cụ thể đó là hai hàm phạt $\text{cost}_1()$ và $\text{cost}_2()$ tương ứng với $y=0$ và $y=1$ như bên dưới:


$$\begin{split}
\left\{
\begin{matrix}
\text{cost}_1(z) = \max(0, 1-z) ~ \text{if } y=0 \\
\text{cost}_2(z) = \max(1+z, 0) ~ \text{if } y=1
\end{matrix}
\right.\end{split}$$

Hai hàm này thể hiện chi phí phải bỏ ra nếu phân loại sai các nhãn lần lượt thuộc $0$ hoặc $1$. Dạng tổng quát của chúng là $\max(0, t)$ còn được gọi là hàm hingloss. Đây là một trong những hàm mất mát mà bạn sẽ gặp khá nhiều trong machine learning.

Bên dưới là hình dạng của hai hàm $\text{cost}_1()$ và $\text{cost}_2()$.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 404
id: SuD4dGXE-TOj
outputId: dc386229-9a8e-40f2-8229-2ffee9dc63f2
---
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize = (18, 6))
z = np.linspace(-3, 3, 100)

def sigmoid(z):
  return 1/(1+np.exp(-z))

y0 = -np.log(sigmoid(z)) # Trường hợp ground truth = 0
y1 = -np.log(1-sigmoid(z)) # Trường hợp ground truth = 1

cost1 = np.maximum(0, 1-z)
cost2 = np.maximum(1+z, 0)

# Hàm mất mát nếu ground truth = 0
ax[0].plot(z, y0)
ax[0].plot(z, cost1)
ax[0].set_xlabel('z')
ax[0].set_ylabel('L(y, yhat)')
ax[0].legend(labels = ['cross-entropy', 'cost1'])
ax[0].set_title('y=0')

# Hàm mất mát nếu ground truth = 1
ax[1].plot(z, y1)
ax[1].plot(z, cost2)
ax[1].set_xlabel('z')
ax[1].set_ylabel('L(y, yhat)')
ax[1].legend(labels = ['cross-entropy', 'cost2'])
ax[1].set_title('y=1')

plt.show()
```

+++ {"id": "8RFsQgzDAhNd"}

Ta nhận thấy hình dạng của các hàm mất mát $\text{cost}_1$ và $\text{cost}_2$ cũng gần tương tự như cross-entropy. Điểm khác biệt chính đó là giá trị của mất mát bằng 0 nếu $z \geq 1$ (đối với nhãn $y=0$) hoặc $z \leq -1$ (đối với nhãn $y=1$). Theo các hàm mất mát mới này, chúng ta bỏ qua việc phạt phân loại sai những điểm nằm xa đường biên. Đối với những điểm nằm gần đường biên nhất thì mới ảnh hưởng tới hàm mất mát. Tập hợp những điểm nằm gần đường biên sẽ giúp xác định đường biên và được gọi là tập điểm hỗ trợ (_support vector_).


+++ {"id": "tHU9szvSIE-9"}

Như vậy sau khi thay đổi hàm phạt ta thu được hàm mất mát mới dạng:

$$\mathcal{L}(\mathbf{w}) = \sum_{i=1}^{n} -[y_i\text{cost}_1(\hat{y_i}) + (1-y_i)\text{cost}_2(1-\hat{y}_i)]$$

SVM cho phép ta giảm thiểu _quá khớp_ thông qua một thành phần điều chuẩn cũng tương tự như hồi qui Logistic.

$$\mathcal{L}(\mathbf{w}) = C(\sum_{i=1}^{n} -[y_i\text{cost}_1(\hat{y_i}) + (1-y_i)\text{cost}_2(1-\hat{y}_i)])+\frac{\lambda}{2} \underbrace{||\mathbf{w}||_2^2}_{\text{regularization term}}$$

Trong công thức trên thì hằng số $C > 0$ thể hiện ảnh hưởng của sai số phân loại lên hàm mất mát. Trong khi $\lambda > 0$ là hằng số của thành phần điều chuẩn (_regularization term_) thể hiện tác động của độ lớn trọng số hồi qui $\mathbf{w}$ lên hàm mất mát.

Khi tăng tỷ lệ $\frac{\lambda}{C}$ có thể giúp các trọng số của mô hình được kiểm soát về độ lớn, thông qua đó làm cho độ phức tạp của đường biên phân chia giảm và kiểm soát hiện tượng _quá khớp_.




+++ {"id": "kmtXozt_IC7f"}


Đối với phương trình hồi qui Logistic thì chúng ta sẽ xác định nhãn dựa trên dấu của $\mathbf{w}^{\intercal}\mathbf{x}$. Còn trong thuật toán SVM, đối với một tập dữ liệu mà các nhãn là phân tuyến (_linear seperable_) (tức là tồn tại ít nhất 1 đường biên phân loại đúng toàn bộ các điểm) thì chúng ta sẽ mở rộng đường biên phân chia về hai phía là 1 đơn vị. Khi đó một điểm được dự báo là:


$$\begin{split}
y = \left\{
\begin{matrix}
0 ~ \text{if } \mathbf{w}^{\intercal}\mathbf{x} \leq -1 \\
1 ~ \text{if } \mathbf{w}^{\intercal}\mathbf{x} \geq 1 
\end{matrix}
\right.\end{split}$$

Ý nghĩa của việc mở rộng đường biên đó là khiến cho các điểm nằm gần với đường biên sẽ trở nên tách biệt hơn. Tiếp theo chúng ta sẽ tìm hiểu cơ chế nào hoạt động và cách xác định đường biên đối với thuật toán SVM.

+++ {"id": "YV02sDsMElNt"}

# 7.2. Đường biên và lề trong SVM


**Tập dữ liệu của bài toán SVM**

Giả sử tập dữ liệu huấn luyện $\mathcal{Z}$ bao gồm $N$ điểm dữ liệu. Trong đó điểm dữ liệu thứ $i$ là $Z_i = (\mathbf{x}_i, y_i)$ với $\mathbf{x}_i \in \mathbb{R}^{d}$ là véc tơ đầu vào và $y_i$ là biến mục tiêu là một trong hai giá trị $\{-1, 1\}$ phân tuyến (_linear seperable_).

Bên dưới là hình ảnh tập dữ liệu phân tuyến, đường biên và lề trong thuật toán SVM.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 483
id: 0UMe39dDy1ja
outputId: f67faee0-249b-4db1-ef1d-ced031c8ab61
---
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


# we create 40 separable points
X, y = make_blobs(n_samples=40, centers=[(0, 3), (3, 0)], cluster_std=[0.5, 0.5], random_state=6)
idx_cls_0 = np.where(y == 0)
idx_cls_1 = np.where(y == 1)
# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

plt.figure(figsize = (12, 8))

plt.scatter(X[idx_cls_0, 0], X[idx_cls_0, 1], c='red', marker='o', s=50)
plt.scatter(X[idx_cls_1, 0], X[idx_cls_1, 1], c='blue', marker='*', s=100)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
B = clf.decision_function(xy).reshape(XX.shape)
A = B-0.9
C = B+0.8
# plot decision boundary and margins
ax.contour(XX, YY, B,  colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

ax.contour(XX, YY, A, colors='k', levels=[0], alpha=0.5,
           linestyles=['-'])

ax.contour(XX, YY, C, colors='k', levels=[0], alpha=0.5,
           linestyles=['-'])

# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           marker='o',linewidth=1, facecolors='none', edgecolors='k')

ax.text(0, -0.5, 'B', fontsize=18)
ax.text(-1, -0.5, 'A', fontsize=18)
ax.text(1, -0.5, 'C', fontsize=18)
plt.show()
```

+++ {"id": "5YwnGPU0AZzs"}

**Hình 1:** Hình ảnh về tập dữ liệu trong bài toán phân loại nhị phân mà các lớp là _phân tuyến_. Ba đường thẳng `A, B, C` đại diện cho ba đường biên phân chia đúng **mọi điểm dữ liệu**. Những điểm hình tròn nằm bên trái thuộc mặt dương có nhãn $y=1$, những điểm dấu nhân nhằm bên phải thuộc mặt âm nhãn $y=-1$.

Trên đồ thị chúng ta qui ước nhãn $1$ cho các điểm nằm bên trái mặt phân chia (mặt dương) và nhãn $-1$ cho các điểm nằm ở bên phải mặt phân chia (mặt âm). Sở dĩ chúng ta gán nhãn như vậy là vì tại cùng một dữ liệu đầu vào $\mathbf{x}$ thì mặt dương bên trái sẽ có giá trị lớn hơn mặt âm bên phải.


**Lựa chọn đường biên phân chia tương ứng với một phương**

Ba đường thẳng `A, B, C` ở ví dụ trên là ba đường biên phân chia song song và có cùng phương. Trong ba đường biên phân chia thì đường biên `B` là công bằng nhất vì chúng cách đều các điểm gần nhất thuộc hai lớp. Còn lựa chọn `A` và `C` sẽ không công bằng vì chúng ta sẽ dễ thiên vị một lớp hơn lớp còn lại. 

Như vậy để cho công bằng thì đường biên phải luôn nằm chính giữa và cách đều các điểm gần nhất với nó. Đồng thời đối với bài toán Hard-Margin SVM thì tập dữ liệu là phân tuyến nên đường biên cần phải phân loại đúng mọi điểm dữ liệu. Chúng ta coi độ rộng của đường biên là lề (_margin_). Ngoài ra tập hợp những điểm nằm sát đường biên nhất thì được gọi là tập hỗ trợ. Những điểm này sẽ hỗ trợ tìm ra đường biên vì những đường thẳng nét đứt đi qua chúng song song với đường biên.

Trong không gian hai chiều thì đường biên là một đường thẳng. Trong không gian 3 chiều chúng sẽ là một mặt phẳng (_plane_). Trong không gian nhiều hơn 3 chiều chúng ta gọi đường biên phân chia là siêu phẳng (_hyperplane_).

+++ {"id": "oG0NV7Qz_vrb"}

Một câu hỏi đặt ra đó là có vô số những đường biên phân loại, vậy thì đường biên nào là phù hợp nhất?

Mục tiêu của SVM đó là tìm ra một siêu phẳng (_hyperplane_) trong không gian $d$ chiều làm đường biên phân chia sao cho độ rộng **lề** của chúng là lớn nhất vì khi phân chia theo đường biên này thì các nhóm là tách biệt nhất.

![](https://i.imgur.com/oKeJOcW.jpeg)

Giả sử phương trình của đường biên phân chia hai điểm dữ liệu là:

$$b + w_1 x_1 + w_2 x_2 + \dots + w_N x_N = b + \mathbf{w}^{\intercal}\mathbf{x} = 0$$

$b$ là hệ số tự do, $\mathbf{w}$ là các véc tơ hệ số. $\mathbf{x}$ là véc tơ quan sát đầu vào.

Trong chương trình THPT chúng ta đã được học về công thức khoảng cách từ một điểm $A = (x_1, x_2)$ tới một đường thẳng $l$ có phương trình $w_0 + w_1 x_1 + w_2 x_2 = 0$ là:

$$d(A, l) = \frac{|b + w_1 x_1 + w_2 x_2|}{\sqrt{w_1^2 + w_2^2}} = \frac{|b + w_1 x_1 + w_2 x_2|}{||\mathbf{w}||_2}$$

Trong trường hợp tổng quát, khoảng cách từ một điểm bất kỳ $Z_i = (\mathbf{x}_i, y_i)$ tới biên là siêu phẳng $H$ có phương trình $b+\mathbf{w}^{\intercal}\mathbf{x} = 0$ sẽ là:

$$d(Z_i, H) = \frac{|b+\mathbf{w}^{\intercal}\mathbf{x}_i|}{||\mathbf{w}||_2} = \frac{y_i(b+\mathbf{w}^{\intercal}\mathbf{x}_i)}{||\mathbf{w}||_2}$$

Trong công thức trên thì $|b+\mathbf{w}^{\intercal}\mathbf{x}| = y_i(b+\mathbf{w}^{\intercal}\mathbf{x}_i)$ là vì:

* Xét trường hợp nhãn $y_i=-1$ thì điểm $Z_i$ nằm ở mặt âm và có $b+\mathbf{w}^{\intercal}\mathbf{x}_i \leq 0$. Do đó $y_i(b+\mathbf{w}^{\intercal}\mathbf{x}_i) \geq 0$. 

* Xét trường hợp nhãn $y_i = 1$ thì $Z_i$ nằm ở mặt dương và có $b+\mathbf{w}^{\intercal}\mathbf{x}_i \geq 0$. Từ đó suy ra  $y_i(b+\mathbf{w}^{\intercal}\mathbf{x}_i) \geq 0$. 

Trong cả hai trường hợp thì đẳng thức $|b+\mathbf{w}^{\intercal}\mathbf{x}| = y_i(b+\mathbf{w}^{\intercal}\mathbf{x}_i)$ luôn xảy ra.


+++ {"id": "oBVXFVY4EFZw"}


**Tìm đường biên có lề lớn nhất**

Tập hợp các điểm nằm gần nhất với một đường biên sẽ giúp xác định phương trình đường biên nên chúng còn được gọi là tập hợp các điểm hỗ trợ (_support points_), ký hiệu là $S$. Trong hình vẽ thì các điểm được khoanh tròn chính là các điểm thuộc tập hỗ trợ. Để tìm ra đường biên có độ rộng lề là lớn nhất thì chúng ta cần tối đa hoá khoảng cách từ các điểm thuộc tập hỗ trợ tới đường biên. Điều này tương đương với giải bài toán tối ưu:

$$\begin{eqnarray}
\hat{\mathbf{w}}, \hat{b} & = & \arg \max \{\min_{(\mathbf{x}_i, y_i) \in \mathcal{Z}} \frac{b+y_i(\mathbf{w}^{\intercal}\mathbf{x}_i)}{||\mathbf{w}||_2} \} \
\end{eqnarray} \tag{1}$$

Khi nhân vào phương trình đường biên với một hệ số $k$ thì đường biên không thay đổi. Do đó khoảng cách từ mọi điểm tới đường biên không thay đổi. Tức là khoảng cách từ các điểm thuộc tập hỗ trợ tới đường biên không thay đổi và dẫn tới độ rộng của lề là không thay đổi. Nhờ tính chất này chúng ta có thể nhân thêm vào các trọng số $w_i$ của phương trình đường biên một hệ số $k$ sao cho với các điểm dữ liệu thuộc tập hỗ trợ $S$ thì $b+y_i(\mathbf{w}^{\intercal}\mathbf{x}_i) = 1$. Điều đó cũng đồng nghĩa với luôn tìm được một cách nhân với $k$ sao cho đường biên:

$$\min_{(\mathbf{x}_i, y_i) \in \mathcal{Z}} b+y_i(\mathbf{w}^{\intercal}\mathbf{x}_i) = 1$$

Bài toán $(1)$ trở thành bài toán tối ưu với ràng buộc tuyến tính:

$$\begin{eqnarray}
\hat{\mathbf{w}}, \hat{b} & = & \arg \max \frac{1}{||\mathbf{w}||_2} \\
\text{subject} & : & y_i(b+\mathbf{w}^{\intercal}\mathbf{x}_i) \geq 1, \forall i=\overline{1, N} \tag{2}
\end{eqnarray}$$

Điều kiện ràng buộc $y_i(b+\mathbf{w}^{\intercal}\mathbf{x}_i) \geq 1, \forall i=\overline{1, N}$ là vì khoảng cách từ mọi điểm luôn lớn hơn khoảng cách từ điểm hỗ trợ tới đường biên phân chia và khoảng cách này bằng 1 vì theo giả định ta đã nhân với hệ số $k$ vào phương trình đường biên.

Để đơn giản hoá thì bài toán tối ưu $(2)$ có thể nghịch đảo hàm mục tiêu để chuyển sang dạng tương đương:


$$\begin{eqnarray}
\hat{\mathbf{w}}, \hat{b} & = & \arg \min ||\mathbf{w}||_2 \\
\text{subject} & : & y_i(b+\mathbf{w}^{\intercal}\mathbf{x}_i) \geq 1, \forall i=\overline{1, N} \tag{3}
\end{eqnarray}$$

Bài toán tối ưu $(3)$ là một bài toán dạng [Quadratic Form](https://en.wikipedia.org/wiki/Quadratic_form) nên chúng ta có thể dễ dàng tìm được lời giải của chúng thông qua hệ [điều kiện KKT](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions). Để giải bài toán tối ưu này có thể sử dụng package [cvxopt](https://pypi.org/project/cvxopt/) trong python. Đây là một package chuyên biệt giúp giải quyết các bài toán tối ưu lồi. Trong khuôn khổ của cuốn sách này, với mục tiêu đơn giản hoá mọi thứ, chúng ta sẽ không đi sâu vào cách giải hệ điều kiện KKT.

+++ {"id": "fsZKxEApHU-L"}

# 7.3. Sorf Margin Classification

## 7.3.1. So sánh giữa lề cứng (_hard margin_) và lề mềm (_soft margin_)
Đường biên phân chia của thuật toán SVM sẽ chịu ảnh hưởng bởi những điểm thuộc tập hỗ trợ $S$. Trong trường hợp đường biên phân chia **đúng mọi điểm điểm dữ liệu** thì được gọi là bài toán phân loại theo đường biên cứng (_hard margin classification_). Tuy nhiên đường biên cứng tỏ ra hạn chế nếu tồn tại dữ liệu .ngoại lai (_outlier_). Chúng ta cùng phân tích hạn chế này ở hình minh hoạ bên dưới.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 390
id: 9xB3DRLLmB0c
outputId: b9e99c9a-ed17-4c32-8a84-c295ddf55dac
---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import svm

# we create 40 separable points
X, y = make_blobs(n_samples=40, centers=[(0, 3), (3, 0)], cluster_std=[0.5, 0.5], random_state=6)
idx_cls_0 = np.where(y == 0)
idx_cls_1 = np.where(y == 1)
id_max = np.argmax(X, axis=0)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# fit the model
for i in range(2):
    # Adjust outlier
    if i == 1:
      X[id_max[0]] = [1, 2] 

    clf = svm.SVC(kernel='linear', C=100)
    clf.fit(X, y)

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin
    # plot the line, the points, and the nearest vectors to the plane
    ax[i].plot(xx, yy, 'k-')
    ax[i].plot(xx, yy_down, 'k--')
    ax[i].plot(xx, yy_up, 'k--')

    ax[i].scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k',
                cmap=cm.get_cmap('RdBu'))
    

    ax[i].scatter(X[idx_cls_0, 0], X[idx_cls_0, 1], c='red', marker='o', s=50)
    ax[i].scatter(X[idx_cls_1, 0], X[idx_cls_1, 1], c='blue', marker='*', s=100)

    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    x_min = -1.5
    x_max = 4
    y_min = -1
    y_max = 4

    # Put the result into a contour plot
    ax[i].contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

    ax[i].set_xlim(x_min, x_max)
    ax[i].set_ylim(y_min, y_max)
    if i == 0:
      ax[i].set_title('Hard Margin SVM')
    else:
      ax[i].set_title('Hard Margin SVM with Outlier')

plt.show()
```

+++ {"id": "zGv4ZHI9mPMI"}


<!-- ![](https://imgur.com/8B67kPe.png) -->
**Hình 2:** Hình bên trái là _phân loại đường biên cứng_ (_Hard margin SVM_) đối với tập dữ liệu thông thường. Hình bên phải là _phân loại đường biên cứng_ đối với dữ liệu chứa điểm ngoại lai (là điểm hình sao được khoanh tròn nằm bên trái). Phương pháp _phân loại đường biên cứng_ buộc phải phân loại đúng mọi điểm dữ liệu, bao gồm cả điểm ngoại lai. Điều này khiến cho đường biên phân chia bị thu hẹp lại. Khi đó qui luật phân chia sẽ không còn giữ được yếu tố tổng quát và dẫn tới hiện tượng quá khớp (_overfitting_). Kết quả dự báo trên tập _kiểm tra_ khi đó sẽ kém hơn so với tập _huấn luyện_.

Để khắc phục hạn chế của _phân loại đường biên cứng_, kỹ thuật _phân loại đường biên mềm_ (_Sorf Margin Classification_) chấp nhận đánh đổi để mở rộng lề và cho phép phân loại sai các điểm ngoại lai. Cụ thể hơn, thuật toán sẽ chấp nhận một số điểm bị rơi vào vùng của lề (vùng nằm giữa hai đường nét đứt, vùng này còn được gọi là vùng không an toàn) nhưng trái lại, chi phí cơ hội của sự đánh đổi đó là độ rộng lề lớn hơn. Đường biên phân chia được tạo ra từ kỹ thuật này thường nắm được tính _tổng quát_ và hạn chế hiện tượng _quá khớp_.


```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 390
id: aF3C0v2rmaPv
outputId: bf09f606-a643-4b62-a0d6-2a9c17742c15
---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import svm

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# fit the modelf
for i, (name, penalty) in enumerate([('hard margin', 1), ('soft margin', 0.05)]):
    clf = svm.SVC(kernel='linear', C=penalty)
    clf.fit(X, y)

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
    ax[i].plot(xx, yy, 'k-')
    ax[i].plot(xx, yy_down, 'k--')
    ax[i].plot(xx, yy_up, 'k--')

    ax[i].scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k',
                cmap=cm.get_cmap('RdBu'))
    

    ax[i].scatter(X[idx_cls_0, 0], X[idx_cls_0, 1], c='red', marker='o', s=50)
    ax[i].scatter(X[idx_cls_1, 0], X[idx_cls_1, 1], c='blue', marker='*', s=100)

    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    x_min = -1.5
    x_max = 4
    y_min = -1
    y_max = 4

    # Put the result into a contour plot
    ax[i].contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

    ax[i].set_xlim(x_min, x_max)
    ax[i].set_ylim(y_min, y_max)
    if i == 0:
      ax[i].set_title('Hard Margin SVM, C='.format(penalty))
    else:
      ax[i].set_title('Soft Margin SVM, C='.format(penalty))

plt.show()
```

+++ {"id": "BfuzyqCkDiAK"}

**Hình 3:** Phân loại biên cứng (bên trái) và phân loại biên mềm (bên phải) trong SVM. Chúng ta nhận thấy đối với đường biên mềm thì SVM chấp nhận một số điểm rơi vào vùng an toàn để nhằm tạo ra một đường biên phân chia tổng quát hơn. Trong khi phân loại theo đường biên cứng thì không chấp nhận những điểm dữ liệu bị lấn sang phía bên kia của vùng an toàn (là đường nét đứt).

+++ {"id": "0P-aYvMEmZIY"}

## 7.3.2. Suy xét lại hàm chi phí cho phân loại đường biên mềm SVM

Ý tưởng của _phân loại đường biên mềm_ là mở rộng lề. Nhưng chúng ta không thể mở rộng lề ra vô cùng vì như vậy mọi điểm đều nằm trong đường biên phân chia và đường biên phân chia là vô nghĩa. Quá trình mở rộng lề sẽ bị kìm hãn ở một mức độ nhất định sao cho nếu các điểm bị lấn vào đường biên thì không được lấn quá nhiều. Tức là đối với những điểm bị rơi vào _vùng không an toàn_ thì tổng khoảng cách của chúng tới mép của lề (là các đường nét đứt) là nhỏ nhất. Khoảng cách từ một điểm tới mép đường biên (nét đứt) khi nó bị lấn lề là:

$$d(Z_i, H) = \xi_i = |b+\mathbf{w}^{\intercal}\mathbf{x}_i-y_i|$$

**Bài tập:** Chứng minh công thức khoảng cách trên khá đơn giản, xin dành cho bạn đọc.

$xi_i$ chính là giá trị tối đa mà chúng ta cho phép để một điểm bị lần sang phần bên kia của lề. Trong hàm mất mát chúng ta cần tối thiểu hoá thêm tổng khoảng cách của những phần bị lấn này bằng cách cộng thêm chúng vào hàm mất mát:

$$\begin{eqnarray}
\hat{\mathbf{w}}, \hat{b} & = & \arg \min ~[~||\mathbf{w}||_2 + C \sum_{Z_j \in \mathcal{M}} |b+\mathbf{w}^{\intercal}\mathbf{x}_i-y_i|~] \\
& = & \arg \min ~[~||\mathbf{w}||_2 + C \sum_{Z_j \in \mathcal{M}} \xi_i~]\\
\text{subject} & : & y_i(b+\mathbf{w}^{\intercal}\mathbf{x}_i) \geq 1 - \xi_i, \xi_i \geq 0 ~ \forall i=\overline{1, N} \tag{4}
\end{eqnarray}$$

Với $\mathcal{M}$ là tập hợp các điểm bị lấn lề.

* Khi toàn bộ các giá trị $\xi_i = 0$ đồng nghĩa với việc chúng ta không chấp nhận việc lấn lề là xảy ra và đường biên mềm trở thành đường biên cứng.

* Nếu một điểm có $0 \geq \xi_i \geq 1$ thì chúng ta cho phép một điểm rơi vào vùng không an toàn nhưng không được rời xa quá đường biên. Tức là điểm đó vẫn được phân loại đúng nhưng bị lấn vào vùng lề.

* Nếu một điểm có $\xi_i > 1$ thì điểm đó sẽ bị lấn vượt quá đường biên và bị phân loại sai.

Hệ số $C$ là một hệ số rất quan trọng thể hiện tỷ lệ đánh đối giữa độ rộng lề và sự vi phạm bằng cách xâm lấn vào lề. Một hệ số $C$ lớn sẽ cho thấy đóng góp vào hàm mất mát của một điểm vi phạm sẽ lớn hơn việc mở rộng lề. Do đó để hàm mất mát nhỏ thì chúng ta cần hạn chế các điểm vi phạm và chấp nhận một độ rộng lề nhỏ hơn.

Trái lại trường hợp $C$ nhỏ thường trả lại một độ rộng của lề lớn hơn và đồng thời mức độ xâm lấn là nhỏ hơn.

Khi tiến hành tinh chỉnh mô hình, chúng ta quan tâm nhiều tới hệ số $C$ vì nó ảnh hưởng trực tiếp tới hình dạng của đường biên và kiểm soát hiện tượng _quá khớp_.

Trong python để _phân loại đường biên mềm_ thì chúng ta có thể sử dụng module [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) hoặc [sklearn.svm.LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) thông qua việc thiết lập đối số $C$ thấp. Ở ví dụ hình 2 bạn cũng có thể thấy với `Soft Margin SVM` thì chúng ta để $C=0.05$

`LinearSVC` cũng tương ứng với `SVC` với cấu hình `kernel='linear'`, module `LinearSVC` có tốc độ nhanh hơn so với `SVC` nên được khuyến nghị hồi qui với những tập dữ liệu lớn. Khi huấn luyện với bộ dữ liệu kích thước nhỏ (khoảng vài ngàn quan sát) thì có thể sử dụng SVC. Ưu điểm của `SVC` đó là chúng ta được phép lựa chọn đa dạng các phép biến đổi kernel. Trong khi `LinearSVC` là phương pháp dựa trên kernel `linear`. Trong `LinearSVC` cho phép chúng ta lựa chọn được loại hàm điều chuẩn thông qua đối số `penalty` và dạng của hàm mất mát thông qua đối số `loss`.

Tiếp theo chúng ta sẽ cùng tìm hiểu về kernel trong thuật toán SVM.

+++ {"id": "z6lAjnx7y_NU"}

# 7.4. Kernel trong SVM

Trong thuật toán _phân loại đường biên mềm_ SVM chúng ta sẽ quyết định nhãn cho một điểm dữ liệu dựa vào đường biên phân loại như sau:

$$\begin{split}
y = \left\{
\begin{matrix}
1 \text{ if } b + \mathbf{w}^{\intercal}\mathbf{x}_i \geq 0 \\
-1 \text{ if otherwise}
\end{matrix}
\right.\end{split}$$

Như vậy trong trường hợp mô hình phân loại kém thì $b + \mathbf{w}^{\intercal}\mathbf{x}_i$ sẽ là một đường biên rất đơn giản. Để tạo tính phi tuyến cho đường biên phân chia thì trong SVM chúng ta sử dụng các hàm kernel thay cho biến đầu vào. 

Một cách khái quát, giả định các hàm $f_1(.), f_2(.), \dots , f_n(.)$ là các hàm kernel. Khi đó phương trình đường biên sẽ được chuyển sang phương trình của hàm kernel như sau:

$$h(\mathbf{x}, \mathbf{w}) = b + w_1f_1(x_1) + w_2 f_2(x_2) + \dots + w_n f_n(x_n) \tag{5}$$

Thông qua biến đổi kernel có thể tạo ra được những đường biên phân loại phức tạp hơn và giúp cải thiện độ chính xác của mô hình phân loại.

Bên dưới là một số dạng kernel thường được sử dụng trong SVM.

+++ {"id": "xLZ03hQHzDBW"}

## 7.4.1. Kernel RBF

Trên phân phối của tập dữ liệu chúng ta xác định một tập hợp các điểm landmark.
Landmark ở đây có thể được hiểu như là những điểm tiêu biểu đại diện cho các nhãn.

Hàm kernel RGB đo lường mức độ tương đồng giữa một điểm dữ liệu $\mathbf{x}$ bất kỳ với một điểm landmark $l$ có dạng như sau:

$$\phi(\mathbf{x}, l) = \exp(-\frac{||\mathbf{x}-l||_2^2}{2\sigma^2})$$

Ký hiệu $||\mathbf{x}||_2$ là [chuẩn bậc hai](https://phamdinhkhanh.github.io/deepai-book/ch_algebra/appendix_algebra.html#khai-niem-chuan) của $\mathbf{x}$. Các bạn có nhận ra hàm số trên quen thuộc chứ ? Nếu chúng ta coi $l$ như là tâm của các phân phối dữ liệu và $\mathbf{x}$ là các điểm dữ liệu ngẫu nhiên thì hàm $\phi(\mathbf{x}, l)$ chính là _hàm mật độ xác suất_ pdf của phân phối chuẩn có tâm là $l$. Hình dạng của phân phối này là một hình quả chuông đối xứng hai bên qua tâm.

![](https://ds055uzetaobb.cloudfront.net/image_optimizer/1dbcc5a80e3fb541aa4678fcff58bb26ca717902.png)

Giá trị của $\phi(\mathbf{x}, l)$ sẽ tiến gần tới 1 trong trường hợp $\mathbf{x}$ và $l$ gần nhau và trường hợp những điểm này là cách xa nhau thì giá trị $\phi(\mathbf{x}, l)$ sẽ tiến dần tới 0.

Ý tưởng của phương pháp kernel RGB đó là đưa thêm thước đo độ tương đồng giữa điểm dữ liệu với các landmark vào mô hình. Như vậy các điểm phân phối gần landmark thì có giá trị kernel gần 1 và tách biệt so với các điểm nằm cách xa landmark. Những điểm này sẽ có giá trị gần 0. Sử dụng toạ độ được tính toán sau khi chiếu lên không gian kernel thì chúng ta sẽ thấy được sự tách biệt rõ ràng giữa hai nhóm.


![](https://i.imgur.com/wlBAdui.jpeg)

Chẳng hạn trong hình minh hoạ trên chúng ta có hai điểm landmark là $l_1$ và $l_2$ tạo thành một hình dạng phân phối đặc trưng cho một lớp (phân phối được bao quanh bởi đường nét đứt). Điểm $\mathbf{x}_1$ gần $l_1$ và $\mathbf{x}_2$ nằm gần $l_2$. Khi thực hiện phép biến đổi theo kernel RBF trên hai điểm landmark thì chúng ta chuyển sang một hệ trục toạ độ mới là $f_1$ và $f_2$. Giá trị ánh xạ từ một điểm $\mathbf{x}$ lên trục toạ độ này là một điểm có toạ độ:

$$(\phi(\mathbf{x}, l_1), \phi(\mathbf{x},l_2))$$

Thể hiện trên hình bên phải là 3 điểm ảnh $\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3$ của hình bên trái. Ta nhận thấy $\mathbf{x}_1$ do gần $l_1$ hơn nên có $f_1$ cao và $f_2$ thấp; tương tự như vậy $\mathbf{x}_2$ gần $l_2$ hơn nên có $f_2$ cao, $f_1$ thấp. $\mathbf{x}_3$ thì cách xa cả hai điểm landmarks này nên có toạ độ sát điểm $(0, 0)$. Trên không gian chiếu ta dễ dàng phân biệt được ảnh của các điểm này bằng một đường biên nét đứt.



+++ {"id": "wz8Zq4exv_E5"}

## 7.4.2. Các kernel khác cho SVM

Ngoài kernel RBF chúng ta còn một số kernel khác cho SVM như sau:

* Kernel tuyến tính (_linear_): Đây là tích vô hướng giữa hai véc tơ.

$$\phi(\mathbf{x}_1\mathbf{x}_2) = \mathbf{x}_1^{\intercal}\mathbf{x}_2$$

* Kernel đa thức (_poly_): Tạo ra một đa thức bậc cao kết hợp giữa hai véc tơ.

$$\phi(\mathbf{x}_1, \mathbf{x}_2) = (\gamma \mathbf{x}_1^{\intercal}\mathbf{x}_2+r)^d$$

* Kernel Sigmoid: Dựa trên kernel về đa thức, chúng ta đưa chuyển tiếp qua hàm tanh. Hàm tanh có thể biểu diễn theo hàm sigmoid nên đây được gọi là kernel Sigmoid.

$$\phi(\mathbf{x}_1, \mathbf{x}_2) = \text{tanh}(\gamma \mathbf{x}_1^{\intercal}\mathbf{x}_2+r)$$

Trong quá trình huấn luyện SVM chúng ta cần thử với những kernel khác nhau để tìm ra một kernel hiệu quả. Ở mục 6 thực hành các bạn sẽ được làm quen với việc tunning kernel.

Chú ý đối với các từng kernel thì chúng ta lại có thể tunning các siêu tham số (_hyperameter_) của chúng. Chẳng hạn như trong kernel đa thức chúng ta có thể tunning đối với bậc $d$ của đa thức và hệ số $\gamma$. Những phần này sẽ được hướng dẫn chi tiết hơn ở mục 6.

+++ {"id": "RmWcTxQIB1v3"}

# 7.5. Ví dụ về bài toán SVM

Tiếp theo chúng ta sẽ cùng sử dụng SVM để phân loại bộ dữ liệu `iris`.

```{code-cell}
:id: hP30NqyICIBY

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris["data"]
y = (iris["target"] == 2).astype(np.int8) # 1 if virginica, 0 else
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: BiqrBl6jCX5x
outputId: ecf4d2a1-d1eb-4742-eb0b-fb22cf15544a
---
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
svm_pl = Pipeline((
    ("scaler", StandardScaler()),
    ("linear_svc", SVC(C=1, kernel="linear", probability = True))
  )
)

svm_pl.fit(X, y)

scores = cross_val_score(svm_pl, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: {:.03f}, Standard Deviation Accuracy: {:.03f}'.format(np.mean(scores), np.std(scores)))
```

+++ {"id": "17nqrPsuGu7P"}

Dự báo cho một quan sát mới

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: BvU5tqEgGyXG
outputId: 7e6b4a2d-dc6b-4961-8bcd-2caa4ff0657c
---
# Dự báo nhãn
svm_pl.predict(np.array([[1.2, 3.3, 2.2, 4.5]]))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: bZW9OTs3HPlP
outputId: 98d3deda-505e-4c72-f4aa-4b9fc622e378
---
# Dự báo xác suất, chỉ được khi probability trong SVC() được set True.
svm_pl.predict_proba(np.array([[3.2, 3.0, 4.2, 4.5]]))
```

+++ {"id": "xeBsUk06HuNh"}

## 7.5.1. Bài toán SVM cho dữ liệu dạng phi tuyến

Mặc dù SVM có kết quả khá tốt cho bài toán phân loại nhưng có một số tình huống dữ liệu là phức tạp và yêu cầu chúng ta phải thực hiện các phép biến đổi phi tuyến đối với biến đầu vào để tạo thành những đường biên phức tạp hơn. Kỹ thuật chuẩn hoá đa thức (_polynormial_) được áp dụng để tạo ra những biến bậc cao sẽ hữu ích trong những tình huống này:

```{code-cell}
:id: 1ni_f94eIdLj

from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

svm_ply_pl = Pipeline((
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("linear_svc", SVC(C=1, kernel="linear", probability = True))
  )
)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Xvb2anYZK2rd
outputId: d68fdc42-d5b2-4e3b-f1e9-6aa5aab61752
---
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

scores = cross_val_score(svm_ply_pl, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: {:.03f}, Standard Deviation Accuracy: {:.03f}'.format(np.mean(scores), np.std(scores)))
```

+++ {"id": "gOekny1tLDn5"}

Như vậy sau khi áp dụng _chuẩn hoá đa thức_ thì độ chính xác đã tăng lên từ `0.96` lên `0.969`. Đây là một trong những kỹ thuật thường được áp dụng để giúp cải thiện độ chính xác cho SVM.

+++ {"id": "33uV15ha2OzZ"}

Trên thực thế thì kỹ thuật chuẩn hoá đa thức cũng tương tự như việc sử dụng kernel `poly` trong module SVC. Lưu ý rằng mặc dù kỹ thuật chuẩn hoá đa thức thường mang lại sự cải tiến đáng kể về độ chính xác cho mô hình nhưng số lượng biến mà nó tạo ra bao gồm những biến tích chéo (dạng $x_1^p x_2^q$) và biến bậc cao (dạng $x_1^l$) là rất lớn. Do đó sẽ dễ xảy ra hiện tượng _quá khớp_ và đồng thời gia tăng chi phí huấn luyện và tính toán.

Tiếp theo ta sẽ thực hành tunning kernel trong SVM.

+++ {"id": "k6n-vPiNJSiz"}

## 7.5.2. Sử dụng kernel SVM

Khi huấn luyện mô hình SVM chúng ta cần thử với nhiều kernels khác nhau để tìm ra kernel tốt nhất cho bộ dữ liệu huấn luyện. Các kernel phổ biến đó là:
`linear, poly, rbf, sigmoid` như đã được giới thiệu ở mục 5.

Ngoài ra nếu mô hình gặp hiện tượng quá khớp thì chúng ta cần điều chỉnh giảm hệ số $C$ của mô hình SVM để gia tăng ảnh hưởng của thành phần kiểm soát.

```{code-cell}
:id: 3OAnhFfdJWQ6

from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


kernels = ['linear', 'poly', 'rbf', 'sigmoid']

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

all_scores = []
# Đánh giá toàn bộ các mô hình trên tập K-Fold đã chia
for kernel in kernels:
  svm_kn_pl = Pipeline((
      ("scaler", StandardScaler()),
      ("linear_svc", SVC(C=1, kernel=kernel, probability = True))
    )
  )
  scores = cross_val_score(svm_kn_pl, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
  all_scores.append((kernel, scores))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 584
id: MRQljzK3K6Mh
outputId: ca693100-b76e-4be6-f373-8de3e449a43a
---
import matplotlib.pyplot as plt

# Draw bboxplot 
plt.figure(figsize=(16, 8))
plt.boxplot([score[1] for score in all_scores])
plt.xlabel('Scale', fontsize=16)
plt.ylabel('cm', fontsize=16)
plt.xticks(np.arange(len(kernels))+1, kernels, rotation=45, fontsize=16)
plt.title("Scores Metrics", fontsize=18)
```

+++ {"id": "JxTMpi6ALrhC"}

Như vậy ta có thể thấy các kernel hiệu quả chính là `rbf` và `linear` khi cùng có giá trị trung vị vào khoảng 0.97 và cao hơn mức trung bình của kernel kém nhất là `Sigmoid` là 0.07 điểm. Đây là một mức cải thiện khá đáng kể cho một bài toán phân loại nhị phân.

+++ {"id": "mAEmZydX8qOZ"}

## 7.5.3. Tunning siêu tham số cho một kernel

Đối với mỗi một dạng hàm kernel, căn cứ vào phương trình của chúng ta có thể xác định được những siêu tham số cần tunning.

Chẳng hạn như đối với danh sách các kernel được cung cấp ở mục 5.1 thì chúng ta có thể tunning các tham số như sau:

* kernel tuyến tính: tham số C.
* kernel đa thức: tham số $C, \gamma, d$
* kernel RBF: tham số $C, \gamma$.
* kernel sigmoid: tham số $C, \gamma, d$

Công thức tổng quát của một mô hình SVC:

```
sklearn.svm.SVC(*, 
  C=1.0, 
  kernel='rbf', 
  degree=3, 
  gamma='scale', 
  coef0=0.0, 
  class_weight=None, 
  decision_function_shape='ovr',
  random_state=None
)
```


Trong class SVC của sklearn thì hệ số $\gamma$ tương ứng với đối số `coef0`, hệ số bậc đa thức $d$ là đối số `degree`, trọng số $C$ của hàm chi phí chính là đối số `C` và loại kernel là đối số `kernel`.

Ngoài ra trong trường hợp mẫu bị mất cân bằng nghiêm trọng thì chúng ta thiết lập `class_weight` để phạt nặng hơn những trường hợp mẫu thiểu số.

`decision_function_shape` là đối số cho phép chúng ta cấu hình kết quả xác suất dự báo trả về là theo phương pháp `one-vs-rest` hay `one-vs-one`. Nếu theo phương pháp `one-vs-rest` thì mô hình phân loại gồm $C$ nhãn sẽ được chia thành $C$ bài toán phân loại con, mỗi một bài toán tương ứng với một dự báo xác suất thuộc về nhãn $i$. Còn đối với bài toán `one-vs-one` chúng ta sẽ tìm cách xây dựng $C\times(C-1)$ mô hình phân loại cho một cặp nhãn $(i, j)$ bất kỳ. Đối với bài toán phân loại nhị phân thì `decision_function_shape = ovr` tương ứng với dự báo xác suất tương ứng với nhãn $(0, 1)$.

Bên dưới là một ví dụ mẫu về cách tunning tham số trên GridSearch đối với mô hình SVM.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 8jp5hUN--NVV
outputId: c0f63633-abd9-459d-f396-8965d4d7c138
---
from sklearn.model_selection import GridSearchCV

parameters = {
    'clf__kernel':['linear', 'rbf', 'poly', 'sigmoid'],  # Các dạng hàm kernel
    'clf__C':[0.05, 1, 100], # Trọng số của phạt phân loại sai
    'clf__coef0': [2, 4], # Tương ứng với tham số gamma của đa thức
    'clf__degree': [1, 2, 3] # Bậc d của đa thức
}


pipeline = Pipeline(
    steps=[("clf", SVC())]
)

gscv = GridSearchCV(pipeline, parameters, cv=5, n_jobs=12, scoring='accuracy', return_train_score=True, error_score=0, verbose=3)
gscv.fit(X, y)
```

+++ {"id": "UvIDloaBh-Pr"}

# 7.6. Tổng kết

Như vậy qua chương này bạn đọc đã được giới thiệu những kiến thức cơ bản gồm:

1. Hàm mất mát trong SVM.
2. Khái niệm về đường biên và lề.
3. Bài toán phân loại SVM với đường biên mềm và đường biên cứng.
4. Các dạng bộ lọc trong SVM.
5. Phương pháp tunning tham số đối với mô hình SVM.

SVM làm một trong những thuật toán hoạt động khá hiệu quả trong lớp các bài toán phân loại và dự báo của học có giám sát. Nắm vững thuật toán này, bạn đọc sẽ có thêm công cụ để tạo ra những mô hình mạnh giúp giải quyết những vấn đề thực tế.

+++ {"id": "sEL3MNdPE1UR"}

# 7.7. Bài tập

1. Hàm mất mát của SVM có dạng là một hàm có dạng như thế nào?
2. Giả định mô hình hồi qui SVM đang gặp hiện tượng _quá khớp_. Làm thế nào để giảm thiểu hiện tượng quá khớp cho mô hình SVM?
3. Kernel trong SVM là gì? Kernel có tác dụng như thế nào đối với mô hình SVM?
4. Có những dạng kernel chính nào trong SVM? Đặc điểm của chúng là gì?
5. Khi huấn luyện một mô hình SVM thì chúng ta cần tinh chỉnh những siêu tham số nào là chủ yếu?

+++ {"id": "awxD8Dd_ofHz"}

# 7.8. Tài liệu

1. [SVM - wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine)
2. [Support Vector Machine introduction to Machine Learning Algorithms](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)
3. [SVM - Machine Learning Cơ bản](https://machinelearningcoban.com/2017/04/09/smv/)
4. [Chapter 7, Sparse Kernel Machines - Pattern Recognition Learning Information Statistics](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738)
4. [Chapter 5, SVM - Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291)
5. [SVM model sklearn](https://scikit-learn.org/stable/modules/svm.html)
6. [Optimization loss function under the hood par](https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-iii-5dff33fa015d#:~:text=The%20loss%20function%20of%20SVM,the%20raw%20model%20output%2C%20%CE%B8%E1%B5%80x.)
