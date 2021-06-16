---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.8.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++ {"deletable": true, "editable": true, "id": "-NNO86RooAf7"}

# 1. Tại sao là phương trình tuyến tính ?

Ở chương này chúng ta sẽ bắt đầu tìm hiểu về các thuật toán trong machine learning. Tôi khuyến nghị bạn đọc nắm vững kiến thức cơ bản ở các chương 1, 2, 3 về đại số tuyến tính, giải tích và xác suất bởi đây là những kiến thức bản lề để bạn nắm vững hơn kiến thức được trình bày tại chương này.

Trong bài này thì chúng ta tìm hiểu về hồi qui tuyến tính (_linear regression_). Trong chương trình THPT các bạn đã từng ít nhiều nghe qua phương trình tuyến tính chẳng hạn như: 

$$y = ax + b$$

Với $a, b$ là các hằng số.

Phương trình trên có **một** biến đầu vào $x$ nên được gọi là phương trình hồi qui tuyến tính **đơn biến**. Trên mặt phẳng hai chiều, các điểm $(x, y)$ biểu diễn bởi một đường thẳng.

Một trường hợp khác của phương trình tuyến tính cũng khá thường xuyên bắt gặp đó là:

$$y = a x_1 + b x_2 + c$$

Với $a, b, c$ là các hằng số.

Phương trình trên có **nhiều hơn một biến** đầu vào nên được gọi là phương trình hồi qui tuyến tính **đa biến**. Biểu diễn trong không gian ba chiều của nó là một mặt phẳng (_plane_).

Từ hai phương trình trên chúng ta có thể suy ra công thức tổng quát của phương trình hồi qui tuyến tính đó là:

$$y = a_0 + a_1 x _1 + a_2 x_2 + \dots + a_n x_n$$

Theo khái niệm toán học thì các phương trình dạng này được gọi là siêu phẳng (_hyperplane_).

Có một số lý do khiến cho phương trình tuyến tính được lựa chọn để biểu diễn mối quan hệ giữa biến độc lập (các biến $x_i$) và biến phụ thuộc ($y$) trong machine learning. 

* Phương trình tuyến tính có thể khái quát hoá được các phương trình nhân khi thực hiện phép logarith. Chẳng hạn nếu $y = x_1^{\alpha} x_2^{\beta}$ có thể biểu diễn thành 

$$\log{y} = \alpha \log{x_1} + \beta \log{x_2}$$

là một phương trình dạng tuyến tính. 

* Phương trình tuyến tính là định dạng định dạng dễ hiểu, dễ thực hiện. Ví dụ nếu bạn tìm cách biểu diễn $y$ và $x$ theo một phương trình dạng như:

$$y = \frac{sin(\sqrt{x^2+1}).e^x}{cos(x^{3/2})+x^3+2x+1}$$

Thì nó là một quan hệ rất phức tạp và không dễ thực hiện và tính toán.

* Phương trình tuyến tính có thể dễ dàng giải thích mối quan hệ giữa các biến độc lập và phụ thuộc. Thật vậy, trong phương trình $y = a_0 + a_1 x_1 + a_2 x_2 + \dots + a_n x_n$ thì $a_1$ thể hiện tác động biên của $x_1$ lên $y$. Khi $x_1$ tăng/giảm 1 đơn vị thì $y$ tăng/giảm $a_1$ đơn vị.

* Phương trình tuyến tính có thể biểu diễn được mọi mối quan hệ phức tạp của biến. Chúng ta có thể thấy phương trình $y = a x+b$ là một đường thẳng nhưng nếu ta thêm $x^2$ thì phương trình $y = a x^2+bx+c$ đã trở thành một đường cong phi tuyến. Chỉ với phương trình tuyến tính chúng ta có thể biểu diễn được hầu như mọi mối quan hệ dữ liệu phức tạp giữa $x$ và $y$. 

+++ {"id": "OvH2_a_CJR1N"}

# 2. Ứng dụng của hồi qui tuyến tính

Phương trình hồi qui tuyến tính có rất nhiều ứng dụng trong thực tiễn và là một trong những lớp mô hình đặc biệt quan trọng trong machine learning. Chúng ta sẽ không thể kể hết được ứng dụng của nó trong một vài dòng. Nhưng chúng ta có thể xét đến một vài ví dụ tiêu biểu và gần gũi với mọi người. Chẳng hạn như các bạn thường được nghe các dự báo trên báo đài, tivi về chỉ số lạm phát, chỉ số tăng trưởng của quốc gia trong năm nay được dự báo là bao nhiêu? Đó chính là ứng dụng của hồi qui tuyến tính. Một doanh nghiệp muốn đưa ra dự báo về nhu cầu thị trường để chuẩn bị kế hoạch sản suất tốt hơn. Giá cả của các chỉ số chứng khoán, chỉ số tài chính có thể được dự báo dựa trên hồi qui tuyến tính.

Tóm lại hầu hết các biến phụ thuộc mà **liên tục** thì đều có thể dự báo được dựa trên hồi qui tuyến tính.

+++ {"id": "UpmTNWtWPYoc"}

# 3. Hàm loss function

Mục tiêu của tất cả các mô hình học có giám sát (_supervised learning_) trong machine learning là tìm ra một phương trình sao cho nó **khớp** với dữ liệu nhất. Vậy khớp ở đây được hiểu như thế nào? Chúng ta sẽ cần đến những thước đo để biết được mô hình có khớp hay không. Đó chính là loss function.

Mỗi một lớp mô hình sẽ có một loss function khác nhau. Trong hồi qui tuyến tính loss function được lựa chọn là MSE. Đây là một hàm số đo lường trung bình của tổng bình phương sai số giữa giá trị dự báo và giá trị thực tế. Gỉa sử chúng ta xét phương trình hồi qui đơn biến gồm $n$ quan sát có biến phụ thuộc là $\mathbf{y} = \{y_1, y_2,..., y_n\}$ và biến độc lập $\mathbf{x} = \{x_1, x_2,...,x_n\}$. Véc tơ $\mathbf{w} = (w_0, w_1)$ có giá trị $w_0, w_1$ lần lượt là hệ số góc và hệ số ước lượng. Phương trình hồi qui tuyến tính đơn biến có dạng:

$$\hat{y_i} = f(x_i) = w_0 + w_1*x_i$$

Trong đó $(x_i, y_i)$ là điểm dữ liệu thứ $i$.

Mục tiêu của chúng ta là đi tìm véc tơ $\mathbf{w}$ sao cho sai số giữa giá trị dự báo và thực tế là nhỏ nhất. Tức là tối thiểu hoá hàm loss function dạng MSE:

$$\mathcal{L}(\mathbf{w|x, y}) = \frac{1}{2n} \sum_{i = 1}^{n}(y_i - \hat{y_i})^2 = \frac{1}{2n} \sum_{i = 1}^{n}(y_i - w_0 - w_1 *  x_i)^2$$

Ký hiệu $\mathcal{L}(\mathbf{w|X, y})$ thể hiện rằng loss function là một hàm theo $\mathbf{w}$ khi ta đã biết đầu vào là véc tơ $\mathbf{x}$ và véc tơ biến phụ thuộc $\mathbf{y}$. Ta có thể tìm cực trị của phương trình trên dựa vào đạo hàm theo $w_0$ và $w_1$ như sau:

* Đạo hàm theo $w_0$:

$$\begin{eqnarray}\frac{\delta{\mathcal{L}(\mathbf{w|x})}}{\delta{w_0}} & = & \frac{-1}{n}\sum_{i = 1}^{n}(y_i - w_0 - w_1*x_i) \\
& = & \frac{-1}{n}\sum_{i=1}^n y_i + w_0 + w_1 \frac{1}{n} \sum_{i=1}^n x_i\\
& = & -\bar{\mathbf{y}} + w_0 + w_1 \bar{\mathbf{x}}\\
& = & 0\end{eqnarray} \tag{1}$$

* Đạo hàm theo $w_1$:

$$\begin{eqnarray}\frac{\delta{\mathcal{L}(\mathbf{w|x})}}{\delta{w_1}} & = &\frac{-1}{n}\sum_{i = 1}^{n}x_i(y_i - w_0 - w_1*x_i) \\
& = & \frac{-1}{n} \sum_{i=1}^n x_i y_i + w_0 \frac{1}{n}\sum_{i=1}^n x_i+w_1\frac{1}{n}\sum_{i=1}^n x_i^2\\
& = & -\bar{\mathbf{xy}} + w_0 \bar{\mathbf{x}} + w_1 \bar{\mathbf{x}^2}  \\
& = & 0 \tag{2}
\end{eqnarray}$$

Từ phương trình (1) ta suy ra: $w_0 = \mathbf{\bar{y}}-w_1\mathbf{\bar{x}}$. Thế vào phương trình (2) ta tính được:

$$\begin{eqnarray}-\bar{\mathbf{xy}} + w_0 \bar{\mathbf{x}} + w_1 \bar{\mathbf{x}^2} & = & -\bar{\mathbf{xy}} + (\mathbf{\bar{y}}-w_1\mathbf{\bar{x}}) \bar{\mathbf{x}} + w_1 \bar{\mathbf{x}^2} \\
& = & -\bar{\mathbf{xy}} + \mathbf{\bar{y}}\bar{\mathbf{x}}-w_1\mathbf{\bar{x}}^2 + w_1 \bar{\mathbf{x}^2} \\
& = & 0 \end{eqnarray}$$

Từ đó suy ra: 
$$w_1 = \frac{\mathbf{\bar{x}\bar{y} - \bar{xy}}}{\mathbf{\bar{x}^2-\bar{x^2}}}$$

Sau khi tính được $w_1$ thế vào ta tính được:
$$w_0 = \mathbf{\bar{y}}-w_1\mathbf{\bar{x}}$$

Đạo hàm bậc nhất bằng 0 mới chỉ là điều kiện cần để $\mathbf{w}$ là cực trị của hàm loss function. Để khẳng định cực trị đó là cực tiểu thì chúng ta cần chứng minh thêm đạo hàm bậc hai lớn hơn hoặc bằng 0 hay hàm số đó là hàm lồi. Điều này khá dễ dàng và mình xin dành cho bạn đọc. Bài tập bên dưới dây sẽ giúp bạn hiểu dễ hơn cách tìm nghiệm của phương trình hồi qui tuyến tính đơn biến.

**Bài tập:**

chúng ta có 15 căn hộ với diện tích (đơn vị m2):

$$\mathbf{x} = [73.5, 75. , 76.5, 79. , 81.5, 82.5, 84. , 85. , 86.5, 87.5, 89. , 90. , 91.5]$$ 

Mức giá của căn hộ lần lượng là (đơn vị tỷ VND đồng):

$$\mathbf{y} = [1.49, 1.50, 1.51,  1.54, 1.58, 1.59, 1.60, 1.62, 1.63, 1.64, 1.66, 1.67, 1.68]$$

Xây dựng phương trình hồi qui tuyến tính đơn biến giữa diện tích và giá nhà.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 523
id: W_BnIrNGevqQ
outputId: fd5ddae0-860d-4201-9e50-a34178acbccf
---
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline
# area
x = np.array([[73.5,75.,76.5,79.,81.5,82.5,84.,85.,86.5,87.5,89.,90.,91.5]]).T
# price
y = np.array([[1.49,1.50,1.51,1.54,1.58,1.59,1.60,1.62,1.63,1.64,1.66,1.67,1.68]]).T

# Visualize data
def _plot(x, y, title="", xlabel="", ylabel=""):
  plt.figure(figsize=(14, 8))
  plt.plot(x, y, 'r-o', label="price")
  x_min = np.min(x)
  x_max = np.max(x)
  y_min = np.min(y)
  y_max = np.max(y)
  # mean price
  ybar = np.mean(y)
  plt.axhline(ybar, linestyle='--', linewidth=4, label="mean")
  plt.axis([x_min*0.95, x_max*1.05, y_min*0.95, y_max*1.05])
  plt.xlabel(xlabel, fontsize=16)
  plt.ylabel(ylabel, fontsize=16)
  plt.text(x_min, ybar*1.01, "mean", fontsize=16)
  plt.legend(fontsize=15)
  plt.title(title, fontsize=20)
  plt.show()

_plot(x, y, 
      title='Giá nhà theo diện tích',  
      xlabel='Diện tích (m2)', 
      ylabel='Giá nhà (tỷ VND)')
```

+++ {"id": "44WBjiE0hvIK"}

Tính $w_0, w_1$

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: zUTQT9U5h1c_
outputId: 84dbc275-a21e-4978-feea-e6ca4fe5ca20
---
# tính trung bình
xbar = np.mean(x)
ybar = np.mean(y)
x2bar = np.mean(x**2)
xybar = np.mean(x*y)

# tính w0, w1
w1 = (xbar*ybar-xybar)/(xbar**2-(x2bar))
w0 = ybar-w1*xbar

print('w1: ', w1)
print('w0: ', w0)
```

+++ {"id": "n77vFrwcdp9Z"}


Như vậy ta có thể tìm được lời giải của phương trình hồi qui tuyến tính đơn biến thông qua đạo hàm bậc nhất. Tuy nhiên bài với phương trình hồi qui tuyến đa biến thì lời giải sẽ phức tạp hơn một chút vì chúng ta sẽ cần tới kiến thức về giải tích ma trận.

+++ {"id": "_HgSjdc9BnDa"}

# 4. Hồi qui tuyến tính đa biến

Hồi qui tuyến tính đa biến là hồi qui tuyến tính với nhiều hơn một biến đầu vào. Hồi qui tuyến tính đa biến phổ biến hơn so với đơn biến vì trên thực tế rất hiếm các tác vụ dự báo chỉ gồm một biến đầu vào. Phương trình hồi qui của nó có dạng:

$$\hat{y_i} = f(x_1, x_2, \dots, x_p) = w_0 + w_1 x_{i1} + \dots + w_p x_{ip} = \mathbf{w}^{\intercal}\mathbf{x}_i$$

Ở đây ta xem $\mathbf{x}_i$ là một véc tơ đại diện cho quan sát thứ $i$. Cụ thể nó gồm các giá trị $(x_{i1}, x_{i2},\dots,x_{ip})$. Ma trận $\mathbf{X}$ có kích thước $n \times p$ có mỗi dòng là một quan sát và mỗi cột là một biến. Giá trị $x_{ip}$ là quan sát thứ $i$ của biến thứ $p$. Ma trận mở rộng của $\mathbf{X}$ được ký hiệu là $\bar{\mathbf{X}}$ chính là ma trận có thêm véc tơ cột $\mathbf{1}$ được thêm vào đầu tiên. Khi đó đối với toàn bộ tập dữ liệu ta có:

$$\mathbf{\hat{y}} = f(\mathbf{X}) = 
\begin{bmatrix}
    1  & x_{11} & \dots  & x_{1p}  \\
    1  & x_{21} & \dots  & x_{2p}  \\
    \vdots & \vdots & \ddots & \vdots \\
    1  & x_{n1} & \dots  & x_{np}
\end{bmatrix}\begin{bmatrix}
    w_{0}  \\
    w_{1}  \\
    \vdots \\
    w_{p}
\end{bmatrix}
= \bar{\mathbf{X}}\mathbf{w}$$


véc tơ sai số giữa $y-\hat{y}$ có thể được biểu diễn bởi:

$$\mathbf{e} = \mathbf{y}-\mathbf{\hat{y}} = \mathbf{y}-\bar{\mathbf{X}}\mathbf{w}$$

Hàm loss function MSE là trung bình tổng bình phương của các sai số nên nó có dạng:

$$\mathcal{L}(\mathbf{w|x, y}) = \frac{1}{2n} \sum_{i = 1}^{n}(y_i - \hat{y_i})^2 = \frac{1}{2} \mathbf{e}^{\intercal}\mathbf{e} = (\mathbf{y}-\bar{\mathbf{X}}\mathbf{w})^{\intercal}(\mathbf{y}-\bar{\mathbf{X}}\mathbf{w}) = ||\bar{\mathbf{X}}\mathbf{w} - \mathbf{y}||_{2}^{2}$$

Ký hiệu $||\bar{\mathbf{X}}\mathbf{w} - \mathbf{y}||_{2}^{2}$ chính là bình phương của norm chuẩn bậc hai mà các bạn đã được tìm hiểu ở chương đại số. Bằng cách khai triển đại số tuyến tính ta tính được đạo hàm hàm loss function:

$$\frac{\partial\mathcal{L}(\mathbf{w})}{\mathbf{w}} = \mathbf{\bar{X}}^{\intercal}(\mathbf{\bar{X}}\mathbf{w} - \mathbf{y})$$

Nghiệm của phương trình hồi qui:

$$\mathbf{w} = (\mathbf{\bar{X}^{\intercal}\bar{X}})^{-1}\mathbf{\bar{X}}^{\intercal}\mathbf{y} = (\mathbf{A}^{-1}\mathbf{b})$$

Ở Trên để ta đã rút gọn $\mathbf{A} = \mathbf{\bar{X}^{\intercal}\bar{X}}$ và $\mathbf{\bar{X}}^{\intercal}\mathbf{y} = \mathbf{b}$

Phương hình hồi qui đa biến có nghiệm khi $\mathbf{A}$ là khả nghịch.

Bạn đọc có thể không cần hiểu hết công thức đạo hàm và khai triển ma trận ở trên. Mình cũng không khuyến nghị các bạn hồi qui đa biến từ đầu bởi tất cả đã được gói gọn trong hàm LinearRegression của thư viện sklearn.

+++ {"id": "vJioVY1gGw2W"}

# 5. Huấn luyện mô hình hồi qui tuyến tính trên sklearn

Sklearn có thể coi là một package toàn diện của python về data science. Package này có thể cho phép chúng ta huấn luyện hầu hết các mô hình machine learning, xây dựng pipeline, chuẩn hoá và xử lý dữ liệu đầu vào và cross validation dữ liệu.

Vì vai trò quan trọng của sklearn nên tôi sẽ có một chương khác để hướng dẫn chi tiết về cách khai thác package này trong việc huấn luyện mô hình.

Trong phần này chúng ta sẽ cùng tìm hiểu cách huấn luyện mô hình hồi qui tuyến tính trên sklearn. Quay trở lại bài toán trên, nếu chúng ta thêm thông tin về khoảng cách tới trung tâm.

$$\mathbf{x}_2 = [20, 18, 17, 16, 15, 14, 12, 10, 8, 7, 5, 2, 1]$$

Khi đó bài toán trở thành hồi qui đa biến. Trong qui trình xây dựng và huấn luyện mô hình chung chúng ta sẽ lần lượt đi qua các bước chính.

1. Thu thập dữ liệu.
2. Làm sạch dữ liệu.
3. Lựa chọn dữ liệu đầu vào.
4. Chuẩn hoá dữ liệu.
5. Phân chia tập train/test.
6. Huấn luyện và đánh giá mô hình.

Ở bài toán này tôi chỉ muốn cho các bạn thấy cách thức huấn luyện mô hình như thế nào nên chỉ cần thực hiện bước 6.




```{code-cell} ipython3
:id: T-x-ekcHLIwl

from sklearn import linear_model

# area
x1 = np.array([[73.5,75.,76.5,79.,81.5,82.5,84.,85.,86.5,87.5,89.,90.,91.5]]).T
# distance to center
x2 = np.array([[20, 18, 17, 16, 15, 14, 12, 10, 8, 7, 5, 2, 1]]).T
# input matrix X
X = np.concatenate([x1, x2], axis = 1)

# price
y = np.array([[1.49,1.50,1.51,1.54,1.58,1.59,1.60,1.62,1.63,1.64,1.66,1.67,1.68]]).T
```

+++ {"id": "7QCLBzIbLzM8"}

Tiếp theo chúng ta sẽ vẽ biểu đồ giữa khoảng cách tới trung tâm và giá nhà. Quá trình vẽ biểu đồ này rấy quan trọng vì nó giúp ta có một cái nhìn tổng quan về mối quan hệ giữa các điểm dữ liệu và phát hiện những bất thường dữ liệu.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 525
id: lDiWW9XlMlVf
outputId: 10496351-4e5d-44fb-af4d-fbc34ba44fc5
---
import matplotlib.pyplot as plt

_plot(x2, y, 
      title='Giá nhà theo khoảng cách tới TT',  
      xlabel='Khoảng cách tới TT (km)', 
      ylabel='Giá nhà (tỷ VND)')
```

+++ {"id": "gGeF4cV8RAB-"}

Ta nhận thấy rằng khi khoảng cách tới trung tâm giảm thì giá nhà càng tăng. Nhận định này càng củng cố thêm việc lựa chọn biến khoảng cách tới trung tâm làm biến đầu vào là có ý nghĩa. 

Tiếp theo ta sẽ huấn luyện mô hình. Trong sklearn cách thức chung khi huấn luyện mọi mô hình đều là khởi tạo mô hình đó với các tham số và sau đó truyền dữ liệu vào hàm fit để huấn luyện. Đây là đặc điểm mà bạn cần nhớ khi xây dựng mọi mô hình vì nó sẽ lặp lại như vậy.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: z2qd_8xNSHgo
outputId: 3289bff0-3bec-49e9-db1e-b03619837a5e
---
# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=True) # fit_intercept = False for calculating the bias

regr.fit(X, y)

# Compare two results
print( 'Coefficient : ', regr.coef_ )
print( 'Interception  : ', regr.intercept_ )
```

+++ {"id": "nZJr7leHhE6h"}

# 6. Đồ thị hoá kết quả mô hình

Sau khi đã huấn luyện thành công mô hình chúng ta cần trình bày kết quả của mình dưới một dạng trực quan, dễ hiểu. Đây là một trong những kỹ năng quan trọng vì nó giúp mô hình của bạn trở nên có sức thuyết phục hơn với mọi người. 

Kỹ năng đồ thị hoá sẽ được mình giới thiệu sâu hơn ở một chương khác. 

+++ {"id": "LvqF47ZUigAs"}

## 6.1. Biểu diễn trong không gian 2 chiều

+++ {"id": "zjxRpwWbTi5W"}

Để thực hiện dự báo thì chỉ cần khởi tạo ma trận $\mathbf{X}$ đầu vào (có các dòng là các quan sát và các cột là các biến) và truyền vào hàm `predict()`. Ta sẽ dự báo giá nhà ngay trên tập train.

```{code-cell} ipython3
:id: RkO7kkXMhVtt

# Dự báo giá nhà ngay trên tập train
ypred = regr.predict(X)
```

+++ {"id": "rlgtcWKWhhZp"}

Chúng ta muốn biết giá trị dự báo và giá trị thực tế khác biệt như thế nào thì có thể biểu diễn chúng trên không gian hai chiều theo diện tích hoặc theo khoảng cách tới trung tâm.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 525
id: UzYDgrmHhm7X
outputId: c320a8df-6644-428f-cd4e-37c3fa1896bd
---
# Visualize data
def _plot_act_pred(x, y_act, y_pred, title="", xlabel="", ylabel=""):
  plt.figure(figsize=(14, 8))
  plt.plot(x, y_act, 'r-o', label="price actual")
  plt.plot(x, y_pred, '--', label="price predict")
  x_min = np.min(x)
  x_max = np.max(x)
  y_min = np.min(y_act)
  y_max = np.max(y_act)
  # mean price
  ybar = np.mean(y_act)
  plt.axhline(ybar, linestyle='--', linewidth=4, label="mean actual")
  plt.axis([x_min*0.95, x_max*1.05, y_min*0.95, y_max*1.05])
  plt.xlabel(xlabel, fontsize=16)
  plt.ylabel(ylabel, fontsize=16)
  plt.text(x_min, ybar*1.01, "mean actual", fontsize=16)
  plt.legend(fontsize=15)
  plt.title(title, fontsize=20)
  plt.show()


_plot_act_pred(x2, y, ypred, 
      title='Giá nhà theo khoảng cách tới TT',  
      xlabel='Khoảng cách tới TT (km)', 
      ylabel='Giá nhà (tỷ VND)')
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 523
id: n-Ih5kILiURq
outputId: 38eb4d56-b80f-4f06-c9e1-3c672967d5f4
---
_plot_act_pred(x1, y, ypred, 
      title='Giá nhà theo diện tích',  
      xlabel='diện tích', 
      ylabel='Giá nhà (tỷ VND)')
```

+++ {"id": "PQGCi31Cin7i"}

## 6.2. Biểu diễn trong không gian 3 chiều

+++ {"id": "91C64T2ohI0N"}

Không gian 3 chiều sẽ cho chúng ta nhiều thông tin hơn về hình dạng phân bố của biến được dự báo hơn là không gian hai chiều.

Để tạo ra được đồ thị 3 chiều trước tiên chúng ta cần tạo ra một lưới giá trị của $(x_1, x_2)$ thông qua hàm `np.meshgrid()` và sau đó dự báo giá trị của $y$ trên lưới giá trị này. 

Tiếp theo chúng ta sẽ áp dụng phương trình hồi qui vừa huấn luyện được để dự báo mọi mức giá nhà có diện tích trong khoảng từ 90-110 m2 và khoảng cách tới tâm trong khoảng từ 10-30 km.

```{code-cell} ipython3
:id: 6tC-rge3UYwI

# Khởi tạo diện tích
x1 = np.arange(90, 111, 1)
x1 = np.expand_dims(x1, axis=1)
# Khởi tạo khoảng cách tới trung tâm
x2 = np.arange(10, 31, 1)
x2 = np.expand_dims(x2, axis=1)
# Ma trận đầu vào
X = np.concatenate([x1, x2], axis = 1)

# Dự báo 
ypred = regr.predict(X)
```

+++ {"id": "BmnjtUKUkdIf"}

Tạo lưới giá trị cho mọi điểm $(x_1, x_2)$ trong miền xác định.

```{code-cell} ipython3
:id: 9vTXa1pzki-H

# Tạo lưới ma trận
x1grid, x2grid = np.meshgrid(x1, x2)
```

+++ {"id": "M0hHDUTwknwl"}

Dự báo cho mọi điểm $(x_1, x_2)$

```{code-cell} ipython3
:id: R5HsAirCktPZ

ys = []
for i in range(len(x1)):
  x1i=x1grid[:, i:(i+1)]
  x2i=x2grid[:, i:(i+1)]
  X = np.concatenate([x1i, x2i], axis=1)
  yi = regr.predict(X)
  ys.append(yi)

ypred = np.concatenate(ys, axis=1)
```

+++ {"id": "BbbC2-SKf3YR"}

Biểu diễn các điểm dữ liệu trong không gian 3 chiều.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 846
id: DFSoRXYxRwT8
outputId: 31c745a6-8464-4443-a7da-91e5d76971ea
---
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15,15))
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(x1grid, x2grid, ypred,cmap=cm.spring,
                       linewidth=0, antialiased=False)


x_pos = np.arange(80.0,100.0, 5)
x_names = [str(x_tick)+ " km" for x_tick in x_pos]
plt.xticks(x_pos, x_names)

y_pos = np.arange(0.0,30.0, 5)
y_names = [str(y_tick)+ " m2" for y_tick in y_pos]
plt.yticks(y_pos, y_names)

ax.set_zlim(1.5, 2.0)
plt.xlabel('area', fontsize=18)
plt.ylabel('distance', fontsize=18)

plt.show()
```

+++ {"id": "Gtf7J3zx7DT_"}

# 7. Đánh gía mô hình hổi qui tuyến tính đa biến

Ngoài MSE là hàm mất mát dùng để làm mục tiêu tối ưu loss function thì chúng ta có thể dựa trên nhiều chỉ số khác để đánh giá một mô hình hồi qui tuyến tính đa biến. Cụ thể như sau:

+++ {"id": "D6axhxnn971N"}

## 7.1. Chỉ số R-squared:

R-squared cho ta biết mức độ các biến giải thích (biến độc lập) sẽ giải thích được bao nhiêu phần trăm các biến được giải thích (biến phụ thuộc). R-squared càng lớn thì mô hình càng tốt, khi R-squared bằng 95% điều đó có nghĩa rằng các biến giải thích đã giải thích được 95% sự biến động của biến được giải thích.

R-squared được xây dựng dựa trên ba chỉ số:

$$TSS = \sum_{i = 1}^{n} (y_i - \bar{y})^2$$
$$RSS = \sum_{i = 1}^{n} (y_i - \hat{y_i})^2$$
$$ESS = \sum_{i = 1}^{n} (\hat{y_i} - \bar{y})^2$$

Trong đó $TSS$ là tổng bình phương sai số toàn bộ mô hình (_Total Sum Squared_), $RSS$ là tổng bình phương sai số ngẫu nhiên (_Residual Sum Squared_), $ESS$ là tổng bình phương sai số được giải thích bởi mô hình (_Explained Sum Squared_)

Ta sẽ chứng mình được $TSS = RSS + ESS$. Thật vậy:


\begin{eqnarray}
TSS & = & \sum_{i = 1}^{n} (y_i - \bar{y})^2 = \sum_{i = 1}^{n} [(y_i - \hat{y_i}) + (\hat{y_i} - \bar{y})]^2 \\
& = & \sum_{i = 1}^{n} (y_i - \hat{y_i})^2 + \sum_{i=1}^{n}(\hat{y_i} - \bar{y})^2 + 2\sum_{i=1}^{N} (y_i - \hat{y_i})(\hat{y_i} - \bar{y}) \\
& = & ESS + RSS + 2\sum_{i=1}^{N} (y_i\hat{y_i} - y_i\bar{y} - \hat{y_i}\hat{y_i} + \hat{y_i}\bar{y}) \\
& = & ESS+RSS + \underbrace{2\sum_{i=1}^{N}\hat{y_i}(y_i-\hat{y_i})}_{A} + \underbrace{2\bar{
y}\sum_{
i=1}^{N} (\hat{y_i} - y_i)}_{B}
\end{eqnarray}

Ta sẽ chứng minh cả hai hạng tử $A$ và $B$ đều bằng 0. Thật vậy, từ phương trình đạo hàm bậc nhất của loss function theo $w_0$ và $w_1$ ta có :

$$\sum_{i=1}^N x_i(y_i-\hat{y_i}) = 0 \tag{3}$$

$$\sum_{i=1}^N (y_i - \hat{y_i}) = 0 \tag{4}$$

Do đó:

$$\bar{
y}\sum_{
i=1}^{N} (\hat{y_i} - y_i) = 0 \leftrightarrow B  = 0$$

Nhân biểu thức (3) với $w_1$ và biểu thức (4) với $w_0$ và cộng vế với vế :

\begin{eqnarray} w_0\sum_{i=1}^N (y_i - \hat{y_i}) + \sum_{i=1}^N w_1x_i(y_i-\hat{y_i}) & = & 0 \\
\leftrightarrow \sum_{i=1}^{N}(w_0+w_1 x_i) (y_i-\hat{y_i}) & = & 0 \\
\leftrightarrow \sum_{i=1}^{N} \hat{y_i}(y_i-\hat{y_i}) & = & 0 \\
\leftrightarrow B & = & 0
\end{eqnarray}

Dòng 2 suy ra 3 là vì $\hat{y_i} = w_0 + w_1 x_i$. Như vậy $A = B = 0$ suy ra $TSS = ESS + RSS$. Chứng minh đẳng thức trên về mặt toán học không quá khó phải không nào ?

Khi đó: 

$$R^2 = 1 - \frac{RSS}{TSS}$$

Như vậy $R^2$ càng lớn thì giá trị tổng bình phương sai số càng nhỏ.

+++ {"id": "1FEfg9z--BcU"}

## 7.2. Chỉ số MAE và MAPE

MAE là chỉ số đo lường trung bình trị tuyệt đối sai số giữa giá trị dự báo và giá trị thực tế.

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i -\hat{y_i}|$$

Chúng ta có thể thấy về bản chất thì MAE chính là norm chuẩn bậc 1. Khi MAE càng nhỏ thì khoảng cách giữa giá trị dự báo và giá trị thực tế càng nhỏ và mô hình càng tốt. Tuy nhiên giá trị MAE không bao hàm được sự khác biệt về mặt đơn vị. Ví dụ như khi chúng ta đo lường sai số về cân nặng của những con voi và cân nặng của những con chuột thì khả năng rất cao là voi có sai số lớn hơn so với chuột. Nhưng sai số này lớn là do chúng ta chưa xét đến kích cỡ của voi và chuột. Chính vì thế để loại bỏ sự khác biệt về mặt đơn vị thì chúng ta sử dụng chỉ số MAPE.

MAPE là chỉ số đo lường tỷ lệ phần trăm sai số giữa giá trị dự báo và giá trị thực tế . Nó là viết tắt của cụm từ _mean absolute percentage error_ có công thức như sau:

$$\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n}|\frac{y_i-\hat{y_i}}{y_i}|$$

Khi một mô hình có $\text{MAPE} = 5\text{%}$ ta nói rằng mô hình có trung bình sai số là $5\text{%}$ so với giá trị trung bình.



+++ {"deletable": true, "editable": true, "id": "8R3J-BoioAjT"}

# 8. Ridge regression và Lasso regression

Ridge regression và Lasso regression là hai mô hình hồi qui áp dụng kỹ thuật hiệu chuẩn (_regularization_) để tránh overfitting. Trước tiên ta tìm hiểu một chút về overfitting:

Overfitting là hiện tượng mà mô hình chỉ khớp tốt trên tập dữ liệu train nhưng không dự báo tốt trên dữ liệu huấn luyện. Đây là trường hợp thường gặp khi huấn luyện các mô hình machine learning. Hiện tượng này gây ảnh hưởng xấu và dẫn tới mô hình không thể áp dụng được vì các dự báo bị sai khi dự báo thực tế. Có nhiều nguyên nhân dẫn tới overfitting. Một trong những nguyên nhân phổ biến đó là tập dữ liệu huấn luyện và dữ liệu dự báo có phân phối khác xa nhau dẫn tới các qui luật học được ở dữ liệu huấn luyện không còn đúng trên dữ liệu dự báo. Hoặc cũng có thể xuất phát từ phía mô hình quá nhiều tham số nên khả năng biểu diễn dữ liệu của nó không mang tính đại diện.

Regularization là kĩ thuật tránh overfiting bằng cách cộng thêm vào loss function thành phần hiệu chuẩn. Thông thường thành phần này ở dạng norm chuẩn bậc 1 hoặc 2 của các hệ số. Trong trường hợp bậc 2 ta gọi là **Ridge regression**:

$$\mathcal{L}(\mathbf{w}; \mathbf{X}, \mathbf{y}) = \frac{1}{n} \sum_{i=1}^{n} (y_i-\hat{y_i})^2 + \alpha||\mathbf{w}||_{2}^2$$

Đối với trường hợp bậc 1 gọi là **Lasso regression**:

$$\mathcal{L}(\mathbf{w}; \mathbf{X}, \mathbf{y}) = \frac{1}{n} \sum_{i=1}^{n} (y_i-\hat{y_i})^2 + \alpha||\mathbf{w}||_1$$

Đối với những hồi qui này thì chúng ta cần tinh chỉnh hệ số $\alpha$ để tìm ra một hệ số alpha tốt nhất với từng bộ dữ liệu.

Trong trường hợp dữ liệu bị overfitting nặng thì cần giảm overtitting bằng cách gia tăng ảnh hưởng của thành phần _regularization term_ bằng cách tăng hệ số $\alpha$. Nếu mô hình không bị overfitting thì có thể lựa chọn $\alpha$ gần 0. Trường hợp $\alpha=0$ thì phương trình hồi qui tương đương với hồi qui tuyến tính đa biến.

Bên dưới ta sẽ cùng xây dựng phương trình hồi qui đối với Ridge regression.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 525
deletable: true
editable: true
id: KLYIQwnQoAjW
outputId: 20c43de0-7744-4f3b-dbde-b76f5fa0d8ac
---
from sklearn.linear_model import Ridge

rid_regr = Ridge(alpha = 0.3, normalize = True)
rid_regr.fit(X, y)

y_pred_rid = rid_regr.predict(X)

_plot_act_pred(x1, y, y_pred_rid, 
      title='Giá nhà theo khoảng cách tới TT',  
      xlabel='Khoảng cách tới TT', 
      ylabel='Giá nhà (tỷ VND)')
```

+++ {"id": "bPbXiBMGosyZ"}

Xây dựng phương trình đối với Lasso regression

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 525
id: WkoicAYVoxtx
outputId: b68e50c9-8191-44f3-c760-27bcb0733543
---
from sklearn.linear_model import Lasso

las_regr = Lasso(alpha = 0.005, normalize = True)
las_regr.fit(X, y)

y_pred_las = las_regr.predict(X)

_plot_act_pred(x1, y, y_pred_las, 
      title='Giá nhà theo khoảng cách tới TT',  
      xlabel='Khoảng cách tới TT', 
      ylabel='Giá nhà (tỷ VND)')
```

+++ {"id": "Wb4xFCZMvZPz"}

## 8.1. Tunning hệ số alpha

Để lựa chọn ra một hệ số alpha phù hợp với mô hình Ridge regression chúng ta sẽ cần phải tạo ra một list các giá trị có thể của tham số này và dùng vòng lặp for để đánh giá mô hình với trên từng giá trị của tham số. Giá trị được lựa chọn là giá trị mà có MSE trên tập test là nhỏ nhất.

List các giá trị kể trên còn được gọi là không gian tìm kiếm _grid search_.

Tiếp theo chúng ta sẽ tunning hệ số điều chuẩn $\alpha$ cho mô hình hồi qui.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 63DjQ-4TwPK-
outputId: 0bb50388-144a-477a-d49d-5ed825cbbd8f
---
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

grid_search = np.arange(0, 1, 0.05)

def _regression(alpha, X_train, y_train, X_test, y_test, models: dict):
  dict_models = {}
  rid_regr = Ridge(alpha = alpha, normalize = True)
  rid_regr.fit(X_train, y_train)
  y_pred = rid_regr.predict(X_test)
  MSE = np.mean((y_test-y_pred)**2)
  dict_models["MSE"] = MSE
  dict_models["model"] = rid_regr
  model_name = "ridge_" + str(alpha)
  models[model_name] = dict_models
  return models

# Phân chia tập train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
```

+++ {"id": "KTSqwb2P2z9t"}

Huấn luyện mô hình trên grid search.

```{code-cell} ipython3
:id: xDoJb0zg0Km-

models = {}
for alpha in grid_search:
  models = _regression(round(alpha, 2), X_train, y_train, X_test, y_test, models)
```

+++ {"id": "uQAtJCb224WE"}

In kết quả huấn luyện và tìm ra mô hình tốt nhất.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: QW7_MUzj0vbx
outputId: d0edf8d7-f007-425f-9e7b-aee3a7c530ae
---
i = 0
for k, v in models.items():
  if i==0:
    best_model = k
    mse = models[k]["MSE"]
  if models[k]["MSE"] < mse:
    best_model = k

  print("model {}, MSE: {}".format(k, models[k]["MSE"]))
  i+=1

print("-----------------------------------------")
print("Best models: {}, MSE: {}".format(best_model, models[best_model]["MSE"]))
```

+++ {"id": "j0T1sSRR2fqq"}

Vậy mô hình tốt nhất là Ridge Regression với hệ số $\alpha=0$

+++ {"deletable": true, "editable": true, "id": "g6B8XhefoAjx"}

# 9. Tóm tắt

Như vậy ở chương này các bạn đã được học:

1. Phương trình hồi qui tuyến tính đơn biến và hồi qui tuyến tính đa biến.
2. Hàm loss function MSE của hồi qui tuyến tính đơn biến.
3. Các chỉ số đánh giá mô hình hồi qui tuyến tính như `R-squared, MAP, MAPE`
4. Các phương pháp hồi qui tuyến tính với thành phần điều chuẩn như ridge regresssion và lasso regression.
5. Các kỹ thuật visualization kết quả mô hình.
6. Tunning hệ số của mô hình hồi qui.

+++ {"id": "OjNyVfft7u19"}

# 10. Bài tập

+++ {"deletable": true, "editable": true, "id": "3en7L6GSoAkE"}

Từ bộ dữ liệu lưu lượng hành khách sử dụng dịch vụ hàng không qua các năm tại [international airline passengers](https://raw.githubusercontent.com/phamdinhkhanh/LSTM/master/international-airline-passengers.csv) bạn hãy:

1. Phân chia tập train/test sao cho tập test bao gồm 12 tháng cuối cùng và tập train gồm các tháng trước đó.
2. Xây dựng phương trình dự báo lưu lượng hành khách theo phương trình hồi qui tuyến tính đơn biến trên tập train và đánh giá MSE trên tập test.
3. Tạo thêm các biến $x^2, x^3$ và xây dựng phương trình hồi qui tuyến tính đa biến.
4. Huấn luyên mô hình với Ridge Regression và Lasso Regression. Fine tunning hệ số $\alpha$ của thành phần điều chuẩn.
