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

_Phạm Đình Khánh_

# Dữ liệu chuỗi thời gian

## Khái niệm dữ liệu chuỗi thời gian

Trước khi trả lời cho câu hỏi dữ liệu chuỗi thời gian là gì chúng ta sẽ cùng ôn lại những kiểu dữ liệu chính để từ đó biết được dữ liệu chuỗi thời gian sẽ thuộc nhóm dữ liệu nào và có đặc tính là gì. Khi xây dựng mô hình, bạn có thể làm việc với hai kiểu dữ liệu chính:

* Dữ liệu có cấu trúc (_SQL_): Là dữ liệu đã được tổ chức dưới dạng bảng tabular. Một ví dụ thực tiễn nhất về dữ liệu dạng này là _data table_ trên bảng tính excel. Mỗi một trường trong bảng tabular sẽ được định nghĩa trước định dạng số, chuỗi ký tự hoặc thời gian và diễn tả một chiều thông tin của quan sát trên thực tế . Thông thường ở những doanh nghiệp lớn, dữ liệu có cấu trúc sẽ được tổng hợp trong một nhà kho dữ liệu (_dataware house_) để đáp ứng nhu cầu phân tích kinh doanh và xây dựng mô hình.

* Dữ liệu phi cấu trúc (_No SQL_): _No SQL_ là viết tắt của _Non Structured Query Language_. Đây là kiểu dữ liệu phổ biến nhất trong ba kiểu dữ liệu. Những dữ liệu này chưa được qui hoạch và tổ chức dưới dạng những tabular có quan hệ mà trái lại nó tồn tại ở dạng data thô chưa qua xử lý như văn bản, biểu đồ, các dữ liệu dạng _key: value_. Dữ liệu _No SQL_ linh hoạt hơn _SQL_ vì nó có thể lưu trữ nhiều loại trường dữ liệu hơn so với _SQL_. Tốc độ truy vấn và xử lý trên _No SQL_ cũng nhanh hơn so với _SQL_. Tuy nhiên khi áp dụng vào xây dựng mô hình thì chúng ta vẫn cần phải chuyển dữ liệu về dạng _SQL_ để xử lý.

Trong toán học, dữ liệu chuỗi thời gian được định nghĩa là những điểm dữ liệu đã được đánh indice theo thời gian và có khoảng cách điều nhau giữa những quan sát liên tiếp. Đó có thể là dữ liệu về giá chứng khoán hàng ngày, tổng thu nhập quốc dân của một quốc gia hàng năm, tổng doanh số công ty hàng quí,....

Ưu điểm của chuỗi thời gian là nó có thể lưu trữ được trạng thái của một trường dữ liệu theo thời gian. Trong khi đó thế giới luôn vận động, các sự vật, hiện tượng hiếm khi dừng lại ở trạng thái tĩnh mà thường thay đổi. Do đó dữ liệu chuỗi thời gian có tính ứng dụng rất cao và được áp dụng trong rất nhiều lĩnh vực khác nhau như: _thống kê, kinh tế lượng, toán tài chính, dự báo thời tiết, dự đoán động đất, điện não đồ, kỹ thuật điều khiển, thiên văn, kỹ thuật truyền thông, xử lý tín hiệu_. Dữ liệu chuỗi thời gian cho phép các quốc qia trên thế giới hàng năm đưa ra dự báo tăng trưởng GDP của mình và các doanh nghiệp dự báo doanh số và triển vọng thị trường. Chính vì thế dữ liệu chuỗi thời gian đóng một vai trò cực kỳ quan trọng đối với sự phát triển của nhân loại.

## Đặc điểm của chuỗi thời gian

Dữ liệu chuỗi thời gian có những tính chất đặc trưng riêng như:

* Tính xu hướng: Tính xu hướng là yếu tố thể hiện xu hướng thay đổi của dữ liệu theo thời gian. Đây là đặc trưng thường thấy của rất nhiều dữ liệu chuỗi thời gian. Đặc biệt là các chuỗi trong kinh tế lượng như: giá cả thị trường chị ảnh hưởng của lạm phát, dân số thế giới tăng qua các năm, nhiệt độ trung bình trái đất tăng theo thời gian do hiệu ứng nhà kính,.... Tính xu hướng cũng ảnh hưởng không nhỏ tới việc đưa ra nhận định về mối quan hệ tương quan giữa các chuỗi số. Tức là về bản chất các chuỗi không tương quan nhưng do chúng cùng có chung xu hướng theo thời gian nên chúng ta nhận định chúng là tương quan. Ví dụ: Số lượng người bị đuối nước hàng năm và sản lượng kem tiêu thụ có mối quan hệ cùng chiều (hay còn gọi là _tương quan tuyến tính dương_). Không khó để chúng ta nhận định được bản chất của sự tương quan này là do chúng có cùng sự tương quan với nhiệt độ. Khi nhiệt độ tăng lên chúng ta đi tắm biến nhiều hơn và dẫn tới số lượng người bị đuối nước cao hơn và đồng thời khi nhiệt độ cao cũng là lúc người ta ăn kem để giải khát nhiều hơn. Tuy nhiên việc ăn kem không phải là nguyên nhân trực tiếp dẫn tới đuối nước. Do đó khi xây dựng các mô hình chuỗi thời gian chúng ta cần loại bỏ yếu tố xu hướng ở những biến input để tìm ra những chuỗi có sự tương quan thực sự. 

* Tính chu kỳ: Là qui luật có tính chất lặp lại của dữ liệu theo thời gian. Sự thay đổi thời tiết, sự phát triển của các loài động vật cho tới hành vi mua sắm, tiêu dùng của con người đều bị ảnh hưởng của chu kỳ và lặp lại theo thời gian. Chính vì thế tìm ra được yếu tố chu kỳ sẽ giúp ích cho việc dự báo chính xác hơn. Một ví dụ về tầm quan trọng của chu kỳ đó là các doanh nghiệp sản xuất một mặt hàng cụ thể sẽ biết sản lượng tăng vào thời điểm nào trong năm? Cần phải tuyển thêm bao nhiêu lao động? Mua thêm bao nhiêu nguyên vật liệu để đáp ứng được nhu cầu thị trường. Nếu không hiểu được tính chu kỳ của chuỗi thời gian, doanh nghiệp có thể dự báo sai nhu cầu thị trường và dẫn tới thua lỗ.

Vì những tính chất riêng biệt nên khi làm việc với dữ liệu chuỗi thời gian chúng ta cần quan sát và xử lý dữ liệu tốt. Ở chương tiếp theo chúng ta sẽ cùng tập trung phân tích sâu vào các phương pháp xử lý chuỗi thời gian.

## Phát hiện outliers

Outliers là hiện tượng thường xuyên bắt gặp trong chuỗi thời gian. Chúng xuất hiện trong chuỗi thời gian có thể do nhiều nguyên nhân khác nhau như do sự cố, cú sốc kinh tế, các ngày nghỉ lễ, thiên tai, dịch bệnh,.... Nếu không xử lý outliers tốt có thể dẫn tới mô hình dự báo bị chệch vì outliers nằm ngoài qui luật có thể dự báo được của mô hình. Chính vì vậy, trước khi làm việc với dữ liệu chuỗi thời gian chúng ta cần kiểm tra và xử lý outliers bằng cách loại bỏ hoặc thay thế. Để phát hiện outliers, chúng ta có thể thông qua đồ thị hoặc định lượng. Ưu điểm của phương pháp đồ thị là có thể thực hiện nhanh, nhưng nhược điểm đó là nó là một cách xác định mang tính tương đối mà không căn cứ trên một tiêu chuẩn cụ thể như định lượng.

**Biểu đồ line**: Biểu đồ line thường được sử dụng bởi vì nó liên kết các điểm theo thứ tự thời gian, do đó việc phát hiện các điểm nằm ngoài xu hướng chung của chuỗi dễ dàng hơn. Những điểm này có thể được xem như là outliers. 

Bên dưới chúng ta cùng thực hành theo dõi biểu đồ đường của dữ liệu doanh số dầu gội (_shampoo_) trong 36 tháng. Điểm dữ liệu tại tháng _05-2002_ đã được tăng thêm 500 đơn vị để nhằm tạo ra một outlier giả định.

```{code-cell} ipython3
import pandas as pd
from datetime import datetime

df = pd.read_csv("https://raw.githubusercontent.com/phamdinhkhanh/tabml/main/sales-of-shampoo-over-a-three-ye.csv", header=0)
df["Month"] = df['Month'].apply(lambda x: pd.to_datetime("200"+x))
df.set_index("Month", inplace=True)
df.describe()
```

Thống kê mô tả cho thấy trung bình mỗi tháng doanh nghiệp bán được 312.6 đơn vị sản phẩm, tháng thấp nhất là 119.3 và cao nhất là 682. Độ lệch so với trung bình của các tháng là 148.94 đơn vị. Đây là chuỗi có xu hướng phân phối lệch về bên phải vì _median_ (mức phân vị 50%) lớn hơn _mean_.

Để vẽ đồ thị đường trên python thì chúng ta sử dụng package _matplotlib_. Các bạn có thể tham khảo tại [Bài 11 - Visualization trong python](https://phamdinhkhanh.github.io/2019/09/16/VisualizationPython.html) để xem các cách visualize những biểu đồ cơ bản.

```{code-cell} ipython3
import matplotlib.pyplot as plt

# Create outlier in 2002-05
df.loc["2002-05"]["Sales"] = df.loc["2002-05"]["Sales"]+500

plt.figure(figsize=(10, 6))
plt.plot(df["Sales"], marker="o")
plt.xticks(rotation=45)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Shampoo revenue", fontsize=12)
plt.title("Shampoo revenue in three years", fontsize=18)
plt.show()
```

Nhìn vào biểu đồ trên ta có thể phát hiện ra dữ liệu vào tháng _05-2002_ là outlier vì giá trị của nó vượt xa so với những điểm xung quanh và nằm ra ngoài xu hướng của chuỗi.

**Biểu đồ box-plot**: Xác định outliers theo phương pháp này kết hợp giữa đồ thị và định lượng. Cụ thể biểu đồ box-plot xác định các ngưỡng chia là: mức phân vị 25% (là giá trị mà có 25% quan sát nhỏ hơn giá trị này), median, mức phân vị 75% được ký hiệu lần lượt là $Q_1, Q_2, Q_3$. Như vậy median sẽ chính là mức phân vị 50%. Các miền giá trị được xác định từ các ngưỡng trên sẽ tương ứng với:

* Miền giá trị thấp nhất: Chứa các quan sát từ thấp nhất (không bao gồm outliers) tới $Q_1$.

* Miền gía trị thấp: Chứa các quan sát nằm trong khoảng từ $[Q_1, Q_2)$.

* Miền giá trị cao: Chứa các quan sát trong khoảng $[Q_2, Q_3)$.

* Miền giá trị cao nhất: Chứa các quan sát trong từ $Q_3$ tới giá trị cao nhất (không bao gồm outliers).

Chỉ số IQ được xác định chính bằng độ dài của hai miền giá trị cao và thấp.

$$IQR = Q_3-Q_1$$

Dựa trên IQR, outliers sẽ được xác định nếu nằm ngoài miền giá trị cao nhất và miền giá trị thấp nhất. Tức là nếu $x > Q_3 + 1.5*IQR$ hoặc $x < Q_1 - 1.5*IQR$ thì sẽ được xác định là outliers.

Chúng ta cần loại bỏ yếu tố xu hướng là vì đây là một chuỗi có trend nên các giá trị rất thấp ở đầu chuỗi và rất cao ở  cuối chuỗi rất dễ bị nhận nhầm thành outliers do ảnh hưởng của trend. Loại bỏ trend sẽ giúp cho ta đánh giá được đâu mới thực sự là outliers mà không phải là do trend.

Chúng ta có thể lấy sai phân bậc 1 để đánh loại bỏ trend.

```{code-cell} ipython3
:id: N4OU12DxIQW-

diff_sales = df["Sales"][1:].values - df["Sales"][:-1].values
```

Vẽ biểu đồ boxplot

```{code-cell} ipython3
plt.figure(figsize=(10, 6))
plt.boxplot(diff_sales)
plt.ylabel("Shampoo revenue", fontsize=12)
plt.title("Boxplot of Shampoo revenue in three years", fontsize=18)
plt.show()
```

Như vậy chúng ta xác định rằng xuất hiện hai điểm outliers ở miền giá trị thấp nhất và miền giá trị cao nhất. Chúng ta sẽ tìm ra indice của hai điểm này như sau:

```{code-cell} ipython3
import numpy as np

def detect_outliers(series):
  """
    series: 1-D numpy array input
  """
  Q1 = np.quantile(series, 0.25)
  Q3 = np.quantile(series, 0.75)
  IQR = Q3-Q1
  lower_bound = Q1-1.5*IQR
  upper_bound = Q3+1.5*IQR
  lower_compare = series <= lower_bound
  upper_compare = series >= upper_bound
  outlier_idxs = np.where(lower_compare | upper_compare)[0]
  return outlier_idxs

outlier_idxs=detect_outliers(diff_sales)
print("Outlier indices: ", outlier_idxs)
print("Outlier months: ", df.index[outlier_idxs+1].values)
print("Outlier values: ", diff_sales[outlier_idxs])
```

Như vậy từ bbox chúng ta có thể kết luận được các tháng 05/2002 và 06/2002 là những tháng xuất hiện outliers.

## Phát hiện yếu tố chu kỳ và mùa vụ

Yếu tố chu kỳ và mùa vụ là một thông tin quan trọng vì nó thể hiện những qui luật lặp lại tác động tới chuỗi thời gian. Tìm ra được qui luật của chuỗi thời gian theo chu kỳ và mùa vụ sẽ giúp cho chúng ta dự báo chính xác hơn. Để phát hiện ra qui luật này chúng ta có thể thông qua quan sát biểu đồ bằng mắt thường hoặc sử dụng các chỉ số định lượng về tự tương quan.

Đối với phương pháp biểu đồ thì chúng ta cần quan sát qui luật đó trong một khoảng thời gian dài để nhận định được củng cố hơn. Các qui luật mùa vụ sẽ lặp lại theo một số chu kỳ phổ biến chẳng hạn như:

* 4 quí trong 1 năm: Thường thể hiện ở các chuỗi như GPD, kim ngạch xuất nhập khẩu,....
* 12 tháng trong 1 năm: Các chuỗi liên quan tới doanh thu, doanh số, du lịch, dịch vụ,....
* 7 ngày trong 1 tuần: Các chuỗi liên quan tới qui luật mua sắm, tiêu dùng, vui chơi giải trí,....

Khi phán đoán qui luật mùa vụ sẽ cần hiểu về chuỗi mà chúng ta đang phân tích thuộc ngành nghề, lĩnh vực nào, từ đó có thể đưa ra nhận định sơ bộ qui luật của chuỗi và kiểm chứng lại trên đồ thị.

Đối với phương pháp định lượng chúng ta có thể dựa trên hệ số tự tương quan.



```{code-cell} ipython3
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc("figure", figsize=(10, 6))
plot_acf(df["Sales"], lags = 12)
```

Trục $y$ đại diện cho hệ số tương quan và trục $x$ đại diện cho độ trễ. Độ dài của các thanh vuông góc với trục $x$ thể hiện giá trị của hệ số tương quan tương ứng với độ trễ đó (tạm gọi là _hệ số tương quan trễ _) theo công thức sau:

$$\rho(l) = \frac{cov(x_t,x_{t-l})}{\sqrt{\sigma_t\sigma_{t-l}}}$$

Ở đây $l$ là độ trễ và $x_{t-l}$ là giá trị trễ $l$ phiên của chuỗi $x_t$.

![](https://imgur.com/zhl44t8.png)

Khi qui luật mùa vụ lặp lại với chu kỳ $S$ thì hệ số tương quan theo các độ trễ $s, 2s, 3s, \dots $ sẽ lớn hơn những hệ số tương quan với các độ trễ khác. Do đó chúng ta chỉ cần tìm khoảng cách về độ trễ giữa các đỉnh cực trị địa phương trong đồ thị pacf là có thể tìm ra qui luật mùa vụ.

Chẳng hạn trong đồ thị trên ta thấy các độ trễ có tự tương quan lớn nhất là 1, 5, 9. Do đó chuỗi có chu kỳ mùa vụ lặp lại bằng khoảng cách của các độ trễ trên và bằng 4.

## Xử lý dữ liệu khuyết

Khi dữ liệu chuỗi thời gian bị khuyết chúng ta sẽ sử dụng hai phương pháp chính để xử lý dữ liệu missing đó là nội suy tuyến tính _interpolate_ và ngoại suy tuyến tính _extrapolate_.

Phương pháp nội suy tuyến tính được sử dụng trong tình huống bạn muốn điền dữ liệu khuyết **nằm trong khoảng** giữa hai đầu mút.

![](https://imgur.com/bdqmTzO.png)

Trong hình trên bạn đã biết $(x_1, y_1)$ và $(x_2, y_2)$ tương ứng với điểm bắt đầu và điểm kết thúc. Bạn cần nội suy giá trị của $y$ tại thời điểm $x \in (x_1, x_2)$ theo công thức sau:

$$y = y_1 + (x-x_1) \times \frac{y_2-y_1}{x_2-x_1}$$


Trong tình huống bạn muốn biết giá trị của một điểm dữ liệu **nằm ngoài khoảng** thời gian của hai điểm đầu mút thì công thức trên được gọi là ngoại suy tuyến tính.


![](https://imgur.com/RqGuPoF.png)


Bên dưới ta có giả định chuỗi shapoo bị khuyết quan sát tại $indice=1$. Chúng ta có thể sử dụng nội suy tuyến tính để điền giá trị khuyết như sau:

```{code-cell} ipython3
import numpy as np

def _interpolate(p1, p2, x):
  """
  p1: start point (x1, y1)
  p2: end point (x2, y2)
  """
  x1, y1 = p1
  x2, y2 = p2
  y = y1 + (x-x1)*(y2-y1)/(x2-x1)
  return y
  
# Missing at indice=1 
df["Sales"].iloc[1] = np.nan

p1 = (0, df["Sales"].iloc[0])
p2 = (2, df["Sales"].iloc[2])
x = 1
print(p1, p2)
print("interpolation value at x=1: ", _interpolate(p1, p2, 1))
```

Trong pandas đã tích hợp sẵn các hàm nội suy tuyến tính và ngoại suy tuyến tính.

```{code-cell} ipython3
df["Sales"].iloc[1] = np.nan
df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
df.head()
```

Trong công thức trên khi bạn thay giá trị `limit_direction` bằng `backward` thì phương pháp sẽ trở thành ngoại suy tuyến tính. Ngoài ra chúng ta có thể làm phức tạp hơn phương pháp ngoại suy bằng `method=polinormial`.

## Trung bình trượt

Trung bình trượt là phương pháp làm mịn chuỗi bằng cách lấy trung bình của chuỗi trong một khung thời gian. Trung bình trượt có tác dụng theo dõi giao động trong ngắn hạn và nhận biết được xu hướng trong dài hạn của chuỗi. 

Có 2 dạng trung bình trượt chính chúng ta có thể áp dụng đó là trung bình trượt đơn (_Single Moving Average_) và trung bình trượt cấp số nhân (_Exponential Moving Average_).

Trung bình trượt đơn của chuỗi $x_t$ theo khung thời gian $w$ chính là chuỗi trung bình cộng của các giá trị nằm trong khung thời gian.

$$SMA_t = \frac{\sum_{j=t-w}^{t} x_i}{w}$$

```{code-cell} ipython3
# Single moving average in pandas
df_sma = df.rolling(window=4).mean()
plt.figure(figsize=(10, 6))
plt.plot(df["Sales"], label="x")
plt.plot(df_sma["Sales"], label="sma(4)")
plt.xticks(rotation=45)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Shampoo revenue", fontsize=12)
plt.legend(fontsize=12)
plt.title("Shampoo revenue in three years", fontsize=18)
plt.show()
```

Không giống như trung bình trượt đơn, đường trung bình trượt cấp số nhân phân bố trọng số hơn đối với giá gần đây và do đó, nó có thể là một mô hình tốt hơn và nắm bắt tốt hơn chuyển động của xu thế.

```{code-cell} ipython3
# Caculate exponential moving average
df_ewm = df.ewm(span=12, adjust=False).mean()

plt.figure(figsize=(10, 6))
plt.plot(df["Sales"], label="x")
plt.plot(df_ewm["Sales"], label="ewm(12)")
plt.xticks(rotation=45)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Shampoo revenue", fontsize=12)
plt.legend(fontsize=12)
plt.title("Shampoo revenue in three years", fontsize=18)
plt.show()
```

## Loại bỏ yếu tố xu thế

Như chúng ta đã biết xu thế có thể dẫn tới sự tương quan tuyến tính giả mạo. Do đó để đánh giá được các biến thực tế có tương quan hay không chúng ta cần loại bỏ yếu tố xu thế của chuỗi. Để loại bỏ xu thế chúng ta giả định giá trị của chuỗi tuân theo phương trình tuyến tính với thời gian.

$$x_t = \alpha t + x_t'$$

Khi tìm được hệ số $\alpha$ của phương trình trên ta sẽ tính được chuỗi loại bỏ xu thế $x_t'$.

```{code-cell} ipython3
import numpy as np
from scipy import signal

x_detrend = signal.detrend(df["Sales"])
df_detrend = pd.DataFrame({"Sales_Detrend": x_detrend}, index=df.index)

plt.figure(figsize=(10, 6))
plt.plot(df["Sales"], label="x")
plt.plot(df_detrend, label="x_detrend")
plt.xticks(rotation=45)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Shampoo revenue", fontsize=12)
plt.legend(fontsize=12)
plt.title("Shampoo revenue in three years", fontsize=18)
plt.show()
```

## Phân rã mùa vụ

Mô hình sẽ dự báo chuẩn xác hơn nếu tách riêng yếu tố mùa vụ và dự báo thành phần còn lại không chịu ảnh hưởng bởi mùa vụ.

Để phân rã mùa vụ chúng ta có thể dùng mô hình nhân tính (_addictive model_) hoặc mô hình cộng tính (_multiplicative model_). Mô hình nhân tính phù hợp với những chuỗi có độ biến động ổn định theo thời gian. Chẳng hạn như chuỗi shampoo được ví dụ ở trên. Chúng ta không nhìn thấy một dấu hiệu nào cho thấy trong tương lai biên độ giao động của revenue là tăng hoặc giảm so với quá khứ.

Phương trình hồi qui của mô hình cộng tính là một phép cộng của các thành phần mùa vụ và thời gian: 

$$x_t = T_t + S_t + e_t$$

Trong đó $x_t$ là giá trị chuỗi. $T_t, S_t$ lần lượt đại diện cho ảnh hưởng của yếu tố thời gian và yếu tố mùa vụ. $e_t$ là phần dư hoặc sai số dự báo. Ta thấy trong công thức trên thì ảnh hưởng của $S_t$ lên $x_t$ có hệ số không đổi là 1 nên không tạo ra khác biệt quá lớn trong tương lai lên chuỗi. Do đó lớp mô hình này mới phù hợp với các chuỗi có biến động ổn định.

Trái lại, mô hình nhân tính sẽ phù hợp với những chuỗi có độ biến động gia tăng theo thời gian.

Phương trình hồi qui của mô hình nhân tính sẽ thể hiện qua phép tính nhân: 

$$x_t = T_t * S_t * e_t$$

Như vậy sự ảnh hưởng của $S_t$ lên $x_t$ đã gia tăng theo cấp số $T_t$. Cấp số này sẽ gia tăng theo thời gian nên càng xa trong tương lai thì độ biến động của chuỗi càng có xu hướng gia tăng mạnh hơn.

Tiếp theo chúng ta sẽ lấy ví dụ về hai chuỗi shampoo và chuỗi biến đổi nhân tính của shampoo. Dựa vào đồ thị của hai chuỗi này để đưa ra nhận định nên sử dụng mô hình nhân tính hay mô hình cộng tính để loại bỏ yếu tố mùa vụ.

```{code-cell} ipython3
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/phamdinhkhanh/tabml/main/sales-of-shampoo-over-a-three-ye.csv", header=0)
df["Month"] = df['Month'].apply(lambda x: pd.to_datetime("200"+x))
df.set_index("Month", inplace=True)

# Tạo chuỗi nhân tính
multip = np.arange(df.shape[0])
multip = np.exp(multip/20)
df_mul = df.copy()
df_mul["Sales"] = df_mul["Sales"]*multip

plt.figure(figsize=(10, 6))
plt.plot(df["Sales"], label="x_add")
plt.plot(df_mul["Sales"], label="x_mul")
plt.xticks(rotation=45)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Shampoo revenue", fontsize=12)
plt.legend(fontsize=12)
plt.title("Shampoo revenue in three years", fontsize=18)
plt.show()
```

Trả lời: đường màu xanh nước biển nên sử dụng mô hình nhân tính và được màu xanh lá cây nên sử dụng mô hình cộng tính vì phương sai của đường màu xanh nước biển ổn định theo thời gian trong khi phương sai của đường màu xanh lá cây thay đổi theo thời gian.

Để phân ra thành phần mùa vụ chúng ta có thể sử dụng hàm `seasonal_decompose` của package `statsmodels`. Đối với chuỗi shampoo thì chúng ta sẽ lựa chọn mô hình cộng tính.

```{code-cell} ipython3
from random import randrange
from pandas import Series
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df["Sales"], model='additive')
result.plot()
plt.show()
```

Muốn lấy các thành phần seasonal, trend, residual bạn có thể truy cập thông qua

```{code-cell} ipython3
print(result.resid.shape)
print(result.seasonal.shape)
print(result.trend.shape)
```

## Kết luận

Như vậy ở chương này các bạn đã được giới thiệu về khái niệm chuỗi thời gian, các đặc điểm của chuỗi thời gian và các phương pháp xử lý dữ liệu chuỗi thời gian. Đây là một kiểu dữ liệu rất quan trọng và thường gặp trong thực tiễn phân tích kinh doanh và xây dựng mô hình của doanh nghiệp. Dù đã cố gắng đưa vào các kiến thức mang tính ứng dụng thực tiễn, được đúc rút từ quá trình làm việc của tác giả nhưng cũng không tránh khỏi thiếu sót. Rất mong nhận được đóng góp và phản hồi từ bạn đọc.

## Tham khảo

https://online.stat.psu.edu/stat510/lesson/5/5.1

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html

https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html
