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

(sec_pandas_profiling)=
# Pandas profiling

[Pandas-profiling](https://github.com/pandas-profiling/pandas-profiling) là một công cụ hữu hiệu cho việc làm EDA cơ bản. Nó giúp bạn tạo ra một trang html có thể lưu trữ và truyền tải một cách nhanh gọn.
Pandas-profiling giúp bạn có cái nhìn toàn cảnh về bộ dữ liệu, phân phối của từng cột cũng như độ tương quan giữa các cột. Sử dụng công cụ này sẽ giúp bạn giảm rất nhiều thời gian EDA.

```{margin}
Hiện pandas-profiling chỉ hỗ trợ EDA cho từng bảng dữ liệu dưới dạng pandas dataframe. Nếu  bộ dữ liệu của bạn có nhiều bảng liên kết với nhau, bạn sẽ cần thêm các bước xử lý thủ công trước khi có thể sử dụng công cụ này. Bạn đọc sẽ thấy thêm các ví dụ về EDA ở phần sau của cuốn sách.
```

Dưới đây là một đoạn code ngắn giúp làm EDA cho bộ dữ liệu Titanic sử dụng pandas-profiling.

```{code-cell} ipython3
%%capture
import pandas as pd
from pandas_profiling import ProfileReport

df_train = pd.read_csv("../data/titanic/train.csv")

profile = ProfileReport(
    df_train, title="Pandas Profiling Report for Titanic train dataset"
)
profile.to_file("../data_to_web/titanic_train_profiling.html")
```

Bản kết quả có thể được tìm thấy [tại đây](https://machinelearningcoban.com/tabml_book/data_to_web/titanic_train_profiling.html).

Dưới đây là một số biểu đồ đáng chú ý.

```{figure} imgs/titanic_overview.png
---
name: img_titanic_overview
---
Titanic Overview
```

{numref}`img_titanic_overview` thể hiện khái quát những thống kê về bộ dữ liệu.
Có nhiều thông tin hữu ích như số lượng trường dữ liệu (_Number of variables_), số mẫu dữ liệu (_Number of observations_), số giá trị bị khuyết (_Missing cells_), số mẫu dự liệu bị lặp (_Duplicate rows_), số cột dạng hạng mục (_Categorical_) và số cột dạng số (_Numeric_).

```{note}
Lưu ý rằng khi không chỉ định rõ kiểu dữ liệu khi đọc file csv, pandas tự suy ra kiểu dựa trên giá trị của dữ liệu trong cột. Ở đây, `PassengerID` và `PClass` cũng được tính là dạng số vì chúng mang những giá trị có thể chuyển đổi ra số. Nếu bạn muốn chỉ định rõ, bạn cần sử dụng tham số `dtype`.
```

```{figure} imgs/titanic_warnings.png
---
name: img_titanic_warnings
---
Overview Warnings
```

{numref}`img_titanic_warnings` đưa ra những cảnh báo khái quát về những trường dữ liệu có chứa giá trị khuyết (_Missing_), có số lượng hạng mục cao (_High cardinality_), có nhiều giá trị bằng 0 (_Zeros_) cũng như chỉ chứa những giá trị riêng biệt (_Unique_).

```{figure} imgs/titanic_age.png
---
name: img_titanic_age
---
Thống kê cột `Age`
```

Phần tiếp theo của bảng kết quả mang lại nhiều thông tin về phân phối của từng cột dữ liệu. {numref}`img_titanic_age` mô tả các thống kê về cột dữ liệu `Age`. Rất nhiều thông tin bạn có thể tìm thấy ở đây. 


Ở gần dưới cùng có mục "Correlations". Mục này minh họa độ tương quan giữa các cột dự liệu.
Có bốn phương pháp tính độ tương quan giữa các cột với thông tin có thể tìm thấy tại nút "Toggle correlation description". Kết quả của mỗi phương pháp được thể hiện ở ma trận tương quan trong mỗi tab.
 
Các tab "Pearson's, Spearman's, Kendall's" thể hiện sự tương quan giữa các cột dạng số. Tab "Phik" thể hiện cho tất cả cả cột và tab "Cramer's" dành cho các cột dạng hạng mục.

```{figure} imgs/titanic_pearson.png
---
name: img_titanic_pearson
---
Ma trận tương quan Pearson's
```

{numref}`img_titanic_pearson` cho ta thấy rằng cột "Pclass" và cột "Fare" có tương quan ngược với hệ số tương quan màu cam đậm, tức là "Pclass" càng nhỏ thì "Fare" càng lớn và ngược lại. Điều này dễ hiểu vì hạng ghế và giá vé có quan hệ chặt chẽ với nhau.


```{figure} imgs/titanic_phik.png
---
name: img_titanic_phik
---
Ma trận tương quan Phik
```

Cuối cùng, mở tab "Phik" như trong {numref}`img_titanic_phik` để tìm độ tương quan giữa các cột và cột nhãn "Survived", ta thấy rằng cột "Sex" có màu đậm nhất, tức có độ tương quan cao nhất.
Điều này chỉ ra rằng các đặc trưng liên quan đến giới tính sẽ cho kết quả tốt.
Đây có thể là lý do mà ban tổ chức cho một file nộp bài mẫu với kết quả dự đoán dựa trên giới tính của hành khách.

Bạn đọc có thể xem thêm kết quả của bộ dữ liệu California Housing [tại đây](https://machinelearningcoban.com/tabml_book/data_to_web/california_housing_profiling.html).

```{code-cell} ipython3
:tags: [hide-input]

%%capture
import pandas as pd
from pandas_profiling import ProfileReport

df_housing = pd.read_csv("../data/california_housing/housing.csv")

profile = ProfileReport(
    df_housing, title="Pandas Profiling Report for California Housing train dataset"
)
profile.to_file("../data_to_web/california_housing_profiling.html")
```

------------
Các ma trận tương quan này rất hữu ích trong việc chọn ra các cột quan trọng trong việc xây dựng mô hình đầu tiên cho mỗi bài toán. Bạn có thể xem tất cả các tab để chọn ra những cột đó.
Nên nhớ cần kiểm tra xem có hiện tượng rò rỉ dữ liệu (_data leakage_) hay không.

_Trong mục tiếp theo, chúng ta sẽ tìm hiểu các kỹ thuật làm sạch dữ liệu và tạo đặc trưng trước khi xây dựng mô hình._
