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

# EDA cho dữ liệu Titanic

Chúng ta cùng làm quen với bộ dữ liệu Titanic.
Bộ dữ liệu này gồm có ba file:

```{code-cell} ipython3
!ls ../data/titanic
```

Cùng xem nhanh dữ liệu trong ba file này bằng cách hiển thị các dòng đầu tiên của mỗi file bằng phương thức `head()` trong `pandas`.

```{code-cell} ipython3
import pandas as pd
df_train = pd.read_csv("../data/titanic/train.csv")
df_train.head(5)
```

```{code-cell} ipython3
df_test = pd.read_csv("../data/titanic/test.csv")
df_test.head(5)
```

```{code-cell} ipython3
df_sub = pd.read_csv("../data/titanic/gender_submission.csv")
df_sub.head(5)
```

Chúng ta có thể thấy nhanh rằng:

* File `train.csv` và `test.csv` có tập hợp các cột với tên gần như nhau, ngoài trừ việc cột `"Survived"` không xuất hiện ở file `test.csv`. Bài toán đặt ra là dùng các cột còn lại của file `train.csv` để huấn luyện một mô hình sao cho nó có thể dự đoán được cột `"Survived"` này dựa trên những cột của file `test.csv`.

* File `gender_submission.csv` chỉ có hai cột `"PassengerID"` và `"Survived"`; đây là file nộp bài mẫu mà người chơi cần hoàn thiện. Cột `"PassengerID"` bao gồm những mã số hành khách có trong tập `test.csv` trong khi cột `"Survived"` chứa các giá trị dự đoán mà người chơi cần thay thế. Các giá trị mẫu này tương ứng với việc dự đoán chỉ có giới tính `"female"` là sống sót. Đây có thể coi là một giải pháp nền (_baseline_) cho bài toán khi chỉ sử dụng một đặc trưng duy nhất là `"Sex"`.

* Cột `"Cabin"` trong hai file dữ liệu có những giá trị bị khuyết.

## Ý nghĩa của từng trường thông tin

Trước khi đi tìm hướng giải quyết bài toán, chúng ta cần biết ý nghĩa của các cột còn lại (được tìm thấy tại [trang web cuộc thi](https://www.kaggle.com/c/titanic/data):

* `"Pclass"`: hạng ghế. 1 = hạng _Upper_, 2 = hạng _Middle_, 3 = hạng _Lower_. Như vậy, trường thông tin `"Pclass"` vừa có thể coi là một đặc trưng hạng mục, vừa có thể coi là một đặc trưng dạng số vì nó có thứ tự. Đặc trưng này khả năng ảnh hưởng tới khả năng sống sót của hành khách vì hạng sang hơn có thể có các biện pháp an toàn tốt hơn (hoặc cũng có thể ngược lại là chủ quan hơn).

* `"Sex"`: giới tính hành khách.

* `"Age"`: tuổi của hành khách. Nếu tuổi nhỏ hơn 1 thì ở dạng số lẻ (0.42), nếu tuổi là ước lượng thì ở dạng xx.5. Rõ ràng đây cùng sẽ là một đặc trưng tiềm năng để dự đoán kết quả cho bài toán vì trẻ em và người già ở vào nhóm có nguy cơ cao hơn.

* `"Sibsp"`: số lượng anh chị em hoặc vợ/chồng cùng ở trên tàu.

* `"Parch"`: số lượng bô mẹ/con cái cùng ở trên tàu.

* `"Ticket"`: mã số vé.

* `"Fare"`: giá vé.

* `"Cabin"`: mã số cabin.

* `"Embarked"`: Nơi lên tàu, `C` = Cherbourg, `Q` = Queenstown, `S` = Southamton. 

Trong những thông tin trên, chúng ta có thể thấy có những thông tin ở dạng số như `Age, Fare, Parch, Sibsp`, có những thông tin ở dạng hạng mục như `Pclass, Sex, Ticket, Cabin, Embarked`. Đánh giá ban đầu có thể cho ta nhận định rằng có những thông tin có thể hữu ích cho việc xây dựng mô hình như `Pclass, Age, Parch, Sibsp` và những thông in có thể ít hữu ích hơn như `Cabin, Embarked, Ticket, Fare`.

## Một vài thống kê

Để có cái nhìn nhanh về thống kê của mỗi trường thông tin dạng *số*, phương thức `describe()` có thể được sử dụng:

```{code-cell} ipython3
import pandas as pd
df_train = pd.read_csv("../data/titanic/train.csv")
df_train.describe()
```

Một vài quan sát với **tập huấn luyện** này:

* `"PassengerID", "Pclass"` mặc dù là các thông tin dạng hạng mục, chúng vẫn được liệt kê ở đây vì khi không chỉ định cụ thể, các trường thông tin mà toàn bộ các giá trị có thể chuyển đổi về số được coi là thông tin dạng số.

* Ở mỗi trường thông tin, các thống kê được chỉ ra cho các giá trị trong trường đó là:
    * `count`: số lượng phần tử _không bị khuyết_,
    * `mean`: giá trị trung bình,
    * `std`: phương sai của,
    * `min`: giá trị nhỏ nhất,
    * `max`: giá trị lớn nhất,
    * `50%`: trung vị -- giá trị mà ở đó có đúng một nửa số phần tử trong cột có giá trị nhỏ hơn hoặc bằng nó.
    * `25%`: trung vị của các giá trị từ `min` tới `50%`, tức có đúng 25% số phần tử trong cột có giá trị nhỏ hơn hoặc bằng nó,
    * `75%`: trung vị của các giá trị từ `50%` tới `max`, tức có đúng 75% số phần tử trong cột có giá trị nhỏ hơn hoặc bằng nó,
    
* Với cột `Survived`, giá trị trung bình trong cột là `0.384`. Đây là cột _nhãn_ mà mô hình cần dự đoán. Cột này chỉ mang các giá trị 0 và 1 nên ta có thể nói rằng 38.4% giá trị trong cột bằng 1. Việc này chứng tỏ dữ liệu tương đối cân bằng giữa hai lớp 0 và 1.

* Với cột `Age`, ta thấy rằng `count = 714` và nhỏ hơn số lượng phần từ ở các cột còn lại (891). Việc này chứng tỏ có tới 891 - 714 = 177 mẫu dữ liệu có `Age` bị khuyết. Người nhỏ nhất trên tàu mới chỉ 0.42 tuổi, trong khi người nhiều tuổi nhất đã 80.

* Với cột `Sibsp`, số lượng anh chị em hoặc vợ/chồng nhiều nhất với một hành khách là 8, nhưng có tới 75% số hành khách có nhiều nhất 1 anh chị em hoặc vợ/chồng đi cùng. Việc này chứng tỏ phân bố của dữ liệu này khá lệch (_skewed_).

* Cột `Parch` cũng bị lệch tương tự khi có một hành khách có tới 6 con/bố mẹ trong khi 75% số hành khách không có con/bố mẹ đi cùng.

* Cột `Fare` cũng khá lệch khi trung binh là 32 trong khi trung vị chỉ là 14 và giá tri lớn nhất lên tới 512. Những hành khách với giá vé bằng 0 khả năng nằm trong thủy thủ đoàn.

```{note}
Khi một cột có những giá trị bị khuyết, thống kê của cột được tính dựa trên các giá trị còn lại.
```

Với **tập kiểm tra**:

```{code-cell} ipython3
df_test = pd.read_csv("../data/titanic/test.csv")
df_test.describe()
```

Một vài quan sát:

* Số lượng phần tử trong tập này là 418 (bằng `count` trong cột `PassengerID`).

* Các cột `Age, Fare` có nhiều giá trị bị khuyết. Như vậy, mặc dù tập huấn luyện không có giá trị `Fare` nào bị khuyết, tập kiểm tra có một hàng bị khuyết giá trị này.

* Các thống kê trong các cột `Age, SibSp, Parch` và `Fare` tương đối nhất quán với tập huấn luyện.

-------------

Đây là một bộ dữ liệu nhỏ với chỉ hơn 1000 mẫu trong cả hai tập huấn luyện và kiểm tra.
Khi dữ liệu lơn hơn, chúng ta cần có cái nhìn bao quát hơn về dữ liệu thông qua các bảng thống kê của từng trường thông tin.


Vì pandas thường cần load toàn bộ file vào RAM nên nó không phù hợp với các bộ dữ liệu lớn.
Với dữ liệu lớn, mời bạn đọc thêm về [dask](https://dask.org/), [modin](https://modin.readthedocs.io/en/latest/) với cú pháp tương tự pandas hoặc [pyspark](https://spark.apache.org/docs/latest/api/python/) cho việc xử lý dữ liệu trên các hệ phân tán.