---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Làm sạch dữ liệu titanic

```{code-cell} ipython3
%%capture
!rm -rf nb_data/titanic; mkdir -p nb_data/titanic
!pip install kaggle;
!kaggle competitions download -c titanic -p nb_data/titanic;

```

Unzip `titanic.zip`


```{code-cell} ipython3
!cd nb_data/titanic; unzip titanic.zip; cd ../../
```


Sau khi giải nén, thư mục `nb_data/titanic` có ba file `.csv` như trên. Trong ba file này, `train.csv` là dữ liệu dược dùng để huấn luyện, `test.csv` là dữ liệu cần dự đoán, và `gender_submision.csv` là file nộp kết quả mẫu.


```{code-cell} ipython3
import pandas as pd
df_train = pd.read_csv("nb_data/titanic/train.csv")
df_train.info()
```


Phương thức `.info()` trả về thông tin sơ bộ của `df_train`. Thông tin cụ thể về từng trường dữ liệu có thể tìm thấy tại trang [dữ liệu cuộc thi](https://www.kaggle.com/c/titanic/data). Từ kết quả bên trên ta thấy được:

1. Có 891 điểm dữ liệu (891 hàng).
2. Các trường `Age, Cabin, Embarked` có ít hơn 891 giá trị `Non-Null`. Điều này có nghĩa là các trường này có dữ liệu bị khuyết.
3. Các trường `Name, Sex, Ticket, Cabin, Embarked` có kiểu dữ liệu `Dtype` là `object`, tức các trường này được lưu ở dạng `string`. Các trường còn lại có giá trị là số nguyên (`int64`) hoặc số thực (`float64`). Chú ý rằng chúng ta không nên ngầm hiểu các trường này thực sự có ý nghĩa là số, ở đây vì chúng ta không chỉ rõ kiểu dữ liệu khi đọc file `.csv` bằng `pd.read_csv` nên `pandas` tự suy ra kiểu dữ liệu. Trong bộ dữ liệu này mỗi `PasengerId` là một mã hành khách, có thể là một giá trị bất kỳ không lặp lại. Chúng ta cần xác định rõ kiểu của từng trường dữ liệu, mô hình ML thường có những cách xử lý khác nhau giữa kiểu dữ liệu dạng số và dạng hạng mục.

Tiếp theo, để có cái nhìn nhanh về các trường dữ liệu dạng số, thuộc tính `describe()` được sử dụng:


```{code-cell} ipython3
df_train.describe()
```




Kết quả trả về giúp chúng ta có cái nhìn khái quát về phân phối của các trường dữ liệu dạng số. Một vài quan sát:

1. Cột `Survived` là cột chứa nhãn của mỗi hàng. Giá trị nhỏ nhất bằng 0 và giá trị lớn nhất bằng 1, cùng với việc kiểu dữ liệu là `int64`, ta có thể nói rằng kiểu dữ liệu của trường này là nhị phân [^1]. Ở đây, giá trị `1` mang nghĩa là sống sót và `0` mang nghĩa ngược lại. Giá trị trung bình `mean = 0.3838` cũng giúp chỉ ra rằng khoảng 38.4% dữ liệu thuộc class `1` so với khoảng 61.6% dữ liệu thuộc class `0`. Dữ liệu có bị lệch nhưng không quá nghiêm trọng.

2. Cột 
 


[^1]: Khi không xác định rõ kiểu dữ liệu, `pandas` tự chuyển kiểu nhị phân về `int64`, số thực về `float64`. Với các tập dữ liệu lớn, việc không xác định rõ kiểu khi đọc dữ liệu có thể dẫn tới tình trạng tràn bộ nhớ.

