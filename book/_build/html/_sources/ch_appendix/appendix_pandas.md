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

# 2.1. Khởi tạo dataframe

Đây là cách thường ít được áp dụng vì khi làm việc chúng ta thường đọc dữ liệu từ những file dữ liệu có sẵn được lưu dưới dạng `csv` hoặc `txt`. Nhưng đôi khi chúng ta cũng cần khởi tạo dataframe từ đầu chẳng hạn như bạn muốn lưu kết quả log file của chương trình vào một dataframe và save dưới dạng `csv` sau đó. Việc lưu trữ dưới dạng dataframe sẽ giúp cho bạn dễ dàng thực hiện các phép lọc, thống kê và visualize trực tiếp từ dataframe một cách dễ dàng hơn. 

Đưới đây mình sẽ giới thiệu hai cách khởi tạo dataframe chính trực tiếp từ câu lệnh `pd.DataFrame(.)`.



+++ {"id": "XUX-h_E5trFF"}


## 2.1.1. Khởi tạo thông qua dictionary

Về định dạng dictionary chúng ta đã được học ở [chương phụ lục - dictionary](https://phamdinhkhanh.github.io/deepai-book/ch_appendix/appendix_dtypes_basic.html#dictionary). Nội dung của dictionary sẽ gồm key là tên cột và value là list giá trị của cột tương ứng.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 173
id: ki4Ajy-qig-y
outputId: 3040df50-8988-445d-cf9c-be36d4aa86d7
---
import pandas as pd
from IPython.display import display
pd.set_option('max_colwidth', 40)
pd.set_option('precision', 5)
pd.set_option('max_rows', 10)
pd.set_option('max_columns', 30)


dict_columns = {
    'contents':['Author', 'Book', 'Target', 'No_Donation'],
    'infos':['Pham Dinh Khanh', 'ML algorithms to Practice', 'Vi mot cong dong AI vung manh hon', 'Community'],
    'numbers':[1993, 2021, 1, 2]
}

df = pd.DataFrame(dict_columns)
display(df)
```

+++ {"id": "3mZfjB3FmHjn"}

Hàm display của `IPython` giúp cho DataFrame hiển thị được trên code khi run dưới dạng script file. các options của `pd.set_option()` lần lượt có tác dụng:

* `max_colwidth`: Qui định chiều rộng tối đa của một cột.
* `precision`: Độ chính xác của các sau dấu phảy của các cột định dạng float.
* `max_columns`, `max_rows`: Lần lượt là độ số lượng cột và số lượng dòng tối đa được hiển thị.

Tiếp theo chúng ta sẽ khởi tạo thông qua list các dòng.

## 2.1.2. Khởi tạo thông qua list các dòng

Theo cách này chúng ta sẽ truyền vào data là một list gồm các tupple mà mỗi tupple là một dòng dữ liệu. đối số `columns` sẽ qui định tên cột theo đúng thứ tự được qui định ở mỗi dòng. 

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 173
deletable: true
editable: true
id: idzxlT3BSzJe
outputId: 7d4c511a-c142-4bb1-9f3e-62fc6405d5ac
---
import pandas as pd

records = [('Author', 'Pham Dinh Khanh', 1993), 
           ('Book', 'ML algorithms to Practice', 2021), 
           ('Target', 'Vi mot cong dong AI vung manh hon', 1), 
           ('No_Donation', 'Community', 2)]
           
# Khởi tạo DataFrame
df = pd.DataFrame(data = records, columns = ['contents', 'infos', 'numbers'])
df
```

+++ {"id": "oKhh-Uwap4aD"}

Để lưu trữ một dataframe dưới dạng một file `csv` chúng ta dùng hàm `.to_csv(.)` tham số truyền vào là đường link save file. Chẳng hạn bên dưới ta lưu dataframe vào một file "data.csv" cùng thư mục với file notebook.

```{code-cell} ipython3
:id: 4GV9k0OxqFAP

df.to_csv("data.csv")
```

+++ {"id": "Tq6hF0D4pcg7"}

## 2.1.3. Đọc dữ liệu từ file

Chúng ta cũng có thể khởi tạo bảng bằng cách đọc file `csv, txt, xls, xlsx, dat` thông qua hàm `pd.read_csv(.)`. Hàm này không chỉ đọc được những file có trên máy tính của bạn mà còn có thể download những file có trên mạng. Bên dưới chúng ta thực hành đọc dữ liệu về giá nhà ở tại Boston từ bộ dữ liệu `BostonHousing`. Bộ dữ liệu này gồm các trường:


* crim: Tỷ lệ phạm tội phạm bình quân đầu người theo thị trấn.
* zn: Tỷ lệ đất ở được quy hoạch cho các lô trên 25.000 foot square.
* indus: Tỷ lệ diện tích thuộc lĩnh vực _kinh doanh phi bán lẻ_ trên mỗi thị trấn.
* chas: Biến giả, = 1 nếu được bao bởi sông Charles River, = 0 nếu ngược lại.
* nox: Nồng độ khí Ni-tơ oxit.
* rm: Trung bình số phòng trên một căn hộ.
* age: Tỷ lệ căn hộ được xây dựng trước năm 1940.
* dis: Khoảng cách trung bình có trọng số tới 5 trung tâm việc làm lớn nhất ở Boston.
* rad: Chỉ số về khả năng tiếp cận đường cao tốc.
* tax: Giá trị thuế suất tính trên đơn vị `10000$`.
* ptratio: Tỷ lệ học sinh-giáo viên trên mỗi thị trấn.
* black: Tỷ lệ số người da đen trong thị trấn được tính theo công thức:
$1000(\text{Bk} - 0.63)^2$ ở đây $\text{Bk}$ là tỷ lệ người da đen trong thị trấn.
* lstat: Tỷ lệ phần trăm dân số thu nhập thấp.
* medv: median giá trị của nhà có người sở hữu tính trên đơn vị `1000$`.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: l5ZQBcFbcC51
outputId: 855d245c-06dc-43df-ea9a-5c40b1c23847
---
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/phamdinhkhanh/datasets/master/BostonHousing.csv", sep=",", header = 0, index_col = None)
df.head()
```

+++ {"id": "enkuwJmyqOrE"}

Trong hàm `pd.read_csv()` chúng ta sẽ khai báo các thông số chính bao gồm :

* sep: Là viết tắt của seperator, ký hiệu ngăn cách các trường trong cùng một dòng, thường và mặc định là dấu phảy.
* header: Mặc định là indice của dòng được chọn làm column name. Thường là dòng đầu tiên của file. Trường hợp file không có header thì để `header = None`. Khi đó indices cho column name sẽ được mặc định là các số tự nhiên liên tiếp từ 0 cho đến indice column cuối cùng.
* index_col: Là indice của column được sử dụng làm giá trị index cho dataframe. cột index phải có giá trị khác nhau để phân biệt giữa các dòng và khi chúng ta để index_col = None thì giá trị index sẽ được đánh mặc định từ 0 cho đến dòng cuối cùng.

+++ {"id": "lDNmOXEVkFJ0"}

Hàm `df.head()` mặc định sẽ hiển thị ra 5 quan sát đầu tiên của dataframe. Chúng ta muốn hiển thị 5 quan sát cuối cùng thì dùng hàm `df.tail()` và 5 quan sát ngẫu nhiên thì dùng hàm `df.sample(5)`.

+++ {"id": "MtEf_Mosjd1P"}

Hàm `df.info()` sẽ cho ta biết định dạng và số lượng quan sát `not-null` của mỗi trường trong dataframe.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 1EpBPw5EjoCl
outputId: 50f25d60-191a-438d-dd79-6d7bae213ed0
---
df.info()
```

+++ {"id": "99yomrSTkrYX"}

Hoặc chúng ta có thể dùng hàm `df.dtypes` để kiểm tra định dạng dữ liệu các trường của một bảng.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
deletable: true
editable: true
id: BuSwc1UDSzJs
outputId: 2f9f3613-ed14-4084-f381-26a6d424fb05
---
# Check for datatype
df.dtypes
```

+++ {"id": "h3WBoFA_s_fZ"}

Nếu muốn kiểm tra chi tiết hơn những thống kê mô tả của dataframe như trung bình, phương sai, min, max, median của một trường dữ liệu chúng ta dùng hàm `df.describe()`

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 297
deletable: true
editable: true
id: _I58Z6Z5SzJ7
outputId: 0a32cdc0-f68c-4e63-b5f6-87a261e5ffe2
---
# Thống kê mô tả dữ liệu
df.describe()
```

+++ {"id": "ETd2i2lpuevj"}

## 2.1.4. Export to CSV, EXCEL, TXT, JSON

Đây là câu lệnh được sử dụng khá phổ biến để lưu trữ các file dữ liệu từ dataframe sang những định dạng khác nhau. Những định dạng này sẽ cho phép chúng ta load lại dữ liệu bằng các hàm `read_csv(), read_xlsx(), read_txt(), read_json()` sau đó.

```{code-cell} ipython3
:id: wAySTF-XugIY
:class: no-execute
%%script echo skipping

# Lưu dữ liệu sang file csv
df.to_csv('BostonHousing.csv', index = False)
# Lưu file excel
df.to_excel('BostonHousing.xls', index = False)
# Lưu dữ file json
df.to_json('BostonHousing.json') #do not include index = False, index only use for table orient
```

+++ {"deletable": true, "editable": true, "id": "ICRmGEYGSzKZ"}

# 2.2. Thao tác với dataframe

+++ {"id": "fk0tPbXNI6p4"}

## 2.2.1. Truy cập dataframe

Chúng ta có thể truy cập dataframe theo hai cách.

**Truy cập theo slice index:** Theo cách này chúng ta chỉ cần truyền vào index của dòng và cột và sử dụng hàm `df.iloc[rows_slice, columns_slice]` để trích xuất ra các dòng và cột tương ứng. Cách lấy slice cho rows và columns hoàn toàn tương tự như truy cập slice index trong [list](https://phamdinhkhanh.github.io/deepai-book/ch_appendix/appendix_dtypes_basic.html#list). 

_Note:_ `iloc` là viết tắt của indice location, tức là truy cập quan indice.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: VHG7YUzRRfvM
outputId: 635b034c-c6a1-4851-b2b0-e4a1015e8ce0
---
# Lựa chọn 5 dòng đầu và 5 cột đầu của df
df.iloc[:5, :5]
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: bg9DUpfRRdot
outputId: f4faa30d-e74f-49d0-f127-8b29d85d9e31
---
# Lựa chọn 5 dòng từ 5:10 và 2 cột từ 2:4
df.iloc[5:10, 2:4]
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: 89TxUvLHRxgn
outputId: 8e764ab3-24f2-42fa-a2ec-bc494475e393
---
# Lựa chọn 5 dòng cuối và các cột 1 và 3
df.iloc[-5:, [1, 3]]
```

+++ {"id": "69Vl8zuxSAIq"}

Ngoài ra ta cũng có thể truy cập các dòng theo row index của dataframe thông qua câu lệnh `df.loc[]`.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 235
id: AGUpcbzFSKbr
outputId: 0256b8e3-b670-4a80-c25d-19838e68be21
---
# Truy cập các dòng có index là 10:15
df.loc[10:15]
```

+++ {"id": "91f7rXWrR96K"}

**Truy cập theo column names:** Đây là cách được sử dụng phổ biến vì nó tường minh hơn. Theo cách này chúng ta sẽ truy cập các trường của dataframe bằng cách khai báo list column_names của chúng.

Ví dụ bên dưới chúng ta cần lấy ra các trường `['crim', 'tax', 'rad']` từ bảng `df`. Ta sẽ làm như sau:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: 6he7OmOyTo57
outputId: 891ff46f-e0df-4ca3-dc4c-89ca7ef59f0c
---
df[['crim', 'tax', 'rad']].head()
```

+++ {"id": "Wh_E3q8qTwtv"}

**Kết hợp cả hai cách**: Chúng ta có thể truy cập dataframe bằng cách kết hợp cả hai cách theo hướng sử dụng column names đối với cột và slice index đối với dòng:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: eqGPl5P3UG_V
outputId: 29e4116f-e724-47a4-ae0d-e008e7450e85
---
# Lấy ra các dòng từ 10:15 của các trường 'crim', 'tax', 'rad'
df[['crim', 'tax', 'rad']].iloc[10:15]
```

+++ {"id": "eDoGlT-IplRF"}

## 2.2.2. Lọc dataframe

Chúng ta có thể lọc dataframe thông qua các điều kiện đối với các trường. Điều kiện của trường được thể hiện như một biểu thức logic và bao trong dấu `[]`. Giả sử chúng ta muốn lọc ra các thị trấn mà có số phòng ở trung bình trên căn hộ là trên 4 thì truyền vào dấu `[]` điều kiện `df['rm'] > 4`.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: SoQVlccfp0ov
outputId: 57c284c6-dbfc-436d-829b-c191963b420e
---
df[df['rm'] > 4].head()
```

+++ {"id": "GtGYWFP6seqN"}

Nếu chúng ta muốn kết hợp nhiều điều kiện thì dùng biểu thức logic `and` hoặc `or`. Ví dụ: Muốn lọc thêm điều kiện thuế suất trên 250 ngoài điều kiện số phòng thì ta làm như sau: 

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: Q7Mky5hPs2RU
outputId: a20d47c3-24f4-441f-aae0-da37df8ad7f1
---
df[(df['rm']>4) & (df['tax']>250)].head()
```

+++ {"id": "tgUpVCMgtQKw"}

**Muốn lọc các cột theo định dạng dữ liệu thì như thế nào?**

Ta dùng hàm `df.select_dtypes()` để lọc các cột theo định dạng dữ liệu. Những định dạng chính bao gồm `integer, float, object, boolean`. Ví dụ: Bên dưới chúng ta lọc các trường có định dạng dữ liệu là `float`.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: kZOkjAettlBF
outputId: e820d4cf-6b50-43a2-e83a-0c3d8322bb47
---
df.select_dtypes('float').head()
```

+++ {"id": "TWenj3EKzhYo"}

**Lọc các cột theo pattern của tên cột**

Khi làm việc với dữ liệu lớn sẽ có những tình huống mà bạn bắt gặp các cột thuộc về cùng một nhóm và chúng có chung một pattern. Chẳng hạn như về age sẽ có `age_1, age_2, age_3`,.... Làm thế nào để bạn lọc ra được những biến này từ dữ liệu? Chúng ta sẽ dùng hàm filter. Đây là hàm cực kỳ tiện ích khi lọc cột từ những bộ dữ liệu lớn mà bạn sẽ thường xuyên sử dụng sau này.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: 7kK6Ne7X0O68
outputId: 2b25e043-d51d-45b2-86b2-a5909305976e
---
df2 = pd.DataFrame({
    'name':['a', 'b', 'c', 'd', 'e'],
    'age_1':[1, 2, 3, 4, 5],
    'age_2':[3, 5, 7, 9 , 10],
    'age_3':[2, 5, 2, 5, 6]
})

df2.head()
```

+++ {"id": "JPh7n2Ty0l29"}

Lựa chọn các cột bắt đầu là `age` thông qua hàm filter.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: KAukQpBs0ptf
outputId: 1f076a3f-e340-48fb-9aff-e2f1b70ee70a
---
df2.filter(regex='^age', axis=1)
```

+++ {"id": "2IWsc4Sc1-w7"}

Trong pandas thì `axis=1` là làm việc với cột và `axis=0` là làm việc với dòng. Giá trị của `regex=^age` có nghĩa là lọc các cột có chuỗi ký tự là `age` đứng đầu.

+++ {"id": "lBG6yLOp3ROI"}

## 2.2.3. Sort dữ liệu

Trong nhiều trường hợp bạn sẽ cần sort dữ liệu theo chiều từ thấp lên cao hoặc từ cao xuống thấp để biết đâu là những quan sát nhỏ nhất và lớn nhất cũng như việc tạo ra một đồ thị có trend rõ ràng và thể hiện quan hệ tuyến tính giữa các biến theo trend.

Để sort dữ liệu chúng ta sử dụng hàm `df.sort_values(.)`. Lựa chọn là `ascending = True` giúp sort theo thứ tự tăng dần, trường hợp `False` sẽ giảm dần.

Giả sử bên dưới chúng ta cùng sort giá trị của căn nhà theo chiều giảm dần.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
deletable: true
editable: true
id: tVOlfl66SzKe
outputId: 20abdf6d-3405-4a7c-846e-3c75483c831f
---
#Sort data
df.sort_values('medv', ascending = False).head()
```

+++ {"id": "B8aY1uMg4wwM"}

chúng ta cũng có thể sort theo một nhóm các trường. Ví dụ để sort đồng thời giá trị của căn nhà và giá trị thuế suất thì ta truyền vào list các trường cần sort là `['medv', 'tax']`.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: --W6U5kN46OH
outputId: 142ba670-116a-4549-ee94-5e0f2e675faa
---
df.sort_values(['medv', 'tax'], ascending = False).head()
```

+++ {"id": "6hs8u-KI58Em"}

## 2.2.4. Các hàm đối với một trường

+++ {"id": "SLhCPegz9iot"}

### 2.2.4.1. Min, max, mean, meadian, sum

Trên một trường dữ liệu của dataframe đã tích hợp sẵn các hàm tính toán như `min, max, mean, median, sum` để tính các giá trị đặc trưng cho từng trường.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
deletable: true
editable: true
id: 8f1IvxVaSzKv
outputId: 62fde870-3d84-434f-9a1e-a50589e29b4a
---
# min, max, mean, median, sum
print(df['tax'].min(), df['tax'].max(), df['tax'].mean(), df['tax'].median(), df['tax'].sum())
```

+++ {"id": "Q2br0ypm6yzR"}

### 2.2.4.2. Hàm cut

Hàm cut giúp ta phân chia giá trị của một trường liên tục vào những khoảng theo ngưỡng cắt. Kết quả trả ra là nhãn của từng khoảng mà chúng ta khai báo.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: Ys2dkJAa61Tw
outputId: b5f0603d-147b-4c3c-c906-d00a3639f7b2
---
bins = [-999999, 250, 400, 999999]
labels = ['low', 'normal', 'high']
# low: -999999 <- 250
# normal: 250 <- 400
# high: 400 <- 999999
df['tax_labels'] = pd.cut(df['tax'], bins=bins, labels=labels)
df[df['tax_labels']=='high'].head()
```

+++ {"id": "fd-gKdLf9kZI"}

### 2.2.4.3. Hàm qcut

Trong trường hợp chúng ta không muốn chia các bin dựa vào ngưỡng mà chỉ muốn khai báo số lượng bins và để cho hàm số tự quyết định ngưỡng để chia đều các quan sát vào các bins thì sử dụng hàm `pd.qcut(.)` (qcut là viết tắt của quantile cut). Bên dưới ta sẽ chia thành 3 bins (số bins sẽ được khai báo trong `q=3`) với labels tương ứng là `['low', 'normal', 'high']`

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 16Qlc0JK9oGl
outputId: 7932aafb-f481-439a-b9ac-1f1ce7e9f881
---
import numpy as np

labels = ['low', 'normal', 'high']
tax_labels = pd.qcut(df['tax'], q=3, labels=labels)
np.unique(tax_labels, return_counts = True)
```

+++ {"id": "xYUQzpmg-uyg"}

Trường hợp muốn xác định tỷ lệ phần trăm **luỹ kế** của các ngưỡng phân chia ta có thể khai báo q là list gồm các ngưỡng luỹ kế. Ví dụ bên dưới ta muốn chia làm ba khoảng giá trị, mỗi khoảng chiếm 33% thì ta khai báo ngưỡng luỹ kế `q = [0, 0.33, 0.66, 1]`

+++ {"id": "NdoooKORAJ82"}

### 2.2.4.4. Apply

Apply sẽ giúp ta biến đổi giá trị của một trường theo một hàm số xác định trước. Hàm số biến đổi được áp dụng trong apply sẽ là một hàm `lamda`. Hàm lambda là một khái niệm rất quan trọng trong python, hàm số này có cú pháp dạng `lambda x: formula`.

Phân tích kỹ hơn thì chúng ta thấy nó không có return. Điều này là phù hợp với ý nghĩa của hàm lambda vì nó không yêu cầu gía trị trả về ngay. Thực tế nó giống như một lời hứa sẽ thực hiện hàm đó tại thời điểm áp dụng một cách ngầm định bên trong một hàm khác (ở đây là hàm apply).

Ví dụ bên dưới ta muốn nhân đôi giá trị của tax thì có thể sử dụng hàm apply với lambda như sau:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: eLCcAq-BAZ6i
outputId: cec19196-41c3-478f-91c3-87116ce890c4
---
df['tax'].apply(lambda x: 2*x).head()
```

+++ {"id": "o7_il3CCAz-p"}

Ta cũng có thể áp dụng cho nhiều trường một lúc. Khi đó cần khai báo `axis=1` để biết rằng ta đang áp dụng trên từng cột, nếu axis=0 thì sẽ áp dụng trên từng dòng.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: nlch5qPtBAcW
outputId: 778712dc-8d49-4f39-a522-1cefeefc0455
---
df[['tax', 'medv']].apply(lambda x: 2*x, axis=1).head()
```

+++ {"id": "q8QX2M-tCleA"}

### 2.2.4.5. Map

Map là hàm giúp biến đổi giá trị của một biến sang giá trị mới dựa trên dictionary mà chúng ta áp dụng. Giá trị cũ sẽ là key và giá trị mới sẽ là value.

Bên dưới ta sẽ map các giá trị của trường `df['tax_labels']` sang các giá trị tiếng Việt.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: NfMuCbBNCzdk
outputId: b828dd0f-f3bf-4d54-dd52-fd9eef3d00d0
---
dict_tax = {
      'low':'thap',
      'normal':'tb',        
      'high':'cao'
    }
    
df['tax_labels'].map(dict_tax).head()
```

+++ {"id": "bzHoU7CWDbRi"}

## 2.2.5. Biểu đồ matplotlib trên pandas

Chúng ta có thể nói rằng pandas rất mạnh vì nó đã wrap dường như toàn bộ các đồ thị cơ bản của matplotlib vào bên trong các hàm thành phần của pandas column. Do đó việc visualize trở nên vô cùng ngắn gọn, thậm chí là chỉ trên một dòng.

Bên dưới chúng ta sẽ cùng lướt qua nhanh các đồ thị cơ bản khi visualize trên `pd.column`. Biến được áp dụng đồng nhất cho các đồ thị là `tax`.

**1. biểu đồ line**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 282
id: -FHtcPvT_wiM
outputId: 46d712d0-cf56-4825-d3f1-3371ba9820ff
---
df['tax'].plot()
```

+++ {"id": "qlV2PPNHGZYz"}

**2. Biểu đồ line kết hợp với point**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 282
id: 1cTXZvGUGelF
outputId: da107429-101e-4044-e869-013dd7b2caa4
---
df['tax'].plot(marker='o')
```

+++ {"id": "wvMU3iEyEbur"}

**3. Biểu đồ barchart**

Biều đồ này được dùng phù hợp khi chúng ta muốn so sánh chênh lệch giữa các nhóm về mặt giá trị tuyệt đối.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 322
id: gKpHgzepE-of
outputId: f7467ebf-72dc-48ee-ad73-ca4c0f2b144b
---
df_summary = df[['tax_labels', 'tax']].groupby('tax_labels').sum()
df_summary.plot.bar()
```

+++ {"id": "Ah126m4pFcFE"}

Ở đây ta sẽ phải dùng thêm hàm groupby để tạo thành bảng thống kê tổng thuế theo `tax_labels` rồi mới vẽ biểu đồ. Khi quen thuộc bạn có thể viết gọn hai câu lệnh lại thành một line như sau:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 322
id: PLKEBQH2FyVH
outputId: 498c75ab-78d5-4bad-faff-508d08364235
---
df[['tax_labels', 'tax']].groupby('tax_labels').sum().plot.bar()
```

+++ {"id": "FzUziXUMF2-B"}

**4. Biểu đồ pie**

Đây là biểu đồ dùng để thể hiện giá trị phần trăm. Phù hợp khi so sánh giá trị tương đối giữa các nhóm.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 265
id: W6wAkAhtF592
outputId: ae7d7fe2-9c06-41fd-9396-03665b118a09
---
df_summary['tax'].plot.pie(autopct = '%1.1f%%')
```

+++ {"id": "ZOq4DQnmHi8a"}

**5. Biểu đồ boxplot**

Biểu đồ boxplot sẽ được sử dụng để quan sát phân phối của biến đối với các giá trị min, max và các ngưỡng phân vị 25%, 50%, 75%. Căn cứ vào boxplot ta có thể biết được khoảng biến thiên của biến rộng hay hẹp, biến phân phối lệch trái hay phải.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 282
id: O4VRHviBHRgX
outputId: eb227d1a-2a31-4dba-8654-7b0e46e6ee35
---
df[['tax', 'age', 'b']].boxplot()
```

+++ {"id": "WkXiMgAtIlSE"}

**6. Biểu đồ area**

Biểu đồ area cho ta biết diện tích nằm dưới đường biểu diễn và trên trục hoành.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 297
id: XFjqMuzTIfyc
outputId: 56afae84-0f70-4d3f-9563-9a4ad0cb20bd
---
df_summary['tax'].plot.area()
```

+++ {"id": "IFztwZvSvtXl"}

# 2.3. Reshape dataframe trên pandas

+++ {"id": "Jq65_kYsv2R9"}

## 2.3.1. Melt

Hàm melt là hàm được lấy ý tưởng từ ngôn ngữ R. Hàm này sẽ làm cho bảng của chúng ta trở nên bớt cồng kềnh hơn bằng cách rút gọn nhiều measurements thành hai cột variable và value trong đó cột variable qui định loại measurement và value là giá trị của measurement. Bảng của bạn sẽ có ít cột hơn đáng kể nên trông giống như các cột measurement đang bị tan chảy vậy. Do đó nó có tên gọi là melt.

Bạn sẽ dễ hình dung hơn những gì mình nói thông qua ví dụ bên dưới. Giả sử bảng của mình gồm Ho, Ten là các dimensions và `ChieuCao, CanNang, Tuoi, Diem` là những measurements. 

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 111
id: 8l0VJfAgv7Vx
outputId: b0a9ec1b-a16b-47c4-ac93-1bb7b535f186
---
df5 = pd.DataFrame({
  'Ho':['Pham','Nguyen'],
  'Ten' :['Cong', 'Dong'],
  'ChieuCao':[170, 175],
  'CanNang':[60, 65],
  'Tuoi': [25, 27],
  'Diem': [8.5, 9.0],
})

df5
```

+++ {"id": "IidWw-xpv94A"}

Ta nhận thấy `Ho, Ten` là những dimension, bây giờ ta sẽ giữ nguyên những trường này và làm tan chảy các cột.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 297
id: hbvbAXWRwDb4
outputId: dc59fb23-39d5-4b91-87ec-1c3f98fa5373
---
df5.melt(['Ho', 'Ten'])
```

+++ {"id": "ozZ0QLkTwB2Q"}

Ta nhận thấy bảng đã trở nên gọn gàng hơn khi các cột được đưa vào trường `variable` và giá trị của chúng được đưa vào `value`.

Cách biến đổi `melt` sẽ phù hợp với các bảng đã phân chia sẵn dimension, measurement rõ ràng và số lượng measurements của bảng là lớn.

+++ {"id": "TA1WbKXRwN2B"}

## 2.3.2. Biến đổi Dummy

Cách biến đổi dummy là một cách rất hiệu quả để biến đổi một biến category thành một one-hot véc tơ. Cụ thể cũng với bảng `df5` ở trên, ta nhận thấy biến `Ho` gồm hai giá trị là `Pham` và `Nguyen`. Chúng ta có thể tạo thành một one-hot vector sao cho nếu giá trị đầu tiên là 1 thì tương ứng với họ `Pham` và giá trị thứ hai là 1 thì họ `Nguyen` (chưa xét tới trường hợp tồn tại họ khác `Pham` và `Nguyen`). Thường thì bạn sẽ nghĩ đến sử dụng hàm [LabelBinarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html) của sklearn nhưng pandas cung cấp cho bạn một hàm đơn giản hơn để thực hiện việc này. Đó là `pd.get_dummies()`

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 111
id: uqWHQ9YYwTGq
outputId: 10a95cfb-f1f3-4372-94c7-a474874f406b
---
pd.get_dummies(df5)
```

+++ {"id": "TqPAwMtkwWT6"}

Ta thấy hàm này sẽ tìm **toàn bộ** các biến là category (có định dạng trường là `object`) và tự động trải phẳng chúng. Khi đó các cột mới được tạo thành chính là `Ho_Nguyen`, `Ho_Pham` là những nhãn thuộc biến `Ho`. Giá trị của những trường này bằng 0 hoặc 1.

Mặc dù có cú pháp cực kì đơn giản nhưng hàm `pd.get_dummies()` lại cực kì hữu ích đối với data scientist khi xây dựng mô hình mà bạn cần ghi nhớ.

+++ {"id": "7cUTQsxQuyGp"}

# 2.4. Thống kê theo nhóm trên pandas

Khi làm việc với dữ liệu bảng chúng ta thường xuyên phải thống kê dữ liệu theo các nhóm để bắt dữ liệu tạo ra những thông tin insight hữu ích cho phân tích và ra quyết định. Ngoài ra những feature tốt, có sức mạnh phân loại và dự báo cao có thể được tạo thành từ việc thống kê dữ liệu theo nhóm. Quá trình thống kê và phân tích dữ liệu mặc dù tốn kém về mặt thời gian nhưng lại rất quan trọng đối với mô hình. Vì vậy chúng ta cần thực hiện chúng kỹ lưỡng và cần kết hợp giữa kỹ năng thống kê và kinh nghiệm thực tiễn.

Ở mục 5 này chúng ta sẽ làm quen với hai câu lệnh kinh điển trong pandas được sử dụng nhiều trong thống kê theo nhóm trên pandas đó là `df.groupby()` và `pd.pivotable()`. 

+++ {"id": "5Wp7x7Gpwe9d"}

## 2.4.1. df.groupby()

groupby là câu lệnh cho phép bạn áp dụng những hàm số trên measurements dựa trên việc phân nhóm dữ liệu theo các dimensions.

Nếu bạn chưa hiểu về khái niệm measurement và dimension thì mình có thể giải thích đơn giản là: measurement là những biến có thể cộng trừ nhân chia và đo đếm được còn dimension là những biến dùng để phân nhóm dữ liệu. Ví dụ chiều cao là một measurement có thể đo theo dimension là giới tính gồm các nhóm nam/nữ.

Cú pháp của hàm `df.groupby()` khá đơn giản:

```
df.groupby(by=None, 
  ...
)
```

Chúng ta cần xác định các chiều dimension trong `by`. Phía sau `groupby()` là một list các measurements mà ta cần áp dụng hàm lên trên những trường này. 

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: pb8dGHWGBRpU
outputId: 4d61b87d-fb5f-4330-a7b4-74e15570d013
---
df.groupby('tax_labels')['tax'].sum()
```

+++ {"id": "3b8CG_ZoBR4l"}

Theo cách trên thì ta chỉ áp dụng được với những hàm tính toán như `sum, avg, min, max` có sẵn trong dataframe. Nếu muốn sử dụng `groupby()` cho mọi biến đổi hàm chúng ta có thể dùng hàm `lambda` trong `apply()`.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: OaIUHewz7067
outputId: 6f178d3b-407e-4469-90de-dc65c24f49df
---
df.groupby('tax_labels')['tax'].apply(lambda x: sum(x))
```

+++ {"id": "b96HWU7gCYbe"}

Nếu muốn áp dụng tính toán cho nhiều measurements một lúc thì truyền vào một list các measurements. Chẳng hạn bên dưới ta truyền vào một list gồm `['tax', 'rm']`

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 173
id: knFc1LNM8Wbu
outputId: 89ccf77d-1596-4f8d-f0a8-6de2d141c1c1
---
import numpy as np
df.groupby('tax_labels')[['tax', 'rm']].apply(lambda x: np.mean(x))
```

+++ {"id": "W5m_klMu8gO0"}

Chúng ta cũng có thể tự định nghĩa các hàm được tuỳ biến theo ý muốn:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: nV8Tkyor8pf-
outputId: 37edcfdd-4c0e-4003-8537-9785d20ff32a
---
# Tính quantile 90% của mỗi nhóm tax_labels.
def quantile(x):
  q_90 = np.quantile(x, 0.9)
  return q_90

df.groupby('tax_labels')['tax'].apply(lambda x: quantile(x))
```

+++ {"id": "kJQxlYNL_DUi"}

Hoặc group theo nhiều chiều dữ liệu. Khi đó phải truyền vào `groupby()` một list các dimension.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: LFK3VSBQ_IPL
outputId: 18cebc78-3fb5-4dad-b1ba-f1f635325426
---
df.groupby(['tax_labels', 'chas'])['tax'].apply(lambda x: quantile(x))
```

+++ {"id": "Bl6G2CX3C_xY"}

Ưu điểm của `groupby()` đó là nhanh gọn, dễ hiểu. Nhưng nhược điểm của `groupby()` đó là chúng ta chỉ có thể áp dụng cùng một biến đổi hàm số cho mọi measurements. Ở `pivot_table` bạn có thể tuỳ biến sâu hơn từng hàm đối với từng measurement nhưng cú pháp sẽ phức tạp hơn một chút.

+++ {"id": "FLlbvbZMwdj3"}

## 2.4.2. Pivotable

+++ {"id": "iQjXrj6uu-Q2"}


Pivot table là một công thức có ứng dụng rất quan trọng trong pandas. Nó giúp cho chúng ta thực hiện các thống kê trên các biến measurement theo các chiều dimension. 

Bạn sẽ hình dung ra cách áp dụng của `pivot_table()` thông qua ví dụ bên dưới.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 390
id: Ami93HxRu5Ab
outputId: 0809a73f-9f1f-4ed0-b524-d059683b96a9
---
import numpy as np

pd.pivot_table(df, 
               columns = ['tax_labels', 'chas'], 
               index = ['rad'], 
               values = 'tax',
               aggfunc = np.sum)
```

+++ {"id": "YG1KoSPvvPZY"}

Bạn hình dung ra nội dung của bảng thống kê trên chứ?

Bảng thống kê trên sẽ tính tổng số thuế thu được phân theo các cột là `tax_labels` và `chas` (tax_labels gồm low, normal và high và chas gồm 0-không bao bởi sông, 1-bao bởi sông).

Các dòng lại được phân nhóm theo chỉ số mức độ tiếp cận đường cao tốc `rad` gồm các giá trị `1,2,3,4,5,6,7,8,24`.

Như vậy ta có thể hình dung được trong công thức của pivot_table, các đối số của nó có ý nghĩa như sau:

* `columns`: List các dimensions của cột mà chúng ta cần thống kê.
* `index`: List các dimensions theo dòng mà chúng ta cần thống kê.
* `values`: List các measurement chúng ta sử dụng để tính toán.
* `aggfunc`: Qui định hàm số chúng ta sẽ dùng để biến đổi measurement. Trong ví dụ này chúng ta áp dụng hàm np.sum cho toán bộ các measurement.

+++ {"id": "XZK_uq7RvT54"}

**Làm sao để qui định mỗi measurement một công thức?**

Giả sử chúng ta cần tính thêm trung bình số phòng trên căn hộ. Tức là thêm trung bình của trường `rm` trong khi vẫn cần tính tổng của trường `tax`. Khi đó cần khai báo `aggfunc` dưới dạng một dictionary có key là tên của measurement và value là công thức của measurement. 

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 421
id: ySI68-shvUBM
outputId: 5b04a30e-01a4-4f42-a592-e03a29cd2ad2
---
pd.pivot_table(df, 
               columns = ['tax_labels', 'chas'], 
               index = ['rad'], 
               values = ['tax', 'rm'],
               aggfunc = {
                   'tax': np.sum,
                   'rm': np.mean
                   })
```

+++ {"id": "zdl_2mV1vcFR"}

Bảng của chúng ta đã tăng gấp đôi số cột. Dòng đầu tiên của bảng là `rm`, `tax` là những thông tin ứng với từng measurement.

Bạn thấy đó, `pd.pivot_table()` hoàn toàn đơn giản và rất hiệu quả phải không nào?

+++ {"id": "ib1Fh5ZGUalg"}

# 2.5. Join, Merge và Concatenate bảng

Những doanh nghiệp lớn thường tổ chức cơ sở dữ liệu dưới dạng những bảng dữ liệu có quan hệ. Những bảng này được liên kết với nhau bởi key dưới những quan hệ dữ liệu như one-to-one, many-to-one hoặc one-to-many. Những kiến trúc phổ biến trong data warehouse như `star schema` và `snowflake schema` sẽ giúp cho chúng ta nhanh chóng join các bảng lại với nhau để tạo ra những bảng raw data tổng hợp phục vụ cho các nhu cầu phân tích, thống kê và xây dựng mô hình.

Ngoài ngôn ngữ SQL là công cụ chính để làm việc với những hệ cơ sở dữ liệu có quan hệ, Data scientist cũng cần nắm vững những kỹ năng liên kết join, merge và concatenate bảng trên pandas mà thông qua chương này mình sẽ giới thiệu tới các bạn.


+++ {"id": "-HRX_8rehFhQ"}

## 2.5.1. Các kiểu join

Chúng ta có 4 kiểu join chính là `left join, right join, inner join, full join` được thể hiện qua biểu đồ venn bên dưới:

![](https://www.dofactory.com/img/sql/sql-joins.png)

Chúng ta có hai bảng bên trái và bên phải với những phần thông tin chung (giao nhau giữa hai vòng tròn) và riêng. Phần diện tích màu xanh lá cây là Kết quả của phép join. Chúng ta có thể hình dung kết quả của phép join đó là:

* left join: Lấy bảng bên trái làm gốc và đưa thêm thông tin bảng bên phải nếu nó xuất hiện ở bảng bên trái.
* right join: Tương tự như left join nhưng bảng bên phải sẽ làm gốc.
* inner join: Lấy những thông tin mà xuất hiện **đồng thời** ở cả hai bảng.
* full join: Lấy những thông tin xuất hiện ở **một trong hai** bảng.

Bên dưới chúng ta cùng thực hành join.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: qYmppFhmiwC2
outputId: bfef8815-7053-4791-bfa1-7890e449e4ba
---
import pandas as pd

df_sinhvien = pd.DataFrame({
    'ID':['001', '002', '003', '004', '005'],
    'Name':['Pham Van Nghia', 'Tong Thuy Linh', 'Le Van Dai', 'Tran Quang Nghia', 'Doan Thu Ha'],
    'Age': [25, 26, 25, 23, 22],
    'Province':['Nam Dinh', 'Thanh Hoa', 'TP Ho Chi Minh', 'Da Nang', 'Can Tho']  
})

df_sinhvien
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: 4nVt3wevuCve
outputId: 6019dea0-c87f-4f91-b4e5-5f98b1485e0f
---
df_score = pd.DataFrame({
    'ID': ['001', '002', '003', '006', '007'],
    'Math': [6.75, 9, 8, 7, 10],
    'Physic': [8, 9, 9, 8.5, 9],
    'Chemistry': [7, 9.5, 7.5, 9, 10],
    'Province': ['Nam Dinh', 'Thanh Hoa', 'TP Ho Chi Minh', 'Quang Nam', 'Nghe An']
})

df_score
```

+++ {"id": "12V9DejAvF18"}

## 2.5.2. Câu lệnh pd.merge()

+++ {"id": "0r31cXz1u4dV"}

Cú pháp chung của câu lện `pd.merge()` đó là:



```
pd.merge(
    left,
    right,
    how="inner",
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    sort=True,
    suffixes=("_x", "_y"),
    copy=True,
    indicator=False,
    validate=None,
)
```

Trong đó một số trường quan trọng:

* left: bảng bên trái
* right: bảng bên phải
* how: Phương pháp join gồm `left, right, inner, outer`
* left_on: Trường ở bảng bên trái sử dụng để join
* right_on: Trường ở bảng bên phải sử dụng để join
* left_index: Mặc định là False. Nếu True, sử dụng row index ở bảng bên trái như là trường join.
* right_index: Mặc định là False. Nếu True, sử dụng row index ở bảng bên phải như là trường join.
* suffixes: Nếu hai bảng tồn tại các trường trùng nhau thì sử dụng suffix để phân biệt trường nào thuộc bảng nào. Mặc định là `('_x', '_y')`.

Tiếp theo áp dụng câu lệnh trên để merge bảng sinh viên với điểm theo key là ID sinh viên.

**inner join:** Theo cách này chỉ ID xuất hiện ở đồng thời hai bảng mới được lựa chọn.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 142
id: 3oyJHcBPu_Wt
outputId: 496c8839-85d0-4710-96f0-94263d80d618
---
pd.merge(df_sinhvien, df_score, 
         left_on='ID', 
         right_on='ID', 
         how='inner', 
         suffixes=['_Sv', '_Score'])
```

+++ {"id": "8yR-JK2O8Z_K"}

**left join:** Những ID xuất hiện ở bảng bên trái sẽ được lựa chọn.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: 4nz9MYqJ6xa_
outputId: 73fca285-6b8a-4426-986d-a7444b6d56b6
---
pd.merge(df_sinhvien, df_score, 
         left_on='ID', 
         right_on='ID', 
         how='left', 
         suffixes=['_Sv', '_Score'])
```

+++ {"id": "dbd8cW8r8lgA"}

**right join:** Những ID xuất hiện ở bảng bên phải sẽ được lựa chọn.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: 0a6Yopq660Pq
outputId: bcaab6ea-feea-45a3-dc51-d053478b92c1
---
pd.merge(df_sinhvien, df_score, 
         left_on='ID', 
         right_on='ID', 
         how='right', 
         suffixes=['_Sv', '_Score'])
```

+++ {"id": "cGdztfOO8qIH"}

**outer join:** Tất cả ID xuất hiện ở bảng bên trái hoặc bảng bên phải sẽ được lựa chọn.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 266
id: nrkLh4-o63fD
outputId: a704cbd3-94e2-4c27-f811-34fdc5400df0
---
pd.merge(df_sinhvien, df_score, 
         left_on='ID', 
         right_on='ID', 
         how='outer', 
         suffixes=['_Sv', '_Score'])
```

+++ {"id": "YbXlsQ4T5xkj"}

## 2.5.3. df.join()

DataFrame có một hàm là hàm _join()_ có chức năng tương đương với merge, dùng để liên kết bảng theo các keys.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: TW4nX7pcTvN3
outputId: d67a36a6-86dd-401b-d52e-58dfdf644999
---
df_sinhvien.join(df_score, lsuffix='_Sv', rsuffix='_Score')
```

+++ {"id": "CYwulae8UOo8"}

Mặc định bảng sẽ join theo index của dòng, các đối số lsuffix` và `rsuffix`lần lượt được sử dụng để qui định hậu tố (_suffix_) cho bảng bên trái và bảng bên phải nếu xuất hiện trường trùng tên.

Nếu muốn thực hiện hàm `join()` theo một trường nào đó, chúng ta phải thiết lập index cho bảng là trường cần join rồi sau đó mới thực hiện join. Ví dụ bạn cần join theo ID:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 235
id: MltNbc7CU7sf
outputId: 7955922b-736e-4a9e-f2e4-111f9c319f2e
---
# Thiết lập index
df_sinhvien.set_index('ID', inplace = True)
df_score.set_index('ID', inplace = True)

# Join bảng
df_sinhvien.join(df_score, lsuffix='_Sv', rsuffix='_Score')
```

```{code-cell} ipython3
:id: NMuMkWo9MO5s

# reset lại index
df_sinhvien.reset_index('ID', inplace = True)
df_score.reset_index('ID', inplace = True)
```

+++ {"id": "68m8u-a5fvQa"}

## 2.5.4. Câu lệnh pd.concat()

Câu lệnh `pd.concat()` được sử dụng để nối hai bảng theo dòng hoặc theo cột. Đây là câu lệnh được sử dụng khá phổ biến để tạo bảng tổng hợp từ các bảng dữ liệu nhỏ. Một ví dụ khá cụ thể đó là trong package [vnquant](https://github.com/phamdinhkhanh/vnquant) dữ liệu mỗi mã chứng khoán sẽ bị phân trang. Nếu download lần lượt trừng trang thì sẽ lâu, do đó để tăng tốc thì chúng ta sẽ download song song nhiều trang một lúc và sử dụng lệnh `pd.concat()` để nối dữ liệu thành một bảng chính.

Cú pháp chung của lệnh `pd.concat()` sẽ như sau:

```
pd.concat(
  objs, 
  axis=0, 
  join='outer', 
  ignore_index=False, 
  keys=None, 
  levels=None, 
  names=None, 
  verify_integrity=False, 
  sort=False, 
  copy=True
)
```

Trong đó:

* objs: Là list các bảng cần concanate.
* axis: Mặc đinh là 0, nối theo dòng. Trái lại là 1 nếu nối theo cột.
* join: `inner` chỉ lấy các dòng hoặc cột có cùng index; `outer` lấy cả các dòng hoặc cột khác index.


**Nếu muốn nối theo dòng thì làm thế nào?**

Để nối hai bảng theo dòng thì ta sẽ để `axis=0`, đây là giá trị mặc định của đối số này trong `pd.concat()`.

Nếu chúng ta muốn nối hai bảng theo dòng và chỉ lấy trường thông tin mà cả hai bảng đều có thì cần thiết lập `join='inner'`.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 359
id: 3ulMhhMHg0L7
outputId: 11ccf65d-44fc-4425-e12f-be7d82422e98
---
pd.concat([df_sinhvien, df_score], 
          axis=0,
          join='inner')
```

+++ {"id": "1qfMF-t2j67F"}

Ta cũng có thể nối hai bảng theo dòng và lấy tất cả các trường thông tin ở cả hai bảng thì sẽ thiết lập `join='outer'`. Đây là giá trị mặc định của đối số này. Giá trị `NaN` tự động được fill đối với những thông tin không có. 

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 359
id: gZfhnSgpj0LA
outputId: 6ea298ab-66ec-47af-82f1-d7f269ae751c
---
pd.concat([df_sinhvien, df_score], 
          axis=0, 
          join='outer')
```

+++ {"id": "dAr-h-fUit61"}

**Làm sao để nối theo cột**

Để nối theo cột thì khai báo `axis=1`.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: FuUXCkCFijDJ
outputId: 738448a0-1a3c-4e67-9318-bd4a4cc5f1eb
---
pd.concat([df_sinhvien, df_score], 
          axis=1)
```

+++ {"id": "fV3f6ZcqmKQQ"}

Khi đó các dòng sẽ được liên kết theo row index ở mỗi bảng. Nếu chúng ta muốn các dòng được liên kết theo một trường nào đó như 'ID' thì cần `set_index()` là trường đó trước khi nối.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 266
id: Bh6tzuQvmSZn
outputId: ae53a1c7-e7d8-495e-8aeb-be78b1b3b15f
---
pd.concat([df_sinhvien.set_index('ID'), df_score.set_index('ID')], 
          axis=1)
```

+++ {"id": "EXKns_HYol4w"}

Nếu ta chỉ muốn các dòng mà ID xuất hiện ở cả hai bảng thì thêm `join='inner'`.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: LNX6HRxdnh2m
outputId: d5d06a40-c8aa-49a9-e8a6-993d9dff52d1
---
pd.concat([df_sinhvien, df_score], 
          axis=1,
          join='inner')
```

+++ {"id": "hHXTjmbgJbg1"}

## 2.5.5. append()

Ngoài câu lệnh `pd.concat()` thì bản thân một dataframe cũng có hàm `append()` được sử dụng để nối bảng **Theo dòng**.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 359
id: 9duxCNqdw7gN
outputId: 9c6fe2cc-e685-42f6-acd0-12cd6e9edf30
---
df_sinhvien.append(df_score)
```

+++ {"id": "wiklzVHlx9d3"}

Câu lệnh này sẽ thường được sử dụng trong tình huống bạn muốn tạo bảng tổng hợp từ nhiều bảng con có cùng cấu trúc.

Ví dụ: Bạn muốn tạo ra một bảng về lợi tức chứng khoán của toàn bộ các ngành từ số liệu chứng khoán của từng ngành.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 204
id: 2mlKCdfixKy7
outputId: c27a6c6d-c529-4b77-a604-9b4291ce30f0
---
import numpy as np
import random
import string

df_chungkhoan = pd.DataFrame(columns = ['Nganh', 'Interest'])

for i in range(5):
  r = np.random.uniform(-0.01, 0.01)
  nganh = random.choice(string.ascii_lowercase)
  df_sector = pd.DataFrame({'Nganh': [nganh], 'Interest': [r]}, index=[i])
  df_chungkhoan = df_chungkhoan.append(df_sector)

df_chungkhoan
```

+++ {"deletable": true, "editable": true, "id": "1M9TueqrSzRk"}

# 2.6. Kết nối SQL

Đối với những data scientist làm việc trong những doanh nghiệp quản lý dữ liệu trên data warehouse như Ngân Hàng, công ty Chứng Khoán, Bảo Hiểm thì thường xuyên phải kết nối SQL để truy vấn dữ liệu. Python có rất nhiều các packages cung cấp instance connection tới SQL cũng như biến đổi data trên cú pháp của SQL. Mình sẽ không thể giới thiệu hết toàn bộ những packages này mà sẽ giới thiệu tới các bạn hai packages phổ biến nhất đó là:

**sqlalchemy**

Đây là một pakage cho phép chúng ta kết nối và truy vấn trên những dữ liệu SQL một cách trực tiếp theo mô hình server-client side. Chúng ta sẽ phải khai báo một số thông tin quan trọng để khởi tạo kết nối như:

* Tên server là gì?
* Tên database trong server cần truy vấn.
* port: Cổng kết nối, thường mặc định của MSSQL là 1443.
* username: Tên user.
* password: Mật khẩu truy cập.

**Chú ý**: Ở phần ví dụ thực hành liên quan tới SQL thì mỗi máy sẽ có một cấu hình khác nhau. Để thực hành được code bên dưới trước tiên máy tính của bạn cần cài SQL Server và có sẵn những database trong server.

Bạn sẽ cần khai báo đúng các trường cấu hình truy cập trong `DB` và tên bảng tại `TableName`.

Nếu bạn thực hành bị lỗi các ví dụ tại mục 3 này, hãy tạm thời bỏ qua chúng.

```{code-cell} ipython3
:deletable: true
:editable: true
:id: JNqJK60rSzRn
:class: no-execute
%%script echo skipping

from sqlalchemy import create_engine, MetaData, Table, select, engine
# Create parameters
TableName = 'WorkOrder'

DB = {
    'drivername': 'mssql+pyodbc',
    'servername': 'LAPTOPTCC-PC',
    #'port': '1443',
    #'username': '',
    #'password': '',
    'database': 'TestDB',
    'driver': 'SQL Server Native Client 11.0',
    'trusted_connection': 'yes',
    'legacy_schema_aliasing': False
}

# Create the connection
engine = create_engine(DB['drivername'] + '://' + DB['servername'] + '/' + DB['database'] + '?' + 'driver=' + DB['driver'] 
+ ';' + 'trusted_connection=' + DB['trusted_connection'], legacy_schema_aliasing=DB['legacy_schema_aliasing'])

conn = engine.connect()

# Required for querying tables
metadata = MetaData(conn)

#Table to query
tbl = Table(TableName, metadata, autoload = True, schema = 'dbo')
# tbl.create(checkfirst = True)

#select all
sql = tbl.select()

#run sql code
result = conn.execute(sql)

df3 = pd.DataFrame(data = list(result), columns = result.keys())
df3.head()
```

+++ {"id": "CofxsxPtkSnQ"}

Sau khi sử dụng connection thì chúng ta nhớ đóng lại connection để giải phóng memory và port.

```{code-cell} ipython3
:deletable: true
:editable: true
:id: QV-ZbvMASzR0
:class: no-execute
%%script echo skipping

#close connection to free memory
conn.close()
```

+++ {"id": "MYdLXbZ7kZv0"}

Chúng ta cũng có thể thực thi các lệnh của SQL thông qua engine SQL mà chúng ta đã khởi tạo. Kết quả sẽ được truy vấn và tính toán trực tiếp từ server trả về  như câu lệnh ta yêu cầu.

```{code-cell} ipython3
:deletable: true
:editable: true
:id: 9sOBrCjcSzSA
:class: no-execute
%%script echo skipping

#another way, use read_sql_query() function from pandas. This function use directly engine without initialize connection
pd.read_sql_query("SELECT TOP 5 * FROM WorkOrder", engine)
```

+++ {"deletable": true, "editable": true, "id": "LVAczueISzSI"}

**pyodbc**

Đây là package được thiết kế riêng để truy vấn trên những hệ cơ sở dữ liệu sử dụng kết nối ODBC của Microsoft.

```{code-cell} ipython3
:id: GTA2MpbXldKG
:class: no-execute
%%script echo skipping

import pandas.io.sql
import pyodbc

server = 'LAPTOPTCC-PC'
db = 'AdventureWorks'

#create connection
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=' 
                      + DB['servername']
                      + ';DATABASE=' 
                      + DB['database'] 
                      + ';Trusted_Connection=yes')
```

+++ {"id": "VpUuNkFklemN"}

Sau khi khởi tạo kết nối thì ta cũng có thể sử dụng các câu lệnh của SQL như thông thường để truy vấn và tính toán thông tin như bên dưới:

```{code-cell} ipython3
:deletable: true
:editable: true
:id: 3b6ltxYVSzSK
:class: no-execute
%%script echo skipping

#query db
sql = """SELECT * FROM WorkOrder"""

df4 = pandas.io.sql.read_sql(sql, conn)
df4.head()
```

+++ {"id": "7xMOcaU-l0qf"}

**Cursor cho SQL**

Cursor là một con trỏ dẫn tới một vùng nhớ mà lưu trữ dữ liệu. Sử dụng cursor sẽ giúp ta tiết kiệm bộ nhớ vì chúng ta không phải phân bổ bộ nhớ cho dữ liệu ngay mà chỉ sử dụng địa chỉ để trỏ tới dữ liệu. Cursor được sử dụng phổ biến trong SQL đặc biệt là trong các vòng for. Trong `pyodbc` chúng ta sẽ sử dụng `cursor()` trong vòng for để duyệt qua các dòng như bên dưới:

```{code-cell} ipython3
:id: 2fEuPIC6l8iu
:class: no-execute
%%script echo skipping

import pyodbc

con = pyodbc.connect("DRIVER={SQL Server};"
                     "SERVER=LAPTOPTCC-PC;"
                     "DATASET=AdventureWorksDW2012;"
                     "Trusted_Connection=yes;")
cursor = con.cursor()

cursor.execute("SELECT TOP 10 * FROM AdventureWorksDW2012.dbo.DimCustomer")

for row in cursor.fetchall():
    print('row = ', row)
```

+++ {"id": "RNLHTkygGw6c"}

# 2.7. Tổng kết

Qua bài hướng dẫn này bạn đã được làm quen với những chức năng của pandas trong phân tích, xử lý và biến đổi dữ liệu. Tổng kết lại chúng ta đã đi qua các mục:

* Cách đọc, lưu và khởi tạo dataframe
* Thao tác dữ liệu trên dataframe: truy cập bảng, sort, filter và các hàm cơ bản trên dataframe.
* Thay đổi shape của bảng qua melt và dummy.
* Thống kê theo groupby và pivot_table.
* Các lệnh join, merge, concatenate bảng.
* Kết nối với dữ liệu SQL.

Những kiến thức trên không cover hết toàn bộ về pandas nhưng là những kiến thức hay dùng nên bạn đọc cần nắm vững.

Tiếp theo là bài tập thực hành cho bài viết này.


+++ {"id": "E79OeAJKG2dl"}

# 2.8. Bài tập

Sử dụng bộ dữ liệu [Churn Customer](https://raw.githubusercontent.com/phamdinhkhanh/datasets/master/Churn_Modelling.csv) bạn hãy.

1. Đọc dữ liệu từ file csv, có phần header là row thứ nhất và index là trường RowNumber.
2. Thống kê mô tả đối với các trường trong bảng dữ liệu này.
3. Tính trung bình điểm `CreditScore` theo `Geography`.
4. Phân đều `Age` thành 5 nhóm độ tuổi sao cho mỗi nhóm chiếm 20% số quan sát.
5. Vẽ biểu đồ barchart thống kê số lượng khách hàng theo nhóm độ tuổi vừa tạo được.

+++ {"id": "q8C2zYneG4pc"}

# 2.9. Tài liệu

+++ {"id": "zmnEQAxMtBPb"}

1. [pandas-docs](https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html)
2. [pandas-tutorial w3school](https://www.w3schools.com/python/pandas/default.asp)
3. [pandas introduction - pandas.pydata.org](https://pandas.pydata.org/pandas-docs/stable/getting_started/intro_tutorials/index.html)
4. [pandas dataframe - pandas.pydata.org](https://pandas.pydata.org/docs/reference/frame.html)
