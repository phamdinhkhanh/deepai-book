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

# 1.2. Định dạng sequence

Định dạng sequence là định dạng mà cho phép chúng ta lưu được nhiều giá trị trong cùng một biến.

+++ {"id": "xzg60pK3eOUm"}

## 1.2.1. List

List là định dạng cho phép lưu trữ nhiều item trong cùng một biến. Đây là một trong bốn định dạng built-in data type dùng để lưu trữ collection bên cạnh các định dạng khác là `tupple, dictionary, set`.

Một list sẽ có hai tính chất là iterable (có thể duyệt qua các phần tử bên trong nó) và mutable (có thể thay đổi giá trị bên trong nó).

**Các biến đổi đơn giản với list gồm những gì?**

Để khởi tạo một list chúng ta bao quanh list bởi dấu ngoặc vuông. Bên trong là tập hợp các thành phần của list cách nhau bởi dấu phảy.

Trong một list chúng ta có thể duyệt qua các phần tử trong list; Thực hiện các thao tác CRUDE (gồm các chức năng cập nhật, thêm, sửa, xoá) các thành phần trong list; concatenate hai list. Cụ thể như bên dưới:

```{code-cell} ipython3
# CRUDE
# Khởi tạo list
list1 = ['physics', 'chemistry', 1997, 2000] # Hoặc dùng từ khoá list thay cho []: list1 = list('physics', 'chemistry', 1997, 2000)
print('list1: ', list1)

# Duyệt qua các phần tử trong list. Tính chất Iterable
for item in list1:
  print(item)

# Cập nhật 1 phần tử trong list. Tính chất mutable
list1[0] = 'math'
print('list1 update: ', list1)

# Xóa một phần tử trong list
list1.remove(2000)
print('list1 after delete: ', list1)

# Độ dài của 1 list:
print('length of list1: ', len(list1))

# Gán thêm phần tử cho 1 list:
list1.append(2019)
print('list1 after append: ', list1)

# concatenate 2 list với nhau
list2 = ['people', 'teacher', 'student']
list_concate = list1 + list2
print('list1 and list2 after concate: ', list_concate)

# hoặc chúng ta có thể concate thông qua phép cộng
list3 = list1
list3 += list2
print('list1 and list2 after concate: ', list3)

# Lưu ý nếu ta dùng lệnh append để append một list thì sẽ tự động tạo thêm một phần tử mới ở vị trí cuối cùng chứa list được append
list3.append(list2)
print('list1 append list2: ', list3)

# trích xuất 1 phần tử trong list
item3 = list1.pop(3)
print('item 3 indice of list1: ', item3)
print('list1 after pop 3 indice: ', list1)
```

+++ {"id": "bne5rCTy1Y1R"}

Các phép biến đổi khác:
* Tìm ra index của một phần tử nằm trong list theo giá trị.

```{code-cell} ipython3
list4 = ['vi', 'mot', 'cong', 'dong', 'AI', 'vung', 'manh', 'hon']
# Tìm ra index của từ đầu tiên có giá trị là 'cong'
print(list4.index('cong'))
```

+++ {"id": "yjyEnEEr2M-E"}

* Mở rộng các thành phần của list

```{code-cell} ipython3
list4.extend(['khanh', 'blog']) # Tương tự như phép cộng list
print(list4)
```

+++ {"id": "-VlK7czA23tq"}

* Insert thêm vào một vị trí bất kỳ của list. Ví dụ: `list4.insert(5, 'Viet Nam')` thì từ `Viet Nam` sẽ được thêm vào vị trí index thứ 6 nằm trong `list4`.

```{code-cell} ipython3
list4.insert(5, 'Viet Nam')
print(list4)
```

+++ {"id": "W1NRaR01AH78"}

**Vòng for đối với list**

Chúng ta có thể sử dụng vòng for để truy cập các phần tử trong một list như sau:

```{code-cell} ipython3
for item in list4:
  print(item)
```

+++ {"id": "lGsvJN7CAb48"}

Tuy nhiên trong một số trường hợp ta muốn viết cú pháp gọn hơn, kết quả trả về  ngay là một list thì có thể sử dụng vòng **for bên trong list**. Đây là một style cực mạnh, cho phép ta xử lý list ngắn gọn và thông minh hơn trong nhiều tính huống.

```{code-cell} ipython3
[item for item in list4]
```

+++ {"id": "9dKL-FLQux8S"}

**Làm sao truy cập các phần tử của list?**

Để truy cập các phần tử của một list chúng ta có nhiều cách.

* Sử dụng slice index: Theo cách này chúng ta sẽ xác định một lát cắt gồm các index liên tiếp trong một list và khai báo chúng trong dấu ngặc vuông bên cạnh tên list. 

Ví dụ: Để truy cập vào các phần tử liên tiếp của một list, chẳng hạn từ `i` đến `j` chúng ta chỉ cần thông qua khoảng slice của list đó là `i:j+1` (lưu ý phải cộng một tại phần tử cuối là `j+1`).

```{code-cell} ipython3
# Truy cập vào phần tử 1 của list1
print(list1[1])

# Truy cập vào phần tử có index từ 1->3.
print('* From indices 1->3: list1[1:4]\n', list1[1:4])

# Truy cập vào 2 index đầu tiên
print('* 2 first indices: list1[:2]\n', list1[:2])

# Truy cập vào 2 index cuối cùng
print('* 2 last indices: list1[-2:]\n', list1[-2:])

# Truy cập vào các phần tử từ phần tử đầu tiên tới phần tử liền trước vị trí cuối cùng là hai vị trí.
print('* From first to -2 indices (2 orders to touch last): list1[:-2]\n', list1[:-2])

# Truy cập từ index thứ 2 đến index cuối cùng
print('* From second indice to last indice: list1[2:]\n', list1[2:])
```

+++ {"id": "OvqWkorwyab3"}

* Sử dụng list các thứ tự cần truy cập:

Cách này chúng ta sẽ khai báo các vị trí phần tử của list mà chúng ta muốn truy cập. Những vị trí này có thể cách rời nhau. Ưu điểm của phương pháp này là có thể truy cập bất kỳ vị trí nào trong list, nhược điểm là phải gõ nhiều.

```{code-cell} ipython3
# Truy cập vào vị trí có indice 1 và 3 trong list1
[item for (i, item) in enumerate(list1) if i in [1, 3]]
```

+++ {"id": "tTM1PXJWxheO"}

**Các thành phần của một list có được phép trùng nhau?**

List cho phép các phần tử được trùng nhau. Tính chất này không được chấp nhận ở dictionary và set.

```{code-cell} ipython3
:id: gFV0BhXjxrXE

list4 = ['machine', 'learning', 'algorithms', 'to', 'practice', 'practice']
```

+++ {"id": "qHB50gfG1fHo"}

Ngoài ra các thành phần trong một list có thể khác nhau về định dạng. Bên dưới là list chứa thành phần vừa có kiểu ký tự và kiểu số.

```{code-cell} ipython3
:id: XhHHSX9k1tF1

list4 = ['machine', 'learning', 'algorithms', 'to', 'practice', 2021]
```

+++ {"id": "tKytXcFLv81f"}

**Sort một list thì như thế nào ?**

Chúng ta dùng lệnh `sort`. Có hai lựa chọn là `reverse=True` (đảo ngược) hoặc `reverse=False (tăng tiến).

```{code-cell} ipython3
list5 = [4, 5, 1, 7, 2]
# Sort tăng tiến
list5.sort()
print('* normal sort list5: \n', list5)
# Sort reverse
list5.sort(reverse=True)
print('* reverse sort list5: \n', list5)
```

+++ {"id": "oZIjPfoa5Chp"}

Theo cách trên thì hàm `sort()` được coi là một method của biến `list5`. Sau khi gọi hàm `list5.sort()` thì phép sắp xếp được kích hoạt và thay thế chính giá trị của list5 gốc (đây là tính chất inplace). Chúng ta có thể sử dụng hàm `sorted()` độc lập bên ngoài để sort giá trị của list5 mà giá trị của list5 sẽ không bị thay đổi.

```{code-cell} ipython3
list5 = [4, 5, 1, 7, 2]
print('* sorted(list5): \n',sorted(list5))
```

+++ {"id": "PMN9KKwm6dof"}

Ngoài ra chúng ta có thể sort theo key là một hàm bất kỳ trong `sorted()`. Theo cách này chúng ta có thể sắp xếp biến đầu vào $x$ sao cho giá trị của hàm đầu ra $(x-3)^2$ là tăng dần.

```{code-cell} ipython3
sorted(list5, key = lambda x: (x-3)**2)
```

+++ {"id": "r7l0mKFQeUL1"}

## 1.2.2. Tuple

Tuple cũng tương tự như list nhưng là định dạng immutable, tức là không thể sửa, xóa, cập nhật.

Các phần tử của tuple được cách nhau bởi dấu phảy và bao quanh bởi ngoặc đơn.

```{code-cell} ipython3
# Khởi tạo tuple
tuple1 = ('a', 'b', 'c', '2020')
print('tuple1: ', tuple1)

# Truy cập vào các phần tử của tuple
for item in tuple1:
  print(item)

# Độ dài
print('length tuple1: ', len(tuple1))

# Concate 2 tuple
tuple2 = ('x', 'y', 'z')
tuple_concate = tuple1 + tuple2
print('tuple concatenate: ', tuple_concate)

# Truy cập vào phần tử đầu tiên
tuple1[0]
```

+++ {"id": "2tRAS11-4mri"}

Lưu ý: Tupple là định dạng immutatble tức là **không thay đổi được**. Chúng ta sẽ không thể thêm/sửa/xoá các phần tử của tupple. VD: Cập nhật tuple1[0] = 0 sẽ báo lỗi.

```{code-cell} ipython3
tuple1[0] = 0
```

+++ {"id": "VVaqaAJMeYiA"}

## 1.2.3. Dictionary

Dictionary là định dạng cho phép truy cập các giá trị thông qua một key **duy nhất**. key giống như tên gọi và giá trị là thứ được trả về thông qua key.

Mỗi một phần tử của dictionary được đặc trưng bởi một cặp `{key: value}`. Các phần tử ngăn cách nhau bởi dấu phảy và bao quanh dấu `{}`.

**Các biến đổi đơn giản trên dictionary gồm những gì?**

Chúng ta có thể truy cập các phần tử của dictionary thông qua giá trị `key` của nó. Các giá trị tương ứng với một `key` cũng có thể được cập nhật bằng phép gán.

```{code-cell} ipython3
# Khởi tạo một dictionary
dict1 = {'name': 'khanh', 'age': '27', 'job': 'AI research engineer', 'love': 'math'}
# In ra giá trị thông qua key
print('name: ', dict1['name'])
print('age: ', dict1['age'])
print('job: ', dict1['job'])
print('love: ', dict1['love'])

# Cập nhật giá trị một phần tử
dict1['love'] = 'girl'
print('love update: ', dict1['love'])

# Độ dài của dict1
print('dict1 length: ', len(dict1))

# Truy cập vào toàn bộ các key
print('all keys: ', dict1.keys())

# Truy cập vào toàn bộ giá trị
print('all values: ', dict1.values())

# Thêm một phần tử mới cho dict
dict1['IQ'] = '145'
print('dict after update: ', dict1)
```

+++ {"id": "QCUlBSUH7j82"}

Cách khởi tạo dictionary trên là thủ công, bạn có thể khởi tạo một dictionary gồm 100 keys liên tiếp có cùng một giá trị thông qua hàm `dict.fromkeys(keys, value)`.

```{code-cell} ipython3
# Khởi tạo dictionary gồm các keys từ 0 tới 3, giá trị đều là []
dict2 = dict.fromkeys(range(4), [])
dict2
```

+++ {"id": "-_GqYV4Z6IGO"}

**Sort một dictionary như thế nào?**

Chúng ta dùng hàm sorted như bên dưới. Khi đó giá trị trả về của dict sẽ tự động sort theo key.

```{code-cell} ipython3
sorted(dict1)
```

+++ {"id": "Gj8zI3PX8J9K"}

Muốn customize sâu hơn cách sort của hàm `sorted`, ta có thể dùng thêm chức năng `key` của hàm `lambda`. Đây là một vũ khí khá lợi hại:

```{code-cell} ipython3
# Chỉ sort dict1 theo giá trị cuối của dictionary
sorted(dict1, key=lambda x: x[-1:])
```

+++ {"id": "g1MMJJZp82Qj"}

**Nếu không muốn sort theo keys mà sort theo values thì sao?**

Đây là một ứng dụng rất phổ biến mà bạn sẽ thường xuyên bắt gặp nó khi làm việc với dữ liệu. Để thực hiện phương pháp sort này, hẳn bạn còn nhớ cú pháp `vòng for trong list` ở chương list chứ ? Chúng ta sẽ áp dụng nó ở đây.

```{code-cell} ipython3
{k: v for k, v in sorted(dict1.items(), key=lambda item: item[1])}
```

+++ {"id": "PFQrsAVJeg-2"}

## 1.2.4. Set

Set là một tợp hợp gồm các giá trị **duy nhất**. Set không cho phép truy cập thông qua index vì nó không được đánh index. các phần tử trong set không thể thay đổi được nhưng set có thể thêm bớt. Để khởi tạo một set thì các phần tử của set cách nhau bởi dấu phảy và bao quanh dấu `{}`.

**Các phép biến đổi cơ bản đối với set**

```{code-cell} ipython3
:id: DobXx9oSejea

# Khởi tạo một set
set1 = set([1, 2, 3, 4, 'alo']) # or {1, 2, 3, 4, 'alo'}
print('set1: ', set1)

# Truy cập vào các phần tử của set
for item in set1:
  print(item)

# Độ dài của set:
print('length set1: ', len(set1))

# Xóa 1 phần tử trong set
set1.discard(4)
print('set1 after discard 4: ', set1)

# Hợp 2 set với nhau:
set2 = {1, 3, 5, 8}
print('set2: ', set2)
set_union = set1.union(set2)
print('union set1 and set2: ', set_union)

# Tìm ra các phần tử chung bằng phép giao:
set_intersection = set1.intersection(set2)
print('intersection set1 and set2:', set_intersection)

# Tìm ra các phần tử thuộc set2 nhưng không thuộc set1
print('belong set2 but not set1: ', set2 - set1)

# Kiểm tra quan hệ giữa 2 set
print('check set2 belong set1: ', set1 >= set2)
print('check set1 belong set2: ', set1 <= set2)
```

+++ {"id": "Hqh2H4oECJ1w"}

# 1.3. Tóm tắt

Như vậy qua bài viết này bạn đã nắm vững được các kiến thức cơ bản về những định dạng dữ liệu built-in trong python gồm:

* Các kiểu dữ liệu `float, integer, boolean, string`, miền giá trị và cách thức làm việc với chúng.

* Định dạng dữ liệu sequence `list, tuple, dictionary, set`, cách sử dụng, những cú pháp thường gặp trong xử lý dữ liệu kèm theo những `tip and trick` để giải quyết nhanh vấn đề.

Bạn đọc hãy cùng làm những bài tập bên dưới để cải thiện kỹ năng của mình.


+++ {"id": "DIHw5MQJDS4-"}

# 1.4 Bài tập

Cho một dictionary gồm tên và điểm số.

```{code-cell} ipython3
:id: DxxhVWlqDcap

points = {'an': 9, 'binh': 7, 'khanh': 8, 'linh': 10, 'huyen': 6}
```

1. Hãy sắp xếp dictionary trên theo điểm số tăng dần.
2. Hãy tạo một dictionary có key là điểm số và giá trị là tên.

Tiếp theo, cho một list như sau:

```{code-cell} ipython3
:id: j2fYKVWuEN9s

target = ['vi', 'mot', 'cong', 'dong', 'AI', 'vung', 'vung', 'manh', 'manh', 'hon']
```

3. Tìm các giá trị duy nhất của list đó.
4. Đếm số lần lặp lại của mỗi phần tử trong list.
5. Tạo ra một list mới mà nếu phần tử đó đã xuất hiện trong list rồi thì chỉ lấy duy nhất giá trị bắt gặp đầu tiên.

+++ {"id": "UXNIM3rxFAEi"}

Cho một đoạn văn bản như sau:

```{code-cell} ipython3
:id: qX7hQmdFFC_d

'Machine Learning Algorithms to Practice là một cuốn sách cân bằng giữa lý thuyết và thực hành'
```

+++ {"id": "S4AeWcFTFMpb"}

6. Hãy phân chia câu văn trên thành một list mà mỗi một từ trong câu là một phần tử của list.
7. Đảo ngược vị trí các từ trong list vừa tạo được.

Tiếp theo, tính các giá trị sau:
8. Giá trị: $n! = 1\times 2 \times 3 \dots \times n$
9. Tích của các số chẵn liên tiếp nhỏ hơn $n$.
10. Tích của các số lẻ liên tiếp nhỏ hơn $n$.

+++ {"id": "H-ryufTCGU78"}

# 1.5. Tài liệu tham khảo

1. [Datatype - w3school](https://www.w3schools.com/python/python_datatypes.asp)
2. [Python Guide - Stanford](https://web.stanford.edu/class/cs106ap/handouts/python-guide.html)
3. [VariableType - tutorialspoint](https://www.tutorialspoint.com/python/python_variable_types.htm)