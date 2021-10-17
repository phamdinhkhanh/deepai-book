#!/usr/bin/env python
# coding: utf-8

# # 3.1. Khởi tạo một mảng trên numpy

# ## 3.1.1. Khởi tạo ngay từ đầu
# 
# Để khởi tạo một mảng trên numpy chúng ta sử dụng câu lệnh rất quen thuộc là `np.array()`. Numpy cho phép chúng ta khởi tạo mảng được cấu hình theo định dạng dữ liệu cụ thể như `float, interger, boolean, string`. Chúng ta cũng lưu ý rằng các phần tử của một mảng trong numpy phải đồng nhất về định dạng dữ liệu.

# In[1]:


import numpy as np

A = np.array([[1, 2],
              [3, 4]])

print(A)
print("dtype of matrix A: ", A.dtype)


# Ban đầu các phần tử của ma trận $\mathbf{A}$ gồm toàn những giá trị nguyên. Chúng ta có thể xác định định dạng dữ liệu cho ma trận ngay tại lúc khởi tạo thông qua đối số `dtype`.

# In[2]:


B = np.array([[1, 2],
              [3, 4]], dtype=np.float32)

print(B)
print("dtype of matrix B: ", B.dtype)


# Những mảng kiểu float thì giá trị các phần tử có thêm dấu `.` ở cuối để phân biệt với mảng số nguyên.
# 
# Ngoài cách thay đổi trên, định dạng của ma trận cũng có thể được biến đổi thông qua hàm `A.astype()`. Đây là một hàm thuộc tính có sẵn ở mỗi mảng.

# In[3]:


# Lưu ý phải gán A = A.astype() để lưu thay đổi cho A
A = A.astype(np.float32)
print(A.dtype)


# ## 3.1.2. Khởi tạo ngẫu nhiên
# 
# Nếu khởi tạo một ma trận nhỏ khoảng vài phần tử thì việc gõ tay là khả thi. Nhưng đối với những mảng kích thước lớn chúng ta sẽ không thể nhập hết toàn bộ các giá trị. Khi đó ta sẽ khởi tạo ngẫu nhiên cho những biến này.
# 
# 
# * `np.random.randn(d0, d1, d2,..., dn)`: $d_i$ là chiều thứ $i$ của mảng. Theo cách này các giá trị sẽ được lấy mẫu ngẫu nhiên từ phân phối chuẩn hoá (_normal distribution_) có trung bình bằng 0 và phương sai bằng 1.

# In[4]:


R = np.random.randn(2, 3)
print(R)


# * `np.random.normal(loc=0.0, scale=1.0, size=None)`: Khởi tạo mảng mà các phần tử của mảng tuân theo phân phối chuẩn (_Gaussian distribution_) với trung bình chính là `loc` và phương sai là `scale`. Phân phối Gaussian (hay còn gọi là phân phối chuẩn) là trường hợp tổng quát của phân phối chuẩn hoá, chúng có hàm mật độ xác suất `pdf` (_probability density function_) được tính dựa trên hai tham số trung bình và phương sai như sau:
# 
# $$pdf(x; \mu, \sigma) = \frac{\exp(\frac{-(x-\mu)^2}{2\sigma^2})}{\sqrt{2\pi \sigma^2}}$$
# 
# Chúng ta sẽ gặp lại phân phối này ở chương lý thuyết xác suất.

# In[5]:


R = np.random.normal(loc=1, scale=2, size=(2, 3))
print(R)


# * `np.random.uniform(low=0.0, high=1.0, size=None)`: Các phần sẽ được khởi tạo theo phân phối đều trong khoảng từ `low` tới `high`. Trong phân phối đều thì mật độ xác suất tại mọi điểm là như nhau giữa `[low, high]`:
# 
# $$pdf(x; low, high) = \frac{1}{high-low}$$

# In[6]:


R = np.random.uniform(low=-1, high=1, size=(2, 3))
print(R)


# Trong trường hợp bạn chỉ muốn sinh ngẫu nhiên đối với số nguyên thì có thể sử dụng hàm `np.randint()`

# In[7]:


R = np.random.randint(low=-5, high=5, size=(2, 3))
print(R)


# Ngoài ra còn các khởi tạo ngẫu nhiên theo phân phối khác như `t-student, gamma, beta, chi-square, Fisher, ....`. Bạn đọc có thể sử dụng như list các hàm mình liệt kê bên dưới:
# 
# * `np.random.standard_t(df, size=None)`: Phân phối t-student.
# * `np.random.chisquare(df, size=None)`: Phân phối Chi-square.
# * `np.random.f(dfnum, dfden, size=None)`: Phân phối Fisher.
# * `np.random.gamma(shape, scale=1.0, size=None)`: Phân phối gamma
# * `np.random.beta(a, b, size=None)`: Phân phối beta

# # 3.2. Đọc và save numpy từ file

# ## 3.2.1. Save numpy
# 
# Chúng ta có thể lưu numpy dưới nhiều định dạng khác nhau như file `npy` và `txt`.

# In[8]:


# from google.colab import drive
# import os

# Mount google driver
# drive.mount('/content/drive/')
# os.chdir("drive/My Drive/mybook")

import numpy as np

# Khởi tạo một mảng ngẫu nhiên A
A = np.random.randn(3, 3)
A


# Save mảng A theo hàm `np.save()` và `np.savetxt()` có cú pháp:
# 
# * `np.save(file, arr)`: Output là định dạng npy. Save được mọi tensor với số chiều tuỳ ý.
# * `np.savetxt(file, arr)`: Output là định dạng txt. Cách này ít phổ biến hơn vì ta chỉ save được mảng 1D và 2D.
# 
# Trong các công thức trên thì `file` là đường link tới vị trí save và `arr` là mảng cần save.

# In[9]:


# Định dạng npy
np.save('numpy_save/A', A)
# Định dạng txt
np.savetxt('numpy_save/A.txt', A)
# List các file
get_ipython().system('ls -a numpy_save')


# ## 3.2.2. Load numpy từ file
# 
# Sau khi save numpy, ta có thể load chúng lại theo hàm `np.load()` và `np.loadtxt()`.

# In[10]:


A = np.load("numpy_save/A.npy")
print(A)


# In[11]:


np.loadtxt("numpy_save/A.txt")
print(A)


# ## 3.2.3. convert mảng từ dataframe
# 
# Trong nhiều trường hợp khi xây dựng mô hình chúng ta có thể lấy giá trị của mảng từ dataframe bằng cách gọi thuộc tính `df.values`.

# In[12]:


import pandas as pd

df = pd.read_csv("numpy_save/A.txt", header=None, sep=" ")
df


# In[13]:


A = df.values
print(A)


# # 3.2. Truy cập mảng trên numpy
# 
# Chúng ta có thể truy cập mảng con theo các chiều thành phần của mảng cha dựa vào khai báo indices trên từng chiều. Lưu ý rằng trong lập trình thì số thứ tự sẽ giảm 1 so với thực tế do theo qui ước của python thì STT (_số thứ tự_) bắt đầu từ 0. Do đó ở dưới khi mình nói đến STT thì bạn hiểu là mình đang nói tới STT trong lập trình và được giảm 1 so với thực tế. Ví dụ: dòng số 0 tức là dòng đầu tiên (số 1) trong thực tế.
# 
# Để khai báo indices chúng ta có thể sử dụng slice indices được ngăn cách bởi dấu `:`. Slice indices giúp rút ngắn việc phải nhập từng indices. Nó rất phù hợp với các tình huống các vị trí cần truy cập là liên tục nhau. Bên dưới là 3 công thức slice indice thường sử dụng:
# 
# * `indice_start:(indice_end+1)`: lấy toàn bộ các indices bắt đầu từ `indice_start` và kết thúc là `indice_end`. 
# 
# * `-num_indices:`: Lấy một số lượng `num_indices` ở vị trí cuối cùng. Ký hiệu `-num_indices` có thể hiểu là trước vị trí cuối cùng `num_indices` vị trí.
# 
# * `:num_indices`: Lấy một số lượng `num_indices` ở vị trí đầu tiên.
# 
# Trong trường hợp các vị trí là không liên tục thì ta có thể liệt kê ra các vị trí cần lấy ở từng chiều vào một list.

# In[14]:


import numpy as np 
X = np.array([[1, 2, 3, 4],
              [4, 5, 6, 2],
              [7, 8, 9, 1],
              [2, 5, 1, 5]])


# Tiếp theo ta sẽ truy cập vào các phần tử của X. Bạn sẽ hình dung cách truy cập qua lần lượt các ví dụ bên dưới:

# In[15]:


# Truy cập vào phần tử thuộc dòng 1 và cột 2
print(X[1, 2])


# Khi truy cập theo slice indice thì vị trí kết thúc phải cộng thêm một. Tức là chúng ta cần truy cập các dòng có indice từ 1 đến 2 thì phải khai báo indice là `1:3`. Đây là một điều dễ bị nhầm lẫn đối với người mới bắt đầu lập trình mà chúng ta cần đặc biệt lưu ý.

# In[16]:


# Truy cập vào các dòng từ 1 đến 2 và các cột từ 1 đến 2
print(X[1:3, 1:3])


# In[17]:


# Truy cập vào 2 dòng, 2 cột đầu tiên
print(X[:2, :2])


# In[18]:


# Truy cập vào 2 dòng, 2 cột cuối cùng
print(X[-2:, -2:])


# In[19]:


# Truy cập vào dòng 0 và 2. Cách lấy indices không liên tục.
print(X[[0, 2], :])


# # 3.3. Thay đổi shape của mảng

# ## 3.3.1. Reshape mảng
# 
# Chúng ta có thể thay đổi shape cho mảng dựa vào câu lệnh `np.reshape()`. Hàm này vừa là một thuộc tính sẵn có trong mỗi mảng (tức là với ma tận A ta có thể gọi `A.reshape()`) vừa là hàm của numpy.

# In[20]:


B = np.array(np.arange(12))
print("B: \n", B)
print("B.shape: ", B.shape)


# Mảng B là một véc tơ 1 chiều có độ dài là 12. Chúng ta có thể reshape mảng B thành một ma trận 2 dòng, 6 cột.

# In[21]:


# dùng B.reshape()
B.reshape(2, 6)


# In[22]:


# dùng np.reshape()
np.reshape(B, (2, 6))


# Trong một mảng có $n$ chiều, khi đã biết $n-1$ chiều thì chiều còn lại có thể tính ra được bằng cách chia tổng số phần tử cho tích của $n-1$ chiều. Do đó chúng ta có thể đánh dấu chiều chưa biết bằng -1 để numpy sẽ tự tính.

# In[23]:


B.reshape(2, -1)


# In[24]:


B = np.reshape(B, (2, -1))
print(B)


# ## 3.3.2. Chuyển vị các chiều
# 
# Trong các mô hình deep learning chúng ta có rất nhiều những lựa chọn khác nhau về định dạng đầu vào của một tensor. Ví dụ như đối với dữ liệu ảnh thì có sự khác biệt về input mặc định giữa các framework như pytorch và tensorflow. Tensorflow chấp nhận ảnh đầu vào có các chiều được sắp xếp theo dạng `batch_size x width x height x channels` trong khi pytorch lại chấp nhận kiểu `batch_size x channels x width x height`.
# 
# Chúng ta có thể hoán vị các chiều của mảng nhiều chiều bằng câu lệnh `np.transpose(arr, (d_i, d_j,..., d_n))`. Trong đó, arr là ma trận cần hoán vị chiều và `(d_i, d_j,...,d_n)` là thứ tự các chiều mới mà chúng ta hoán vị.

# In[25]:


# Khởi một bức ảnh ngẫu nhiên có kích thước channels x width x height= 3 x 28 x 28
A = np.random.randint(0, 255, size=(3, 28, 28))
print("A.shape: ", A.shape)
# Chuyển channels về cuối
A = np.transpose(A, (1, 2, 0))
print("A transpose shape: ", A.shape)


# ## 3.3.3. Concatenate và Stack hai mảng
# 
# Giả sử bạn đang có rất nhiều bức ảnh khác nhau. Bạn muốn ghép các bức ảnh đó theo chiều `height` (một bức nằm trái và một bức nằm phải). Để thực hiện được điều này thì bạn cần tới câu lệnh concatenate.

# In[26]:


A = np.random.randint(0, 255, size=(28, 28, 3))
B = np.random.randint(0, 255, size=(28, 28, 3))
print("A.shape:", A.shape)
print("B.shape:", B.shape)
AB = np.concatenate((A, B), axis=1)
print("shape after concatenate: ", AB.shape)


# Ngoài ra muốn tạo một batch cho huấn luyện, trong batch đó gồm 2 ảnh là A và B thì chúng ta sử dụng câu lệnh stack.

# In[27]:


np.stack([A, B]).shape


# In[28]:


# Stack theo chiều vertical
print(np.vstack([A, B]).shape)

# Stack theo chiều horizontal
print(np.hstack([A, B]).shape)


# ## 3.3.4. Mở rộng mảng
# 
# Trong nhiều tình huống, sự khác biệt về số lượng các chiều trong mảng có thể dẫn tới những lỗi liên quan tới shape. Ví dụ: Chúng ta không thể nhân một mảng kích thước 3 chiều với một ma trận kích thước 2 chiều.

# In[29]:


A = np.random.randint(0, 2, size=(1, 2, 3))
B = np.random.randint(0, 2, size=(1, 3))

# A.dot(B)


# Khi đó chúng ta cần mở rộng mảng thêm một chiều ở vị trí trong cùng thì mới thực hiện được phép nhân.

# In[30]:


# Mở rộng mảng
B = np.expand_dims(B, axis=-1)
print("B new shape: ", B.shape)
print("A.dot(B) shape: ", A.dot(B).shape)


# Đối số `axis=-1` có nghĩa là ta sẽ mở rộng mảng tại chiều cuối cùng. Khi đó shape `(1, 3)` trở thành `(1, 3, 1)`.

# # 3.4. Các hàm trên numpy

# ## 3.4.1. min, max, mean, sum
# Cũng giống như pandas, numpy cung cấp một loạt các hàm thống kê thông dụng theo chiều dòng hoặc cột như min, max, mean, sum. Trong tất cả các công thức bên dưới thì `axis=1` là theo dòng và `axis=0` là theo cột.

# In[31]:


B = np.array([[ 0,  1,  2,  3,  4,  5],
              [ 6,  7,  8,  9, 10, 11]])

# Tìm ra phần tử nhỏ nhất của B
B.min()


# In[32]:


# Tìm ra phần tử nhỏ nhất của B theo dòng
B.min(axis=1)


# In[33]:


# Tìm ra phần tử nhỏ nhất của B theo cột
B.min(axis=0)


# Tương tự với các hàm còn lại là max, mean, sum

# In[34]:


print(B.max(axis=1))
print(B.mean(axis=1))
print(B.sum(axis=1))


# ## 3.4.2. minimum, maximum
# 
# Hàm minimum, maximum giúp tìm ra giá trị lớn nhất và nhỏ nhất ở cùng một vị trí giữa hai mảng cùng kích thước. Kết quả trả ra là một mảng có cùng kích thước nhưng có giá trị.

# In[35]:


A = np.array([[1, 3, 6], 
              [2, 7, 9]])

B = np.array([[0, 4, 5], 
              [1, 6, 10]])

# Tìm các phần tử maximum của A và B ở cùng vị trí indice
np.maximum(A, B)


# In[36]:


# Tìm các phần tử minimum của A và B ở cùng vị trí indice
np.minimum(A, B)


# ## 3.4.3. argmax, argmin
# 
# Hàm argmax và argmin thường xuyên được sử dụng trong Machine Learning vì chúng có tác dụng tìm ra các nhãn có xác suất là cao nhất.

# In[37]:


# Tìm ra indice có giá trị lớn nhất của véc tơ
A = np.array([1, 5, 3, 2, 4])
np.argmax(A)


# In[38]:


# Tìm ra indice có giá trị lớn nhất của các dòng
A = np.array([[1, 2],
              [3, 4],
              [5, 0],
              [2, 1]])

np.argmax(A, axis=1)


# **Bài tập:** Cho ma trận $\mathbf{B}$ bên dưới là đầu ra có kích thước `NxC`trong đó N là số quan sát và C là số classes. Mỗi dòng của ma trận là một phân phối xác suất của một quan sát. Thống kê kết quả nhãn thuộc về mỗi loại.

# In[39]:


B = np.array([[0.1, 0.2, 0.7],
              [0.6, 0.4, 0.0],
              [0.1, 0.5, 0.4],
              [0.3, 0.3, 0.4],
              [0.1, 0.8, 0.1]])


# ## 3.4.4. argsort
# 
# Hàm argsort là hàm tìm ra indices được sắp xếp theo số thứ tự tăng dần của một mảng hoặc một véc tơ.
# 
# `np.argsort(arr, ascending, axis)`

# In[40]:


# Tìm ra indice của các cột theo thứ tự tăng dần
np.argsort(B, axis = 0)


# In[41]:


# Tìm ra indice của các dòng theo thứ tự tăng dần
np.argsort(B, axis = 1)


# **Bài tập:** Với ma trận $\mathbf{B}$ như trên, làm thế nào để tìm ra indice của các dòng hoặc các cột theo thứ tự giảm dần?

# ## 3.4.5. np.exp() và hàm softmax
# 
# Hàm exponetial cho phép ta tính số mũ cơ số tự nhiên $e$ của bất kỳ các ma trận nào theo công thức:
# 
# $$f(x) = e^{x}$$

# In[42]:


np.exp(B)


# Từ ma trận `exponential` chúng ta có thể dễ dàng tính được phân phối xác suất tương ứng với từng quan sát theo công thức của hàm softmax:
# 
# $$P(y=i|\mathbf{x}) = \frac{e^{x_i}}{\sum_{i=1}^{C} e^{x_i}}$$
# 
# Trên thực tế việc tính toán $e^{x_i}$ với các giá trị $x_i$ lớn thường dẫn tới sự tốn kém chi phí tính toán và có thể dẫn tới hiện tượng số bị tràn luồng (_overflow_). Chúng ta có thể làm giảm cả tử và mẫu của phân phối xác suất $P(.)$ bằng cách chia cho $e^{x_i}$ nhỏ nhất.
# 
# $$P(y=i|\mathbf{x}) = \frac{e^{x_i-\min{(x)}}}{\sum_{i=1}^{C} e^{x_i-\min(x)}}$$

# In[43]:


# Ma trận phân phối xác suất được tính dựa trên các dòng của ma trận B.
E = np.exp(B)
P = []

for i in range(E.shape[0]):
  Ei = E[i, :]
  pi = Ei/np.sum(Ei)
  P.append(pi)

P = np.array(P)
P


# **Bài tập:** Hày tìm một cách tính khác để tính toán phân phối xác suất của các dòng trên ma trận $\mathbf{B}$.

# ## 3.4.6. Giữ nguyên shape
# 
# Các phép biến đổi trên ma trận trên ta đều thấy làm giảm chiều của ma trận đi một. Ví dụ khi tính `max` theo các dòng của ma trận thì chiều cột sẽ bị tiêu giảm. `numpy` cung cấp cho chúng ta đối số `keepdims` để giữ nguyên số chiều như ma trận đầu tiên.

# In[44]:


# Tính max theo các dòng ta thu được véc tơ độ dài là rows
np.max(B, axis=1)


# Khi sử dụng `keepdims=True` thì kết quả trả ra sẽ là một ma trận, chiều bị tiêu giảm sẽ được thay thế bởi kích thước 1.

# In[45]:


np.max(B, axis=1, keepdims=True)


# ## 3.4.7. Gieo hạt và trộn lẫn dữ liệu trên numpy
# 
# `np.random.seed()` là một hàm gieo hạt giúp cho các kết quả từ những hàm sinh dữ liệu được giữ nguyên kết quả sau mỗi lần chạy. Hàm số này có những ứng dụng rất quan trọng như giúp cố định tập train/test khi phân chia từ dữ liệu tổng thể, cố định giá trị khởi tạo cho cho mạng nơ ron network,.... Bên trong hàm gieo hạt chúng ta truyển vào một giá trị số tự nhiên chính là id của lần gieo hạt chẳng hạn như `np.random.seed(123)`. Giá trị id này sẽ giúp tạo ra một địa chỉ ô nhớ trên RAM để lưu lại quá trình sinh dữ liệu. Nhờ đó khi gọi lại đúng id đó ta có thể khôi phục lại giá trị ban đầu. 
# 
# Bên dưới chúng ta sẽ tạo ra một chuỗi dữ liệu ngẫu nhiên được cố định giá trị bằng phương pháp gieo hạt và kiểm tra xem sau khi chạy lại hàm số đó thì có sinh ra giá trị như lúc trước hay không?

# In[46]:


import numpy as np
np.random.seed(123)
np.random.randint(0, 100, 10)


# In[47]:


np.random.seed(123)
np.random.randint(0, 100, 10)


# Như vậy khi sử dụng lại mã gieo hạt thì kết quả ta thu được từ hai lần khởi tạo ngẫu nhiên véc tơ số nguyên là không thay đổi.

# **Bài tập:** Cho một tập dữ liệu có kích thước là 100. Hãy tìm cách phân chia tập dữ liệu thành tập train và tập test một cách ngẫu nhiên theo tỷ lệ 70/30.

# # 3.5. Các ma trận đặc biệt

# ## 3.5.1. Ma trận đơn vị
# 
# Ma trận đơn vị là một ma trận vuông có các phần tử trên đường chéo chính bằng 1 và các phần tử còn lại bằng 0. Đây là một ma trận rất đặc biệt vì tích của một ma trận khác với ma trận đơn vị sẽ bằng chính ma trận đó.
# 
# $$\mathbf{A}\mathbf{I} = \mathbf{A}$$
# 
# Ma trận $\mathbf{A}$ vuông khả nghịch nhân với ma trận nghịch đảo của nó thì bằng ma trận đơn vị:
# 
# $$\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$$

# In[48]:


# Khởi tạo ma trận đơn vị với shape=5x5
I = np.identity(5, dtype=np.float16)
I


# ## 3.5.2. Ma trận 1
# 
# Ma trận 1 là ma trận gồm toàn những phần tử có giá trị là 1.

# In[49]:


# Khởi tạo ma trận 1 với kích thước (3, 3)
A = np.ones((3, 3), dtype=np.float32)
A


# **Bài tập:** Một ma trận B nhân với ma trận 1 thì sẽ có giá trị bằng bao nhiêu?
# 
# Chúng ta cũng có thể khởi tạo ma trận 1 có kích thước bằng với shape của một ma trận khác không qua câu lệnh `np.ones_like()`

# In[50]:


# Khởi tạo ma trận B ngẫu nhiên kích thước (2, 3)
B = np.random.randn(2,3)
# Khởi tạo ma trân 1 có kích thước bằng với (2, 3)
A = np.ones_like(B)
A


# ## 3.5.3. Ma trận 0
# 
# Cũng tương tự như ma trận 1, ma trận 0 gồm toàn bộ các phần tử bằng 0.

# In[51]:


# Khởi tạo ma trận 0 kích thước (3, 3)
np.zeros((3, 3))


# In[52]:


# Khởi tạo ma trận 0 có kích thước như ma trận B
np.zeros_like(B)


# ## 3.5.4. Ma trận đường chéo chính
# 
# Ma trận đường chéo chính là ma trận vuông mà có các giá trị khác đường chéo chính đều bằng 0. Đường chéo chính ở đây được xác định là những điểm $a_{ii}$ có indice dòng và cột bằng nhau.
# 
# Để khởi tạo ma trận đường chéo chính chúng ta sử dụng câu lệnh `np.diag()` và khai báo véc tơ đường chéo chính.

# In[53]:


np.diag(np.array([1, 2, 3]))


# Trong trường hợp `np.diag()` có đầu vào là một ma trận thì kết quả trả ra chính là đường chéo của ma trận đó (ma trận có thể không vuông).

# In[54]:


A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8 ,9]])

# Đường chéo chính của ma trận A
np.diag(A)


# # 3.6. Các phép toán trên ma trận

# ## 3.6.1. Phép chuyển vị
# 
# Phép chuyển vị một ma trận kích thước $m\times n$ sẽ biến thành một ma trận mới kích thước $n \times m$ sao cho các dòng của ma trận ban đầu sẽ biến thành cột của ma trận chuyển vị.

# In[55]:


print("Original matrix: \n", A)
print("Transpose matrix: \n", A.T)


# ## 3.6.2. Ma trận nghịch đảo.
# 
# Ma trận nghịch đảo của một ma trận vuông $\mathbf{A}$ khả nghịch là một ma trận được ký hiệu $\mathbf{A}^{-1}$. Ma trận này có tích với $\mathbf{A}$ là một ma trận đơn vị.
# 
# $$\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$$

# In[56]:


# Tính ma trận nghịch đảo của A
# pinv là pseudo-inverse of a matrix
A = np.array([[1, 2.5, 2],
              [3, 4.2, 1],
              [5, 2.1, 2]])

A_inv=np.linalg.pinv(A)

print("Inverse A: ")
A_inv


# Tích của ma trận $\mathbf{A}$ với ma trận nghịch đảo của nó là một ma trận đơn vị.

# In[57]:


A.dot(A_inv)


# ## 3.6.3. Hạng (_rank_) của ma trận.
# 
# Để tìm hiểu về hạng ma trận trước tiên ta làm quen với khái niệm độc lập tuyến tính của hệ véc tơ.
# 
# **Hệ véc tơ độc lập tuyến tính là gì?**
# 
# Một hệ véc tơ $\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_n$ là độc lập tuyến tính nếu phương trình:
# 
# $$k_1 \mathbf{e}_1 + k_2 \mathbf{e}_2 + \dots + k_n \mathbf{e}_n = 0$$
# 
# có một nghiệm duy nhất $k_1 = k_2 = \dots = k_n = 0$
# 
# Trái lại, nếu tồn tại một nghiệm mà $k_j \neq 0$ thì hệ véc tơ là phụ thuộc tuyến tính.
# 
# Một hệ véc tơ độc lập truyến tính thì tạo ra một không gian véc tơ. Khi đó mọi véc tơ khác có thể được biểu diễn dưới dạng tổ hợp tuyến tính của những véc tơ này.
# 
# **Hạng của ma trận là gì?**
# 
# Hạng của ma trận là số lượng các chiều trong không gian véc tơ được sinh ra bởi các cột (hoặc dòng) của ma trận $\mathbf{A}$. Nó chính bằng số lượng lớn nhất các cột (hoặc dòng) độc lập truyến tính của ma trận $\mathbf{A}$.

# In[58]:


np.linalg.matrix_rank(A)


# ## 3.6.4. Định thức (_determinant_) của ma trận
# 
# Giả sử ta có một ma trận vuông $\mathbf{M}$ như sau:
# 
# $$\mathbf{M}=\begin{bmatrix} 
# m^{1}_{1} & m^{1}_{2} & \dots & m^{1}_{n}\\ 
# m^{2}_{1} & m^{2}_{2} & \dots & m^{2}_{n}\\ 
# \dots & \dots & \ddots & \dots\\ 
# m^{n}_{1} & m^{n}_{2} & \dots & m^{n}_{n}\\ 
# \end{bmatrix}$$
# 
# 
# Định thức của một ma trận là một giá trị được tính dựa theo công thức:
# 
# $$\det(\mathbf{M})= \sum_{\sigma} \text{sgn}(\sigma) m^{1}_{\sigma(1)}m^{2}_{\sigma(2)}\cdots m^{n}_{\sigma(n)}= m^{1}_{1}m^{2}_{2}\cdots m^{n}_{n}.$$
# 
# Trong đó $\sigma=\{\sigma(1), \sigma(2), \dots, \sigma(n)\}$ là một phép hoán vị các thành phần của tập thứ tự ban đầu $O = \{1, 2, \dots, n\}$ và $\text{sgn}(\sigma)$ là biểu thức nhận hai gía trị $\{1, -1\}$. Nếu số lần hoán vị $\sigma$ để thu được tập $O$ là chẵn thì nhận gía trị 1 và lẻ thì nhận giá trị -1. 
# 
# Định thức có ý nghĩa rất quan trọng đối với ma trận vì nó cho phép chúng ta biết được các véc tơ cột (hoặc dòng) của ma trận đó có độc lập tuyến tính hay không? Hệ phương trình tạo bởi ma trận đó bao nhiêu nghiệm? Thậm chí chúng ta có thể tính được nghiệm của ma trận theo công thức nghiệm Jacobian.

# In[59]:


# Tính định thức của A
np.linalg.det(A)


# ## 3.6.5. Trace của ma trận
# 
# Trace của một ma trận là tổng của các phần tử nằm trên đường chéo chính.
# 
# $$trace{(\mathbf{A})} = \sum_{i=1}^{n}a_{ii}$$

# In[60]:


print("matrix A: \n", A)
print("trace of A: ", np.trace(A))


# ## 3.6.6. Chuẩn Frobenious 
# 
# Tương tự như chuẩn $L_2$ đối với véc tơ. Chuấn Frobenious của một ma trận bằng căn bậc hai của tổng bình phương các thành phần trong ma trận đó.
# 
# $$||\mathbf{A}_{F}||=\sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}^2}$$

# In[61]:


np.linalg.norm(A, ord='fro')


# Ngoài ra ta cũng dễ dàng chứng minh được rằng:
# 
# $$||\mathbf{A}||_{F}^2 = \text{trace}(\mathbf{A}\mathbf{A}^{\intercal}) = \text{trace}(\mathbf{A}^{\intercal}\mathbf{A})$$

# # 3.7. Các phép toán trên ma trận

# ## 3.7.1. Các phép cộng, trừ
# 
# Chúng ta có thể cộng hai hai ma trận có cùng kích thước như sau:
# 
# $$\mathbf{A}_{(m \times n)} + \mathbf{B}_{(m \times n)} = 
#  \begin{bmatrix}
#   a_{11}+b_{11} & a_{12}+b_{12} & \cdots & a_{1n}+b_{1n} \\
#   a_{21}+b_{21} & a_{22}+b_{22} & \cdots & a_{2n}+b_{2n} \\
#   \vdots  & \vdots  & \ddots & \vdots  \\
#   a_{m1}+b_{m1} & a_{m2}+b_{m2} & \cdots & a_{mn}+b_{mn} 
#  \end{bmatrix} $$
# 
# Các phần tử ở cùng một vị trí ở cả hai ma trận được cộng với nhau. Tức là $a_{ij} + b_{ij}$ sẽ tạo ra phần tử $c_{ij}$ trên ma trận đầu ra. Tương tự chúng ta cũng thực hiện như vậy đối với phép trừ hai ma trận.

# In[62]:


import numpy as np
A = np.array([[1, 4, 2],
              [3, 2, 1],
              [4, 6, 7]])

B = np.array([[0, 3, 4],
              [3, 4, 2],
              [4, 6, 8]])

A - B


# ## 3.7.2. Phép nhân ma trận thông thường
# 
# Phép nhân hai ma trận $\mathbf{A}_{m \times n}$ và $\mathbf{B}_{n \times p}$ sẽ tạo thành một ma trận mới kích thước $m \times p$. Khi đó phần tử $c_{ij}$ trên ma trận mới sẽ có tích bằng dòng thứ $i$ trên ma trận $\mathbf{A}$ nhân với cột thứ $j$ trên ma trận $\mathbf{B}$.
# 
# $$c_{ij} = \mathbf{A}_{i:} \mathbf{B}_{:j}$$
# 
# Trên numpy phép nhân này được thực hiện bằng `A.dot(B)` hoặc `np.dot(A, B)`.

# In[63]:


import numpy as np
A = np.array([[1, 4, 2],
              [3, 2, 1]])

B = np.array([[0, 3],
              [3, 4],
              [4, 6]])

np.dot(A, B)
# Hoặc
A.dot(B)
# Hoặc
A@B


# ## 3.7.3. Tích Hadamard (element-wise product) giữa hai ma trận
# 
# Tích Hadamard giữa hai ma trận $\mathbf{A}$ và $\mathbf{B}$ có cùng kích thước là một ma trận có cùng kích thước với $\mathbf{A}$ và $\mathbf{B}$. Khi đó mỗi một phần tử của ma trận bằng tích của các phần tử ở cùng vị trí.
# 
# $$\begin{split}
# \begin{split}\mathbf{A} \odot \mathbf{B} =
# \begin{bmatrix}
#     a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
#     a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
#     \vdots & \vdots & \ddots & \vdots \\
#     a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
# \end{bmatrix}\end{split}
# \end{split}$$
# 
# Chúng ta ký hiệu tích hadamard bằng hình trong có một dấu chấm ở tâm ($\odot$).
# 
# Bạn sẽ nhận thấy rằng tích chập hai chiều (convolutional 2D) trong mạng CNN cũng chính là tích hadarmard giữa bộ lọc và vùng local region.

# In[64]:


# Tích của hai ma trận A và B
C = np.array([[2, 1, 2],
              [1, 0, 1]])

D = np.array([[0, 2, 4],
              [2, 3, 1]])

C*D


# ## 3.7.4. Nhân ma trận với một véc tơ
# 
# Từ phép nhân hai ma trận ta có thể suy ra cách thực hiện phép nhân ma trận với một véc tơ nếu coi véc tơ cũng là một ma trận mà một chiều của nó bằng 1.
# 
# $$\mathbf{A}\mathbf{x} = 
#  \begin{bmatrix}
#   a_{11} & a_{12} & \cdots & a_{1n} \\
#   a_{21} & a_{22} & \cdots & a_{2n} \\
#   \vdots  & \vdots  & \ddots & \vdots  \\
#   a_{m1} & a_{m2} & \cdots & a_{mn} 
#  \end{bmatrix}_{(m \times n)}
#  \begin{bmatrix}
#   x_1 \\
#   x_2 \\
#   \vdots \\
#   x_n
#  \end{bmatrix}_{(n)} = \begin{bmatrix}
#   a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n \\
#   a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n \\
#   \dots   \\
#   a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n 
#  \end{bmatrix}_{(m)}$$

# In[65]:


x = np.array([1, 2, 3])
A.dot(x)


# ## 3.7.5. Nhân ma trận với một scaler
# 
# Khi nhân ma trận với một scaler chúng ta lấy từng phần tử của ma trận đó nhân với scaler.

# In[66]:


A*5


# # 3.8. Các phép toán trên véc tơ
# 
# Với hai véc tơ $\mathbf{a} = (a_1, a_2, \dots, a_n)$ và $\mathbf{b} = (b_1, b_2, \dots, b_n) $ có cùng kích thước. Chúng ta có một số phép toán chính được tính trên hai véc tơ này như sau:
# 
# * **Phép cộng, trừ:**
# 
# $$\mathbf{a}+\mathbf{b} = (a_1 + b_1, a_2 + b_2, \dots, a_n + b_n)$$
# 
# * **Tích vô hướng:** Tích vô hướng giữa hai véc tơ là một số vô hướng.
# 
# $$\langle \mathbf{a}, \mathbf{b}\rangle = (a_1b_1+a_2b_2+\dots+a_nb_n)$$
# 
# * **Tích có hướng:** Tích có hướng giữa hai véc tơ là một véc tơ.
# 
# $$\mathbf{a}.\mathbf{b} = (a_1b_1, a_2b_2, \dots , a_nb_n)$$
# 
# 
# * **Độ đo cosine (cosine-similarity):** Đây là độ đo được sử dụng rất phổ biến để tính toán sự tương đương giữa hai véc tơ đặc trưng. Chẳng hạn như trong faceid chúng ta mã hoá các bức ảnh face-crop dưới dạng những véc tơ và nhận diện người dùng bằng cách tính tương quan véc tơ có sẵn trong database với face của người đó.
# 
# $$\text{cosine}(\mathbf{a}, \mathbf{b}) = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2}\sqrt{\sum_{i=1}^{n} b_i^2}}$$

# In[67]:


a = np.array([1, 2, 3])
b = np.array([4, 5, 5])
# Phép cộng
a+b
# Tích vô hướng
a.dot(b)
# Tích có hướng
a*b
# cosine similarity
a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b))


# In[68]:


# Hoặc cách khác để tính cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
a = np.array([[1, 2, 3]])
b = np.array([[4, 5, 5]])

cosine_similarity(a, b)


# * **Chuẩn của véc tơ:** Về khái niệm chuẩn của véc tơ bạn có thể xem tại [chuẩn véc tơ](https://phamdinhkhanh.github.io/deepai-book/ch_algebra/appendix_algebra.html#khai-niem-chuan)

# **Bài tập:** Khi nào thì hai véc tơ có độ đo _cosine_ bằng 1 và -1?

# # 3.9. Thành phần của mảng
# 
# Mỗi một mảng sẽ đặc trưng bởi các thuộc tính của nó như định dạng, kích thước, số chiều, bộ nhớ. Trong tính toán ma trận, những thuộc tính này sẽ hữu ích để kiểm tra lỗi. Ví dụ muốn biết được ma trận A có nhân được với ma trận B hay không thì chúng ta phải kiểm tra shape của ma trận A và shape của ma trận B xem số dòng của A có bằng với số cột của B không? Kiểm tra kích thước bộ nhớ cũng giúp ta ước lượng dung lượng cần lưu cho những mảng có kích thước lớn. Từ đó sẽ biết được bộ nhớ của máy tính có khả năng tải được chúng hay không ? Nếu mảng không yêu cầu độ chính xác quá cao thì có thể chuyển sang một định dạng có độ chính xác thấp hơn và tiêu tốn ít bộ nhớ hơn.

# In[69]:


print('shape: ', A.shape)
print('number dims: ', A.ndim)
print('dtype: ', A.dtype)
print('item bytes size: ', A.itemsize)
print('bytes size: ', A.nbytes)


# Ngoài ra các thuộc tính của một mảng còn được thể hiện trong `flags`.

# In[70]:


A.flags


# Từ các flags ta có thể thấy mảng `A` là mảng có thể sửa được (`WRITEABLE=TRUE`). Mảng `A` có cách sắp xếp các phần tử liên tiếp theo kiểu của ngôn ngữ C chứ không phải Fortran (`C_CONTIGUOUS = TRUE, F_CONTIGUOUS = FALSE`).

# # 3.10. Bài tập
# 
# Cho ma trận $\mathbf{A}$

# In[71]:


import numpy as np

A = np.array([[10, 1, 2],
              [9, 7, 4],
              [0, 2, 1]])

B = np.array([[2, 3, 4],
              [0, 4, 2],
              [3, 2, 1]])


# Thực hiện các phép tính:
# 
# 1) $\mathbf{A}\mathbf{B}$
# 
# 2) $\mathbf{B}\mathbf{A}^{\intercal}\mathbf{A}\mathbf{B}^{\intercal}$
# 
# 3) $\mathbf{A}\mathbf{B}^{\intercal}\mathbf{B}\mathbf{A}^{\intercal}$
# 
# Nhận xét gì về kết quả của biểu thức 3 và biểu thức 2?
# 
# 4) $\text{rank}(\mathbf{A})$
# 
# 5) $\text{det}(\mathbf{A})$
# 
# 6) $\mathbf{A}^{-1}$
# 
# 7) $\text{trace}(\mathbf{A})$
# 
# Thực hiện các phép biến đổi shape ma trận:
# 
# 8) concatenate ma trận $\mathbf{A}$ và ma trận $\mathbf{B}$ theo dòng.
# 
# 9) Reshape ma trận $\mathbf{A}$ thành véc tơ
# 
# Tính toán:
# 
# 10) Phân phối xác suất sau khi đi qua hàm softmax các dòng của $\mathbf{A}$
# 
# 11) Tìm ra nhãn dự báo cho mỗi dòng.
# 
# 12) Tính tổng, trung bình, min, max của mỗi dòng.

# # 3.11. Tài liệu

# 1. [numpy doc](https://numpy.org/doc/)
# 
# 2. [numpy w3school](https://www.w3schools.com/python/numpy/numpy_intro.asp)
# 
# 3. [numpy tutorialspoint](https://www.tutorialspoint.com/numpy/numpy_indexing_and_slicing.htm)
# 
# 4. [numpy cs231](https://cs231n.github.io/python-numpy-tutorial/)
