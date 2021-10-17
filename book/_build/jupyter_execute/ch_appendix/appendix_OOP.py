#!/usr/bin/env python
# coding: utf-8

# # 5.1. Class và Object
# 
# Class trong OOP chính là những đơn vị thiết kế do người lập trình tạo ra nhằm thiết lập cấu trúc dữ liệu cho chương trình. Class giả lập lại các thực thể trong thực tiễn bằng cách mô hình hoá chúng trong lập trình. Ví dụ như chúng ta có thể tạo ra class User để bao quát các chức năng và thuộc tính của một người dùng trong một hệ thống. Class này sẽ bao gồm hai thành phần:
# 
# * Thuộc tính (_attribute_): name, age, gender, occupation.
# * Phương thức (_method_): buy, search, purchase, click, addToCart.
# 
# Thuộc tính là những số liệu liên quan tới người dùng, còn phương thức chính là những gì mà user có thể thực hiện được.

# ## 5.1.1. Khởi tạo class
# Class được bắt đầu bởi từ khoá `class` và theo sau từ khoá này là tên của class đó và dấu hai chấm. Những phần nằm dưới dòng chứa từ khoá `class` được gọi là class body. Nơi chúng ta sẽ định nghĩa `class` đó được khởi tạo như thế nào, có những phương thức và thuộc tính ra sao. Ví dụ chúng ta muốn khởi tạo class User ở hình thức đơn giản nhất.

# In[1]:


class User:
  pass

# Khởi tạo một object user và kiểm tra type của nó.
user = User()
print(type(user) == User)


# Như vậy class User đã được khởi tạo thành công. Nhưng đây mới chỉ là class rỗng. Chúng ta cần phải khởi tạo các thuộc tính của User thông qua hàm tạo (_constructor_).

# ### 5.1.1. Constructor trong python
# 
# 
# Thuộc tính của một class được gọi là _attribute_. Nó được định nghĩa thông qua một hàm tạo có tên là `__init__()`. Đây là một hàm được qui ước trong convention của ngôn ngữ python, có nghĩa là hàm tạo luôn luôn có tên xác định là `__init__()`.
# 
# Để khởi tạo class User chúng ta phải truyền vào các giá trị thuộc tính của `User`.

# In[2]:


class User:
  def __init__(self, name, age, gender, occupation):
    self.name = name
    self.age = age
    self.gender = gender
    self.occupation = occupation


# Các gía trị bên trong một hàm số được gọi là đối số (_argument_). Chẳng hạn bên trong hàm `__init__()` thì `self, name, age, gender, occupation` chính là đối số. Đối với hàm tạo `__init__()` thì đối số đầu tiên luôn luôn là `self`. Trong tiếng Anh khi bạn nói your-self hoặc my-self có nghĩa là chính bạn. Từ khoá `self` ở đây thể hiện cho chính bản thân class. Nhờ từ khoá self mà chúng ta có thể truy cập được vào các thuộc tính của class User thông qua `self.name, self.age, self.gender, self.occupation` và gán cho nó bằng giá trị của đối số của hàm tạo thông qua nhóm câu lệnh:
# 
# ```
# self.name = name
# self.age = age
# self.gender = gender
# self.occupation = occupation
# ```
# 
# Khi khởi tạo một class thì hàm tạo sẽ được gọi đầu tiên và bên trong một class ta phải truyền vào các đối số (trong ví dụ gồm `name, age, gender, occupation`) được khai báo ở hàm tạo. Cụ thể như sau:

# In[3]:


user = User(name='Pham Dinh Khanh', age=27, gender='male', occupation='AI Engineer')
print(user.name, user.age, user.gender, user.occupation)


# Như vậy các thuộc tính của class `User` có thể thay đổi theo giá trị của đối số được truyền vào hàm tạo. Ngoài ra chúng ta còn có những thuộc tính **không thay đổi** của class. Những thuộc tính bất biến này có thể được khởi tạo bên ngoài hàm tạo.

# In[4]:


class User:
  nation = 'VietNam'
  def __init__(self, name, age, gender, occupation):
    self.name = name
    self.age = age
    self.gender = gender
    self.occupation = occupation

user = User(name='Pham Dinh Khanh', age=27, gender='male', occupation='AI Engineer')
print(user.nation, user.name, user.age, user.gender, user.occupation)


# Như vậy ta thấy thuộc tính nation đã được gán giá trị mặc định là 'VietNam' và chúng ta không thể thay đổi được chúng thông qua hàm tạo.
# 
# Như vậy bạn đã biết cách tạo một class trong python rồi phải không. Hãy làm bài tập sau đây để củng cố kiến thức.
# 
# **Bài tập**: Hãy khởi tạo một class là User với các thuộc tính trong hàm tạo là `user, password, totalAmount, totalClick` và thuộc tính bất biến là `customerType=B2C`.

# ### 5.1.2. Khởi tạo một object
# 
# Chúng ta xem class như là một bản thiết kế thì object chính là thể hiện của class ở thực tại. Cụ thể hơn, chúng ta đã có class User, nhưng muốn dùng được class đó thì phải khởi tạo chúng và gán chúng vào một biến. Khi đó các thuộc tính và phương thức của nó mới được lưu xuống bộ nhớ để có thể làm việc được. `user1, user2` Bên dưới chính là các object của class User.

# In[5]:


user1 = User(name='Pham Dinh Khanh', age=27, gender='male', occupation='AI Engineer')
user2 = User(name='Pham Dinh Khanh', age=27, gender='male', occupation='AI Engineer')


# Mặc dù `user1, user2` cùng là object của class User và có cùng một cách khởi tạo nhưng chúng được lưu trữ ở hai ô nhớ khác nhau và chúng là hai thực thể khác biệt nhau.

# In[6]:


user1 == user2


# ### 5.1.3. Phương thức (method)
# 
# Ở trên chúng ta đã biết về cách khởi tạo một object thông qua hàm tạo. Nhưng object đó mới chỉ gồm các thuộc tính mà chưa có phương thức. Phương thức là những chức năng của đối tượng. Ví dụ như User có các chức năng `buy, search, purchase, click, addToCart`. Trong class thì phương thức là những hàm được định nghĩa bên ngoài với hàm tạo.

# In[7]:


class User:
  nation = 'VietNam'
  def __init__(self, name, age, gender, occupation):
    self.name = name
    self.age = age
    self.gender = gender
    self.occupation = occupation

  # Các hàm buy, search là những hàm phương thức.
  def buy(self, item):
    print('you bought {}'.format(item))
  
  def search(self, term):
    print('you search: {}'.format(term))

user = User(name='Pham Dinh Khanh', age=27, gender='male', occupation='AI Engineer')
print(user.nation, user.name, user.age, user.gender, user.occupation)


# Trong ví dụ trên thì các hàm `buy, search` là những hàm phương thức. Những hàm này phải có đối số đầu tiên là `self` để khi khởi tạo object chúng ta có thể truy cập được hàm. Để truy cập vào một hàm phương thức thì chúng ta có thể dùng `object.function_name()`.

# In[8]:


user.buy('a Book')
user.search('AI Book')


# Như chúng ta thấy các phương thức đã được thực thi.
# 
# **Bài tập:** Thử khởi tạo các hàm phương thức `buy` và `search` mà không có đối số `self` và truy cập chúng. Hãy nhận xét về kết quả nhận được.

# # 5.2. Tính kế thừa
# 
# Trong di truyền học thì con cái thường kế thừa những đặc điểm của cha mẹ như diện mạo, dọng nói, tính cách,.... Trong lập trình những class con cũng có thể kế thừa lại các thuộc tính và phương thức của các class cha mẹ của chúng.
# 
# Để kế thừa lại một class thì chúng ta để lại tên class cha trong dấu `()` của dòng lệnh class ở vị trí đầu tiên.

# In[9]:


# SpecialUser kế thừa lại User bằng cách để User trong dấu ()
class SpecialUser(User):
  pass

user = SpecialUser(name='Pham Dinh Khanh', age=27, gender='male', occupation='AI Engineer')
print(user.nation, user.name, user.age, user.gender, user.occupation)
user.buy('a Book')
user.search('AI Book')


# Như chúng ta có thể thấy, mặc dù không qui định bất kỳ phương thức và thuộc tính nào nhưng SpecialUser đã sở hữu hoàn toàn hàm tạo, các phương thức và thuộc tính của class User.
# 
# Để kiểm tra xem một object có phải được kế thừa từ một class khác không thì chúng ta dùng câu lệnh `isinstance(object, Class)`.

# In[10]:


print(type(user))
isinstance(user, User)


# Ta thấy mặc dù type của object user là class `SpecialUser` nhưng do được kế thừa lại từ class `User` nên object user vẫn là một thể hiện (_instance_) của User.

# ## 5.2.1 Override & Extend
# 
# Bên cạnh việc kế thừa thì con cái vẫn có những đặc trưng riêng khác với cha mẹ. Những đặc trưng riêng này có thể được mở rộng (_extend_) dưới dạng những phương thức và thuộc tính mới hoặc viết đè (_override_) lên những phương thức và thuộc tính đã có sẵn từ cha mẹ. Cần phân biệt hai khái niệm mở rộng và viết đè. Mở rộng có nghĩa rằng phương thức và thuộc tính đó chưa từng có ở cha mẹ. Ví dụ cha mẹ chưa từng đi du học nhưng con cái đi du học. Do đó ta phải tạo ra các phương thức và thuộc tính mới cho class con. Trong khi viết đè là việc ta sửa lại những phương thức và thuộc tính mà ở con cái khác với cha mẹ. Ví dụ như con cái cao mà cha mẹ lại thấp.

# ## 5.2.2. Ví dụ về Extend
# 
# Cũng trên class SpecialUser, chúng ta mở rộng thêm một chức năng nữa mà chỉ class SpecialUser mới có là `getVoucher()`.

# In[11]:


class SpecialUser(User):
  def getVoucher(self, value):
    print("You get a voucher valuing {}".format(value))

user = SpecialUser(name='Pham Dinh Khanh', age=27, gender='male', occupation='AI Engineer')
user.getVoucher('100$')


# Hàm `getVoucher()` là một hàm mở rộng mà chỉ `SpecialUser` mới có. Hàm này không xuất hiện tại `User`.

# ## 5.2.3. Ví dụ về Override
# 
# Tiếp theo chúng ta sửa lại phương thức `buy()` sao cho phương thức này có thêm một đối số nữa là amount.

# In[12]:


class SpecialUser(User):
  def buy(self, item, amount):
    print('you bought {} with the amount of {}'.format(item, amount))

user = SpecialUser(name='Pham Dinh Khanh', age=27, gender='male', occupation='AI Engineer')
user.buy('a book', '100$')


# # 5.3. Module và package
# 
# **Module, package là gì và tại sao cần module, package?**
# 
# Module hiểu ngắn gọn là một file python mà trong đó chứa các khai báo và định nghĩa về hàm số và biến. Các chương trình python sẽ được thiết kế sao cho nội dung được chia nhỏ về các files module để dễ dàng quản lý. Chúng ta có thể dễ dàng import lại các khai báo và định nghĩa từ module này sang module khác.
# 
# Đối với những chương trình mà thường xuyên được sử dụng lặp lại ở những chương trình khác thì chúng ta sẽ tìm cách đóng gói chúng thành những packages. Như vậy khi tái sử dụng chúng, chúng ta chỉ việc gọi lại từ thư viện của interpreter mà không cần viết lại code. Một package sẽ có thiết kế bao gồm nhiều files module.
# 
# **Ví dụ về module, package và cách import chúng**
# 
# Ví dụ trong câu lệnh import matplotlib bên dưới.

# In[13]:


import matplotlib.pyplot as plt


# Thì `matplotlib` chính là package vì nó là thư mục bên ngoài cùng và chứa các module bên trong. `pyplot` chính là một module của `matplotlib`. Trong câu lệnh trên thì chúng ta đã import module `pyplot` dưới một tên định danh là `plt`. Tên này sẽ được sử dụng thay cho pyplot trong toàn bộ file code.
# 
# Ngoài ra chúng ta cũng có thể import một module trong thư viện bằng câu lệnh `from package import module` như bên dưới.

# In[14]:


from matplotlib import pyplot as plt


# **Ví dụ về module và package**
# 
# Giả sử chúng ta cần thiết kế một hệ thống thương mại điện tử đơn giản mà ở đó có các classes như `User, Item và Order`. Khi đó bạn tạo một package `ecommerce` được chứa trong thư mục cùng tên là `ecommerce`. Các module được lưu trữ bên trong package lần lượt gồm `user.py, item.py, order.py` có nội dung như sau:

# In[15]:


# file user.py
class User:
  nation = 'VietNam'
  def __init__(self, name, age, gender, occupation):
    self.name = name
    self.age = age
    self.gender = gender
    self.occupation = occupation

  # Các hàm buy, search là những hàm phương thức.
  def buy(self, item):
    print('you bought {}'.format(item))
  
  def search(self, term):
    print('you search: {}'.format(term))


# In[16]:


# file item.py
class Item:
    def __init__(self, item_id, item_name, item_price):
        self.item_id = item_id
        self.item_price = item_price
        self.item_name = item_name


# Tiếp theo trong `order` chúng ta sẽ cần sử dụng nội dung của các classes `User` và `Item` nên cần phải import những module này trong module order như bên dưới.

# In[17]:


get_ipython().run_cell_magic('script', 'echo skipping', "\n# file order.py\nfrom user import User\nfrom item import Item\n\nclass Order:\n    def __init__(self, user, item, item_quant):\n        self.user = user\n        self.item = item\n        self.item_quant = item_quant\n    \n    def cost(self):\n        value = self.item_quant*self.item.item_price\n        return value\n    \nif __name__ == '__main__':\n    user = User(name='Pham Dinh Khanh', age=27, gender='male', occupation='AI Engineer')\n    item = Item(item_id='123', item_name='keo vuốt tóc', item_price=50.000)\n    order = Order(user=user, item=item, item_quant=2)\n    total_cost = order.cost()\n    print(total_cost)")


# Trên đây là một ví dụ đơn giản về cách tổ chức và thiết kế một package gồm các modules bên trong nó. Đây là một nội dung quan trọng bởi vì trong những dự án lớn thì chúng ta sẽ thường xuyên phải thiết kế chương trình thành các modules và packages để giúp cho code trở nên ngắn gọn và dễ quản lý hơn.
# 
# # 5.4. Tổng kết
# 
# Như vậy qua bài viết này bạn đã được làm quen với lập trình hướng đối tượng trong python. Đây là một hình mẫu thiết kế được áp dụng trên nhiều ngôn ngữ lập trình hiện đại như `Java, C++, C#` nên những gì bạn học được sẽ trang bị cho bạn tư tưởng về OOP khi bạn học bất kỳ ngôn ngữ lập trình nào khác.
# 
# Tóm tắt lại, bài này mình đã giới thiệu tới các bạn:
# 
# 1. Lập trình hướng đối tượng là gì?
# 2. Cách khởi tạo một class trong python.
# 3. Xác định phương thức và thuộc tính trong class của python.
# 4. Khởi tạo một object từ class.
# 5. Tính kế thừa trong python và các ví dụ về override, extend.
# 6. Các khái niệm cơ bản và ví dụ về module và package.

# # 5.5. Bài tập
# 
# 1. Hãy xây dựng một class AI với hai thuộc tính là algorithm (tên thuật toán) và model_type (dạng thuật toán dự báo hay phân loại).
# 
# 2. Viết thêm hai phương thức cho class AI là `fit` (huấn luyện) với đầu vào là các tham số `model, X_train, y_train` và `predict` (dự báo) với đầu vào là các tham số `model, X_pred`. (Không cần phải lấy dữ liệu thật, chỉ cần print ra các đối số bên trong mỗi hàm).
# 
# 3. Gọi các hàm phương thức vừa thêm mới.
# 
# 4. Xây dựng một class DeepLearning kế thừa từ AI. Trong class này có một hàm mở rộng là `train_on_epoch` (huấn luyện trên từng epoch) với đầu vào là `model, X_train, y_train, epoch` và hàm viết đè là `fit` có thêm tham số `learning_rate`.

# # 5.6. Tài liệu 
# 
# 1. [Intro to Object-Oriented Programming (OOP) in Python
# ](https://realpython.com/python3-object-oriented-programming/)
# 2. [Python - Object Oriented - tutorialspoint
# ](https://www.tutorialspoint.com/python/python_classes_objects.htm)
# 3. [Python class and object - w3school](https://www.w3schools.com/python/python_classes.asp)
# 4. [Python3 Object Oriented Programming - Dusty Phillip](https://www.amazon.com/Python-3-Object-Oriented-Programming/dp/1849511268)
