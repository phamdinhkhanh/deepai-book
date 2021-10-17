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

+++ {"id": "SDvD6nAQiTlv"}


Thuật ngữ:

* cây quyết định: decision tree
* node gốc: root node
* node lá: leaf node
* thông tin entropy: information entropy
* chỉ số gini: gini index
* thuần khiết: purity
* vẩn đục: impurity




# 8. Khái niệm về cây quyết định

Trong cuộc sống có rất nhiều tình huống chúng ta quan sát, suy nghĩ và ra quyết định bằng cách đặt câu hỏi. Xuất phát từ đó, trong Machine Learning có một mô hình được thiết kế dưới dạng các câu hỏi, mà ở đó các câu hỏi được sắp xếp dưới dạng cây. Đó chính là mô hình cây quyết định mà chúng ta sẽ tìm hiểu trong bài viết này. 

Vậy cây quyết định là gì? Bản chất của cây quyết định là một đồ thị có hướng được sử dụng cho việc ra quyết định. Lấy ví dụ, sau khi biết điểm thi tốt nghiệp THPT, bạn muốn xây dựng một chiến lược đăng kí ngành học bằng một loạt các lựa chọn:

* Nếu tổng ba môn của bạn là lớn hơn 28.5 bạn sẽ nộp vào ngành CNTT. 
* Trái lại, nếu điểm thi của bạn nhỏ hơn hoặc bằng 28.5 thì vẫn còn cơ hội cho bạn nếu điểm Toán cao vì điểm Toán có hệ số nhân là 2. Do đó bạn quyết định vẫn lựa chọn CNTT nếu điểm Toán được 10. Trường hợp còn lại bạn đăng ký vào ngành KTĐTVT.

Tập hợp các câu hỏi và lựa chọn của bạn có thể ở trên được khái quát thành một cây quyết định:

![](https://imgur.com/7JRSLF4.png)

+++ {"id": "j8AjAsFjvj3A"}

Cây quyết định ở sơ đồ trên còn được gọi là cây quyết định nhị phân vì một câu hỏi chỉ có hai phương án là True hoặc False. Trên thực tế có thể có những dạng cây quyết định khác nhiều hơn hai phương án cho một câu hỏi.

chúng ta có một số khái niệm liên quan tới _cây quyết định_:

* node gốc (_root node_): Là node ở vị trí đầu tiên của cây quyết định. Mọi phương án đều bắt nguồn từ node này. Ở ví dụ trên là (Tổng điểm >= 28.5)

* node cha (_parent node_): Là node mà có thể rẽ nhánh xuống những node khác bên dưới. Node bên dưới được gọi là node con.

* node con (_child node_): Là những node tồn tại node cha.

* node lá (_leaf node_): Là node cuối cùng của một quyết định. Tại đây chúng ta thu được kết quả dự báo. Node lá ở vị trí cuối cùng nên sẽ không có node con.

* node quyết định (_non-leaf node_): Những node khác node lá.

Từ sơ đồ cây quyết định ở trên, chúng ta nhận thấy một cây quyết định được cấu tạo bởi **node và cạnh**. Tại mỗi node là một câu hỏi (chính là các hình chữ nhật bo góc) dạng yes/no được đặt ra đối với một biến đầu vào. Tuỳ thuộc vào đáp án mà tiếp theo bạn sẽ rẽ sang nhánh True hoặc False. Cứ tiếp tục thực hiện rẽ nhánh như vậy một cách truy hồi cho đến khi thu được câu trả lời tại node cuối cùng.

_Lưu ý:_ Trong sklearn khi sử dụng thuật toán CART thì chúng ta xây dựng một cây nhị phân mà mỗi _node không phải lá_ chỉ gồm 2 _node con_. Trong khi đó các _cây quyết định_ (_decision tree_) sử dụng thuật toán ID3 có thể có nhiều hơn 2 _node con_.

+++ {"id": "4pkqhJ5z1Cr7"}
