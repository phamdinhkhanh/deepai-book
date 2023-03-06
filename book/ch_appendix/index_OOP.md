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

+++ {"id": "kvQGTNOPU1ky"}

# 5. Lập trình hướng đối tượng (Object Oriented Programming - OOP)

+++ {"id": "Dyvm55hubkxV"}

**Lập trình hướng đối tượng là gì?**

Các tài liệu định nghĩa lập trình hướng đối tượng (viết tắt là _OOP_) khác nhau. Theo định nghĩa từ wikipedia thì:

`Object-oriented programming (OOP) is a programming paradigm based on the concept of "objects", which can contain data and code: data in the form of fields (often known as attributes or properties), and code, in the form of procedures (often known as methods).
`

Source [Wikipedia](https://en.wikipedia.org/wiki/Object-oriented_programming)

Định nghĩa trên có thể tóm gọn rằng OOP là một phương thức lập trình kiểu mẫu dựa trên khái niệm đối tượng. Trong lập trình OOP, các hành vi và thuộc tính của một đối tượng (_object_) được được đóng gói thành những lớp (_class_) chuyên biệt. Điều này cũng giống như bạn muốn sửa ống nước thì bạn cần một hộp dụng cụ mà bên trong đóng gói tất cả các thứ bạn cần như cưa, máy khoan, keo dán, ống nước,.... với các chức năng như cưa, đục, hàn gắn, nối ống,.... Khi muốn sửa ống nước bạn chỉ cần nhớ tới đối tượng là hộp dụng cụ và lôi ra dùng dễ dàng.

Kiểu mẫu OOP là một thiết kế lập trình sáng tạo và linh hoạt vì nó có thể mô hình hoá các đối tượng trong thế giới thực thành các đối tượng trong lập trình mà ở đây những đối tượng này có dữ liệu đính kèm với nó và có thể thực hiện các chức năng nhất định.

Để minh hoạ cho phương thức hoạt động của OOP thì lập trình game là ví dụ trực quan nhất. Bạn còn nhớ trò chơi `Supper Mario` huyền thoại chứ? trong trò chơi này nhân vật chính là Mario. Để tạo ra nhân vật Mario trong thế giới lập trình thì ta phải mô phỏng lại các đặc điểm (được gọi là thuộc tính) của nhân vật như: Chiều cao, số mạng, điểm số, loại đạn đang sử dụng,.... Nhưng đó mới chỉ là phần bề nổi của nhân vật. Để chúng ta có thể điều khiển được nhân vật thì phải có các hàm chức năng riêng như: Di chuyển, nhảy, bắn đạn, cứu công chúa,.... Đó chính là ý nghĩa của phương thức và thuộc tính trong OOP.


**Ưu điểm của lập trình hướng đối tượng**

Lập trình hướng đối tượng chính là cuộc cách mạng của ngành lập trình, nó giúp cho các sản phẩm lập trình trở nên có hệ thống hơn. Trước khi lập trình hướng đối tượng ra đời thì lập trình hàm là hình mẫu phổ biến của lập trình. Lập trình hàm chỉ quan tâm tới các thủ tục (tức là các hàm) và trình tự thực hiện. Khi vận hành một chương trình theo lập trình hàm sẽ quan tâm tới thứ tự vận hành các hàm thủ tục như thế nào. Nhưng lập trình hướng đối tượng có thể giúp ta làm được nhiều việc hơn thế, đó là:

* Chương trình trở nên có hệ thống hơn: Trước khi tiến hành viết một chương trình ta phải nghĩ đến thiết kế bên dưới của chương trình. Mỗi quan hệ giữa các đối tượng trong chương trình của chúng ta như thế nào? Từ đó chúng ta sẽ đưa ra một thiết kế hệ thống cho chương trình và tiến hành viết chương trình.

* Có khả năng kế thừa: Kế thừa là một ưu điểm cực kỳ vượt trội của OOP. Khả năng kế thừa giúp cho code trở nên ngắn gọn và khoa học hơn, tiết kiệm được thời gian phát triển ứng dụng. Nhờ khả năng kế thừa mà các class con có thể sở hữu toàn bộ các phương thức và thuộc tính của class cha. Các bạn sẽ rõ hơn về khái niệm kế thừa ở những chương ví dụ bên dưới.

* Khả năng đóng gói: Phương châm của OOP là tất cả các phương thức và thuộc tính cần thiết cho một đối tượng đều được gói gọn trong một class. Nhờ vậy code của chúng ta không bị phân tán ở nhiều nơi. Khi cần sử dụng thứ gì thì có thể khởi tạo class. Tất cả các phương thức và thuộc tính đã được đóng gói trong class nên có thể truy cập và sử dụng khi gọi tên chúng.

+++ {"id": "dMh_cr8wU6j4"}
