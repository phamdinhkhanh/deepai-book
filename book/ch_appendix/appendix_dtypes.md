# 1. Định dạng dữ liệu

Định dạng dữ liệu là hạt nhân của các ngôn ngữ lập trình, vì vậy nó rất quan trọng và thường được định nghĩa trong thành phần build-in của một ngôn ngữ. Định dạng dữ liệu cho phép chương trình của chúng ta có thể làm việc với các loại biến khác nhau như biến ký tự, biến số, biến logic, dạng chuỗi, dạng key-value, dạng tập hợp,.... Bên dưới là các data types đặc trưng trong python:

* Dạng string:	str
* Dạng số:	int, float, complex
* Dạng chuỗi:	list, tuple, range
* Dạng key-value:	dict
* Dạng tập hợp:	set, frozenset
* Kiểu logic:	bool
* Kiểu nhị phân:	bytes, bytearray, memoryview

Bạn có thể xem cách bạn khởi tạo các định dạng dữ liệu phổ biến trong python3.

![](https://imgur.com/Je5guMg.png)

**Hình 1:** Bảng danh sách các định dạng dữ liệu phổ biến trong python3 kèm theo ví dụ khởi tạo. Source [w3school](https://www.w3schools.com/python/python_datatypes.asp)

Python3 là ngôn ngữ lập trình linh hoạt, vì thế chúng ta không cần phải định nghĩa định dạng dữ liệu trước khi khởi tạo biến (hay còn gọi là ép kiểu) như những ngôn ngữ khác. Tự động định dạng dữ liệu sẽ được xác định sau khi bạn gán giá trị cho biến.