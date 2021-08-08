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

+++ {"id": "MJ4kWcokWXGF"}

# 6. Sklearn Pipeline

Pipeline là một cách để mã hóa và tự động hóa quy trình làm việc cần thiết để tạo ra một mô hình học máy. Pipeline bao gồm nhiều bước tuần tự thực hiện mọi thứ từ trích xuất (_data extraction_) và tiền xử lý (_preprocessing data_) dữ liệu đến huấn luyện và triển khai mô hình.


![](https://imgur.com/oFng9yE.png)

**Hình 1:** Các bước trong quá trình xây dựng và triển khai mô hình.

Đối với các sản phẩm ML, các quy trình pipeline phải là phần trung tâm của sản phẩm. Nó đóng gói toàn bộ các phương pháp học được tốt nhất để tạo ra một mô hình học máy giải quyết những tác vụ cụ thể của doanh nghiệp và cho phép nhóm thực thi trên quy mô lớn. Cho dù bạn đang duy trì nhiều mô hình trên môi trường sản phẩm hay hỗ trợ một mô hình duy nhất thì để được cập nhật thường xuyên chúng ta nên cần một pipeline end-to-end cho mô hình.

**Lợi ích của ML pipeline là gì?**

Ở thời điểm ban đầu của các mô hình ML có xu hướng hướng đến giải quyết một bài toán cụ thể. Nhóm Data Scientist sẽ tập trung tạo ra một mô hình phục vụ cho một bài toán, chẳng hạn bài toán gợi ý câu search. Team thông thường bắt đầu với một qui trình thủ công mà các bước trong qui trình ML như: `thu thập dữ liệu, làm sạch dữ liệu, huấn luyện mô hình và đánh giá mô hình` dường như là được viết ngắn gọn trong một notebook. Notebook này được vận hành cục bộ để tạo ra mô hình. Sau đó chúng được chuyển giao sang cho kỹ sư để chuyển hoá thành API và áp dụng vào sản phẩm.

![](https://imgur.com/CDkSOm8.png)

**Hình 2:** Workflow của mô hình ML với qui trình thủ công.

Workflow thường mang tính đột xuất và bắt đầu bị phá vỡ khi một team bắt đầu tăng tốc chu kỳ lặp lại của mình vì các quy trình thủ công rất khó lặp lại. Do đó những khối mã lệnh được viết trên block code sẽ không còn phù hợp khi tăng tốc chu kỳ.

Khi team chuyển từ giai đoạn mà họ thỉnh thoảng cập nhật một mô hình duy nhất sang có nhiều mô hình cập nhật thường xuyên trên production, thì phương pháp tiếp cận theo pipeline đóng vai trò cực kì quan trọng. Trong workflow này, bạn không xây dựng và duy trì một mô hình mà bạn phát triển và duy trì một pipeline và pipeline chính là sản phẩm.

![](https://imgur.com/vKvZapp.png)

**Hình 3:** Workflow của mô hình ML pipeline với qui trình tự động.

Một pipeline tự động bao gồm các thành phần được sắp đặt theo một bản thiết kế về cách chúng được kết hợp với nhau để xây dựng và cập nhật toàn bộ mô hình.

Hệ thống pipeline tự động cung cấp khả năng thực thi, lặp lại pipeline dễ dàng và nhanh chóng. Nó cũng cho phép bạn xác định các đầu vào và đầu ra cần thiết được sử dụng trong mô hình. Thông qua pipeline, dữ liệu sẽ được biến đổi từ dạng thô sang tinh có thể huấn luyện, kiểm định và dự báo nhanh chóng. Những ưu điểm của pipeline đó là:

* Đóng gói theo qui trình: Qui trình xây dựng mô hình của một hệ thống được gói gọn lại trong một pipeline và có khả năng tái sử dụng khi cần thiết.

* Khả năng tự động hoá: Mọi bước trong qui trình của mô hình được tự động hoá mà không cần phải can thiệp vào code.

* Triển khai nhanh chu kỳ vòng lặp: Một chu kì từ thu thập dữ liệu tới triển khai mô hình có thể được triển khai ngay khi cập nhật phiên bản mới của pipeline.

* Tự động hoá quá trình kiểm thử và đo lường hiệu suất mô hình.

* Kiểm soát version của pipeline.

Nhờ những hiệu quả và tính ưu việt mà khi huấn luyện và triển khai những mô hình machine learning trên production chúng ta hầu hết sẽ tìm cách thiết kế các pipeline.

Ở chương này bạn đọc sẽ học được những kiến thức mới về pipeline được liệt kê bên dưới:

* Tiền xử lý dữ liệu cho biến phân loại và liên tục.
* Thiết kế một pipeline hoàn chỉnh bao gồm các bước tiền xử lý dữ liệu và dự báo.
* Thực hiện cross validation các mô hình khác nhau trên một pipeline.
* Cách lựa chọn metric phù hợp cho mô hình đối với bài toán phân loại và dự báo.
* Kỹ thuật gridsearch trong tìm kiếm siêu tham số (_hyperparameter_) cho mô hình.

Hãy cũng tìm hiểu các nội dung như bên dưới.