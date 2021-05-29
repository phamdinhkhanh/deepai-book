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

Pipeline là một trong những nội dung quan trọng trong quá trình huấn luyện và triển khai các mô hình machine learning. Thông qua pipeline, dữ liệu sẽ được biến đổi từ dạng thô sang tinh có thể huấn luyện và dự báo được. Một hệ thống pipeline tốt có thể tự động quá quá trình xử lý dữ liệu, huấn luyện và dự báo một cách nhanh chóng bởi pipeline có thể giúp chúng ta:

* Dễ dàng thiết kế một workflow xử lý data mạch lạch và dễ nắm bắt: Trong pipeline ở mỗi bước xử lý chúng ta có thể gán cho chúng một cái tên thể hiện ngắn gọn nội dung của chúng. Pipeline cho phép các xử lý nối tiếp nhau theo chuỗi, chẳng hạng bạn có thể tạo một pipeline gồm các bước theo thứ tự: `Category Embedding, Fill Mixing, Feature Scaling, Dimensionality Reduction, Model training`.

* Đóng gói quá trình xử lý dữ liệu theo thứ tự mong muốn của chúng: Chúng ta có thể đóng gói lại toàn bộ quá trình xử lý dữ liệu phức tạp và cồng kềnh của một hệ thống lớn trong một pipeline và tái sử dụng lại pipeline này khi cần thiết.

* Thông qua pipeline chúng ta có khả năng tái tạo lại dữ liệu: Trong quá trình xây dựng và thử nghiệm mô hình chúng ta sẽ cần thử nghiệm nhiều phương án xử lý dữ liệu khác nhau để đánh giá hiệu quả của từng pipeline lên mô hình. Nhờ việc đóng gói và lưu trữ lại pipeline mà quá trình tái tạo lại dữ liệu được xử lý bởi chúng trở nên dễ dàng.

* Pipeline cho phép ta huấn luyện và dự báo trực tiếp trên đầu vào là dữ liệu thô: Nếu không có pipeline, mô hình chỉ có thể thực hiện dự báo trên đầu vào là những giá trị đã qua xử lý. Nhờ pipeline mà ta có thể thiết kế một hệ thống end-to-end tự động hoá quá trình xử lý, huấn luyện và dự báo bằng cách gắn thêm mô hình vào sau cùng của pipeline.

Nhờ những hiệu quả và tính ưu việt mà khi huấn luyện và triển khai những mô hình machine learning trên production chúng ta hầu hết sẽ tìm cách thiết kế các pipeline.

Ở chương này mình sẽ hướng dẫn cho các bạn cách thức để xây dựng một pipeline đơn giản cho mô hình trên sklearn như thế nào. Cụ thể bạn sẽ học được:

* Tiền xử lý dữ liệu cho biến phân loại và liên tục.
* Thiết kế một pipeline hoàn chỉnh bao gồm các bước tiền xử lý dữ liệu và dự báo.
* Thực hiện cross validation các mô hình khác nhau trên một pipeline.
* Cách lựa chọn metric phù hợp cho mô hình đối với bài toán phân loại và dự báo.
* Kỹ thuật gridsearch trong tìm kiếm siêu tham số (_hyperparameter_) cho mô hình.

Hãy cũng tìm hiểu các nội dung như bên dưới.

+++ {"id": "TSg93QbcWZSR"}