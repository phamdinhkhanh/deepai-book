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

+++ {"id": "W20pqWO-k_tv"}

# 3. Numpy

Trong python, khi làm việc với các tính toán đại số trên ma trận và véc tơ thì chúng ta chủ yếu sử dụng numpy. Numpy là viết tắt của cụm từ `numerical python` tức là thư viện số học của Python. Package này hỗ trợ hầu hết các tính toán trên dữ liệu mảng nhiều chiều (_multidimensional array_) và là một trong những package lõi của machine learning. Numpy có những ưu điểm giúp cho nó hoạt động rất nhanh trên python như:

* Được phát triển trên interface của C nên khắc phục được sự chậm chạp của xử lý đơn luồng trên python.
* Các dữ liệu trên numpy array được lưu trữ trên những vùng ô nhớ liền kề nên có tốc độ truy xuất rất nhanh.
* Các hàm tính toán đại số được tối ưu để cho tốc độ cao.

Khi bắt đầu tiếp cận machine learning, chúng ta cần thành thạo cách xử lý dữ liệu và tính toán trên numpy. Bởi vì các mô hình machine learning đều được huấn luyện và tính toán dựa trên mảng nhiều chiều của numpy. Ngoài ra numpy còn là thư viện được sử dụng nhiều trong các packages khác nằm trong hệ sinh thái machine learning của python như `scikit-learn, scipy, pandas, matplotlib` nên vai trò của nó rất quan trọng.

Hiểu được các tính toán trên numpy cũng sẽ giúp học những framework khác trong deep learning như pytorch, tensorflow, mxnet dễ dàng hơn bởi vì những packages này đều mô phỏng lại cách thức tính toán đại số tuyến tính trên numpy.

Nói về numpy là nói về một package rất rộng và bao quát của đại số tuyến tính và machine learning. Tất nhiên là trong khuôn khổ hạn hẹp của cuốn sách này không nhằm giới thiệu tất cả kiến thức về numpy mà tác giả sẽ lựa chọn ra những kiến thức cốt lõi nhất dựa trên kinh nghiệm làm việc và quá trình tham khảo tài liệu. Cụ thể hơn bài viết sẽ điểm qua các nội dung cần nhớ như: khởi tạo mảng, thay đổi hình dạng ma trận, các phép toán trên dòng và cột, các phép toán trên ma trận và trên véc tơ,.... Qua đó bạn đọc sẽ hiểu được cách làm thế nào để xử lý dữ liệu và tính toán trên numpy thông qua các ví dụ và bài tập.

+++ {"id": "kLSPbDteyqpF"}
