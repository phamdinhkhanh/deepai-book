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

Trong python, khi làm việc với các tính toán đại số trên ma trận và véc tơ thì chúng ta chủ yếu sử dụng numpy. Numpy là viết tắt của cụm từ `numerical python` tức là thư viện số học của Python. Chính vì vậy các chức năng chính của thư viện này tập trung vào hỗ trợ và tối ưu các tính toán trên dữ liệu mảng nhiều chiều (_multidimensional array_). Numpy có những ưu điểm giúp cho nó hoạt động nhanh hơn trên python như:

* Được phát triển trên interface của C nên khắc phục được sự chậm chạp của xử lý đơn luồng trên python.

* Các dữ liệu trên numpy array được lưu trữ trên những vùng ô nhớ liền kề nên có tốc độ truy xuất rất nhanh.

* Các hàm tính toán đại số được tối ưu để cho tốc độ cao.

Ngoài ra numpy còn là thư viện được sử dụng nhiều trong các packages khác nằm trong hệ sinh thái machine learning của python như `scikit-learn, scipy, pandas, matplotlib` nên package này rất thuận tiện trong việc xử lý dữ liệu và huấn luyện mô hình.

Qua bài viết này chúng a sẽ cùng tìm hiểu những chức năng chính của numpy để giúp nó trở thành một thư viện toàn năng được sử dụng trong các tính toán đại số trên python.

+++ {"id": "kLSPbDteyqpF"}
