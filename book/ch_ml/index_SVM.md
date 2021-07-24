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


+++ {"id": "FM-smOg--iku"}

Thuật ngữ:

* Tập hỗ trợ: support vector
* Hàm mất mát: loss function
* Không gian cao chiều: high dimensional space
* Đường biên: boundary
* Lề: margin
* Phân loại đường biên cứng SVM: hard margin SVM
* Phân loại đường biên mềm SVM: soft margin SVM
* Quá khớp: overfitting
* Bài toán gốc: primal problem
* Đối ngẫu: duality
* Bài toán đối ngẫu: duality problem
* Đối ngẫu mạnh: strong duality
* Hàm quyết định: decision function
* Hàm kernel: kernel function
* Phân tuyến: linear seperable
* Điểm hỗ trợ: support points
* véc tơ hỗ trợ: support vector
* Hàm đối ngẫu Lagrange: Lagrange Dual Function
* Hàm biến đổi phi tuyến: non-linear transformation function
* Tạo lập đặc trưng đa thức: Polynormial Features

+++ {"id": "xMf2jaTkDMYv"}

# 7. Giới thiệu về SVM

SVM là viết tắt của cụm từ _support vector machine_. Đây là một thuật toán khá hiệu quả trong lớp các bài toán phân loại nhị phân và dự báo của học có giám sát. Thuật toán này có ưu điểm là hoạt động tốt đối với những mẫu dữ liệu có kích thước lớn và thường mang lại kết quả vượt trội so với lớp các thuật toán khác trong học có giám sát. 

Ưu điểm của SVM đó là:

* Đây là thuật toán hoạt động hiệu quả với không gian cao chiều (_high dimensional spaces_).

* Thuật toán tiêu tốn ít bộ nhớ vì chỉ sử dụng các điểm trong _tập hỗ trợ_ để dự báo trong _hàm quyết định_.

* Chúng ta có thể tạo ra nhiều _hàm quyết định_ từ những hàm kernel khác nhau. Thậm chí sử dụng đúng kernel có thể giúp cải thiện thuật toán lên đáng kể.

Chính vì tính hiệu quả mà SVM thường được áp dụng nhiều trong các tác vụ phân loại và dự báo, cũng như được nhiều công ty ứng dụng và triển khai trên môi trường production. Chúng ta có thể liệt kê một số ứng dụng của thuật toán SVM đó là:

* Mô hình chuẩn đoán bệnh. Dựa vào biến mục tiêu là những chỉ số xét nghiệm lâm sàng, thuật toán đưa ra dự báo về một số bệnh như tiểu đường, suy thận, máu nhiễm mỡ,....

* Trước khi thuật toán CNN và Deep Learning bùng nổ thì SVM là lớp mô hình cực kì phổ biến trong phân loại ảnh.

* Mô hình phân loại tin tức. Xác định chủ đề của một đoạn văn bản, phân loại cảm xúc văn bản, phân loại thư rác.

* Mô hình phát hiện gian lận.

Trong bài viết này chúng ta sẽ cùng tìm hiểu về nội dung của mô hình SVM.

+++ {"id": "WEHR-nF2mkM-"}