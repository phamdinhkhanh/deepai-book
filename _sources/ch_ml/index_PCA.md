---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.3
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "0uRMq7hbD1T-"}

Thuật ngữ:

* Phân tích thành phần chính: Principle Component Analysis
* Phân tích suy biến: Singular Value Decomposition
* Phân tích riêng: Eigen Decomposition
* Véc tơ riêng: Eigenvector
* Trị riêng: Eigenvalue
* Ma trận hiệp phương sai: Covariance Matrix

# 17. Giảm chiều dữ liệu

+++ {"id": "cxQ2e2-DPcgC"}

**Giảm chiều dữ liệu là gì ?**

Có rất nhiều ví dụ về giảm chiều dữ liệu trong thực tiễn. Chẳng hạn như:

- Từ 2 chiều về 1 chiều:

Giả định dữ liệu đầu vào gồm $N$ điểm $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N$ được biểu diễn trong không gian hai chiều bởi hệ véc tơ cơ sở gồm hai véc tơ $\{x_1, x_2\}$. Trên đồ thị 2 chiều thì tập hợp những điểm này có xu hướng phân bố dọc theo một số chiều véc tơ nhất định chẳng hạn như $u_1$ trong hình minh họa.

![](https://imgur.com/9hUsWcu.jpg
)

**Hình 1:** Các điểm dữ liệu là những dấu `x`. Tập dữ liệu này có tính chất phân phối dọc theo phương véc tơ $u_1$ là đường thẳng màu đỏ.

Để giảm chiều dữ liệu, chúng ta sẽ thực hiện một phép chiếu từ mỗi điểm xuống véc tơ $u_1$. Từ dữ liệu gốc $\mathbf{x}_{i} \in \mathbb{R}^2$ ta thu được hình chiếu của chúng là giá trị $\mathbf{z}_i \in \mathbb{R}^1$ trên trục véc tơ $u_1$.

- Từ 3 chiều xuống 2 chiều:

![](https://www.researchgate.net/profile/Ken-Yano/publication/4371879/figure/fig1/AS:279414773436417@1443629081297/2D-mapping-3D-generic-face-model-M-is-mapped-to-a-planner-mesh-D-using-a-piecewise.png)

**Hình 2:** Các chiều của khuôn mặt trong không gian 3 chiều.

Trong hệ thống xác thực khuôn mặt (_face verification_) dữ liệu đầu vào là ảnh PointCloud 3D có thể được chiếu xuống không gian 2D để thu được hình ảnh theo phương chính diện mà vẫn giữ được những đặc trưng chính giúp xác thực khuôn mặt.

**Mục đích của giảm chiều dữ liệu:**

Những bộ dữ liệu lớn thường tiêu tốn nhiều bộ nhớ lưu trữ và thời gian huấn luyện. Do đó khi đối mặt với những bộ dữ liệu kích thước lớn chúng ta thường tìm cách giảm chiều dữ liệu từ không gian cao chiều (_high dimensionality_) xuống không gian thấp chiều (_low dimensionality_) mà vẫn giữ được những đặc trưng chính của dữ liệu nhưng tiết kiệm được chi phí huấn luyện và dự báo.

Lấy một ví dụ, giả sử bạn đang cần phân loại tác vụ ảnh với 1000 nhãn mục tiêu và kích thước ảnh đầu vào là `1000x1000x3`. Như vậy nếu véc tơ hóa ma trận ảnh ta thu được một véc tơ với kích thước 3 triệu chiều. Để xây dựng một mạng _thần kinh nơ ron nông_  (_shallow neural network_)với một layer kết nối toàn bộ 3 triệu điểm ảnh này tới 1000 nhãn mục tiêu sẽ cần số lượng tham số là 3 tỷ. Đây là một mạng nơ ron có kích thước quá lớn và thường vượt quá khả năng tính toán của các máy tính thông thường. Nếu huấn luyện được một mạng nơ ron khổng lồ như vậy thì khả năng mô hình gặp hiện tượng _overfitting_ cũng rất cao. Khi đối mặt với tình huống này chúng ta có thể sử dụng các phương pháp giảm chiều dữ liệu để đạt được hiệu quả tính toán và tránh _overfitting_.

Những bộ dữ liệu cao chiều cũng thường xuất hiện trong dữ liệu dạng bảng (_tabular data_).  Thông thường chúng ta sẽ không sử dụng toàn bộ các biến đầu vào mà thực hiện xếp hạng mức độ quan trọng của chúng nhằm lọc ra một phần nhỏ các biến được coi là quan trọng nhất và loại bỏ những biến nhiễu. Để xếp hạng mức độ quan trọng của biến bạn có thể xem lại nội dung chương [lựa chọn đặc trưng](https://phamdinhkhanh.github.io/deepai-book/ch_ml/FeatureEngineering.html#id3). Một cách khác cũng thường được thực hiện đó là giảm chiều dữ liệu. Phương pháp này không yêu cầu phải loại bỏ bất kì một biến đầu vào nào mà tất cả chúng sẽ được tận dụng nhằm tạo ra những biến được tổ hợp tuyến tính từ chúng. 

Trong bài toán phân cụm, các phương pháp giảm chiều dữ liệu có thể biến đổi dữ liệu về không gian hai chiều hoặc ba chiều nhằm biểu diễn dữ liệu một cách trực quan. Thông qua đó phát hiện được những bất thường dữ liệu (_anomaly detection_) và nhận biết phân bố cụm trong những bài toán học không giám sát.

Trong bài viết này chúng ta cùng tìm hiểu về thuật toán **PCA**, một phương pháp rất hiệu quả trong giảm chiều dữ liệu.

**Phương pháp PCA:**

PCA là viết tắt của cụm từ _principal component analysis_. Thuật ngữ Tiếng Việt còn gọi là _phân tích thành phần chính_. Đây là một phương pháp giảm chiều dữ liệu (_dimensionality reduction_) tương đối hiệu quả dựa trên phép phân tích suy biến (_singular decomposition_) mà ở đó chúng ta sẽ chiếu các điểm dữ liệu trong không gian **cao chiều** xuống một số ít những véc tơ thành phần chính trong không gian **thấp chiều** mà đồng thời vẫn bảo toàn tối đa **độ biến động** của dữ liệu sau biến đổi. Ưu điểm của PCA đó là **sử dụng tất cả** các biến đầu vào nên phương pháp này không bỏ sót những biến quan trọng.

Để hiểu rõ về PCA trước tiên chúng ta sẽ cùng tìm hiểu về phương pháp phân tích suy biến (_Singular Decomposition - SVD_).

