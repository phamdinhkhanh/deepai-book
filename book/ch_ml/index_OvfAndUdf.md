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

+++ {"id": "vDhf13ApiSAZ"}

Trước khi đi vào chương này chúng ta sẽ cùng tìm hiểu các thuật ngữ được đối sánh giữa Tiếng Việt và Tiếng Anh:

* độ chệch: _bias_
* phương sai: _variance_
* quá khớp: _overfitting_
* vị khớp: _underfitting_


# 4. Độ chệch (_bias_) và phương sai (_variance_)

Chắc hẳn trong quá trình xây dựng mô hình bạn đã từng đối mặt với vấn đề mô hình dự báo tốt trên tập huấn luyện những không dự báo tốt trên tập kiểm tra. Trước khi đọc bài viết này, bạn không hiểu nguyên nhân từ đâu và khắc phục như thế nào. Bài viết này sẽ cung cấp cho bạn các kiến thức liên quan tới lỗi mô hình, cách phòng tránh cũng như khắc phục chúng.

**Độ chệch (_bias_) và phương sai (_variance_) là gì?**

Năng lực của những mô hình phân loại và dự báo trong lớp các mô hình học có giám sát của machine learning thường được thể hiện qua hai khía cạnh độ chệch (_bias_) và phương sai (_variance_). Hiểu được chính xác ý nghĩa của hai khái niệm này giúp chúng ta tạo ra những mô hình ít chệch và có độ chính xác đồng đều trên cả tập huấn luyện và tập kiểm tra.

**Độ chệch** là sai khác giữa giá trị dự báo và giá trị ground truth của một mô hình. Khi xây dựng mô hình chúng ta mong muốn sẽ tạo ra độ chệch thấp. Điều đó đồng nghĩa với giá trị dự báo sẽ gần với ground truth hơn. Thông thường những mô hình **quá đơn giản** được huấn luyện trên những bộ dữ liệu **lớn** sẽ dẫn tới độ chệch lớn. Hiện tượng này còn được gọi là mô hình bị chệch. Nguyên nhân của bị chệch là do mô hình quá đơn giản trong khi dữ liệu có mối quan hệ phức tạp hơn và thậm chí nằm ngoài khả năng biểu diễn của mô hình. Vì vậy trong tình huống này để giảm bớt độ chệch thì chúng ta thường sử dụng mô hình **phức tạp hơn** để tận dụng khả năng biểu diễn tốt hơn của chúng trên những tập dữ liệu kích thước lớn. Tuy nhiên một mô hình quá phức tạp cũng có khả năng xảy ra hiện tượng phương sai.

**Phương sai** được hiểu là hiện tượng mô hình của bạn dự báo ra giá trị **thiếu tổng quát**. Yếu tố thiếu tổng quát được thể hiện qua việc giá trị dự báo có thể khớp tốt mọi điểm trên tập huấn luyện nhưng rất **dao động** xung quanh giá trị ground truth trên tập huấn luyện. Những lớp mô hình **phức tạp** được huấn luyện trên tập huấn luyện **nhỏ** thường xảy ra hiện tượng phương sai cao và dẫn tới việc học giả mạo thông qua bắt chước dữ liệu hơn là học qui luật tổng quát.

Khi mô hình có độ chệch lớn hoặc phương sai lớn đều ảnh hưởng tới hiệu suất dự báo. Vì vậy chúng ta cần giảm thiểu chúng để tăng cường sức mạnh cho mô hình. 

Giữa phương sai và độ chệch có một sự đánh đổi qua lại. Chúng ta cùng phân tích kỹ hơn sự đánh đổi này ở mục tiếp theo.
+++ {"id": "Tgdi_h3HQlGV"}