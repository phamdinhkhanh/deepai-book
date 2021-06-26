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

Các thuật ngữ trong bài:

* dương tính: Positive
* âm tính: Negative
* thước đo: metric
* độ chính xác: accuracy
* độ chuẩn xác: precision
* độ phủ: recall
* sai dương tính: false positive rate
* đúng dương tính: true positive rate
* tập huấn luyện: train data
* tập thẩm định: validation data
* tập kiểm tra: test data

# 5. Thước đo mô hình phân loại

Đánh giá mô hình là một khâu quan trọng trong quá trình xây dựng mô hình machine learning. Đánh giá mô hình giúp ta biết được chất lượng của mô hình như thế nào và lựa chọn được mô hình tốt nhất giúp giải quyết tác vụ của mình. Tuy nhiên để tìm được thước đo đánh giá mô hình phù hợp với từng bài toán đòi hỏi chúng ta phải hiểu về ý nghĩa, bản chất và trường hợp áp dụng của từng thước đo.

Nội dung của chương này nhằm cung cấp cho bạn đọc các kiến thức về những thước đo quan trọng, thường được áp dụng trong các mô hình phân loại trong machine learning nhưng chúng ta đôi khi còn chưa nắm vững hoặc chưa biết cách áp dụng những thước đo này sao cho phù hợp với từng bộ dữ liệu cụ thể.

Hãy cùng phân tích và tìm hiểu các thước đo này qua các ví dụ bên dưới.