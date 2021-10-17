#!/usr/bin/env python
# coding: utf-8

# Các thuật ngữ trong bài:
# 
# * dương tính: Positive
# * âm tính: Negative
# * thước đo: metric
# * độ chính xác: accuracy
# * độ chuẩn xác: precision
# * độ phủ: recall
# * sai dương tính: false positive rate
# * đúng dương tính: true positive rate
# * tập huấn luyện: train data
# * tập thẩm định: validation data
# * tập kiểm tra: test data
# 
# # 5. Thước đo mô hình phân loại
# 
# Đánh giá mô hình là một khâu quan trọng trong quá trình xây dựng mô hình machine learning. Đánh giá mô hình giúp ta biết được chất lượng của mô hình như thế nào và lựa chọn được mô hình tốt nhất giúp giải quyết tác vụ của mình. Tuy nhiên để tìm được thước đo đánh giá mô hình phù hợp với từng bài toán đòi hỏi chúng ta phải hiểu về ý nghĩa, bản chất và trường hợp áp dụng của từng thước đo.
# 
# Nội dung của chương này nhằm cung cấp cho bạn đọc các kiến thức về những thước đo quan trọng, thường được áp dụng trong các mô hình phân loại trong machine learning nhưng chúng ta đôi khi còn chưa nắm vững hoặc chưa biết cách áp dụng những thước đo này sao cho phù hợp với từng bộ dữ liệu cụ thể.
# Cụ thể chúng ta sẽ nắm được ý nghĩa, cách thức áp dụng của những thước đo như:
# 
# * Độ chính xác (_accuracy_): Thước đo thông dụng nhất của mô hình phân loại.
# * Độ chuẩn xác (_precision_): Thước đo tỷ lệ dự báo đúng được tính trên mẫu được dự báo là _dương tính_ (_positive_) và trường hợp áp dụng chúng.
# * Độ nhạy (_recall_), tương tự như _độ chuẩn xác_ nhưng được tính trên tỷ lệ của những mẫu thực tế là _dương tính_. 
# * f1 score: Trung bình điều hoà của _độ chuẩn xác_ và _độ nhạy_. Trường hợp nào thì nên áp dụng chúng thay cho _độ chuẩn xác_.
# * chỉ số AUC và đường cong ROC, các hình dạng đặc trưng của ROC và ý nghĩa trong việc đánh giá phẩm chất của các mô hình phân loại nhị phân của học có giám sát.
# * chỉ số Gini và đường cong Lorenz trong đo lường bất bình đẳng phân phối. Ví dụ áp dụng đối với bài toán phân loại tín dụng.
# 
# Hãy cùng phân tích và tìm hiểu các thước đo này qua các ví dụ trực quan bên dưới.
# 
# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# modelMetric.md
# ```
