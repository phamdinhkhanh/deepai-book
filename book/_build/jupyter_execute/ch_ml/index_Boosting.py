#!/usr/bin/env python
# coding: utf-8

# Thuật ngữ:
# 
# * phương pháp kết hợp: ensemble
# * bỏ túi: bagging
# * lấy mẫu tái lặp: boostrapping
# * phương pháp tăng cường: boosting.
# * phân loại nhị phân: binary classification
# * mô hình gốc cây: stump

# # 12. Phương pháp tăng cường (_Boosting_)
# 
# Ở bài trước chúng ta đã được học về mô hình [random forest](https://phamdinhkhanh.github.io/deepai-book/ch_ml/index_RandomForest.html). Đây là lớp mô hình dựa trên hai kĩ thuật chính là _phương pháp kết hợp_ (_ensemble learning_) và _bỏ túi_ (_bagging_). Trong đó _phương pháp kết hợp_ là kĩ thuật sử dụng nhiều mô hình yếu phối hợp với nhau để tạo thành một mô hình dự báo mạnh hơn; _bỏ túi_ là phương pháp huấn luyện các mô hình trên những bộ dữ liệu được _lấy mẫu tái lặp_ (_boostrapping_) từ tập dữ liệu đầu vào.
# 
# Như vậy mô hình _rừng cây_ là kết hợp của nhiều _cây quyết định_ $\hat{f}^1, \hat{f}^2, \dots, \hat{f}^p$. Những cây quyết định này được huấn luyện trên các tập dữ liệu khác nhau là $\mathcal{B}_1, \mathcal{B}_2, \dots, \mathcal{B}_p$ được _lấy mẫu tái lặp_ từ tập huấn luyện $\mathcal{D}$. Kết hợp kết quả dự báo từ nhiều cây quyết định chúng ta sẽ thu được dự báo cho từng quan sát. Một điểm đáng lưu ý là trong mô hình _rừng cây_ thì những _cây quyết định_ là hoàn toàn độc lập.
# 
# Trong bài này chúng ta cùng tìm hiểu về phương pháp (_tăng cường_) _boosting_, đây cũng là một phương pháp kết hợp các _cây quyết định_ nhưng giữa các _cây quyết định_ không hoàn toàn độc lập mà chúng có sự phụ thuộc theo chuỗi. Tức là một _cây quyết định_ được phát triển từ việc sử dụng thông tin được dự báo từ những _cây quyết định_ được huấn luyện trước đó. Trong phương pháp _tăng cường_ chúng ta không sử dụng _lấy mẫu tái lặp_ để tạo dữ liệu huấn luyện mà các mô hình được huấn luyện ngay trên dữ liệu gốc. Giống như _phương pháp kết hợp_, kết quả dự báo của mô hình là sự kết hợp của những cây quyết định con.
# 
# Có nhiều thuật toán phân loại khác nhau được phát triển dựa trên _phương pháp tăng cường_. Trong đó _AdaBoost_ là thuật toán đầu tiên được áp dụng trong bài toán _phân loại nhị phân_. Chính vì vậy, đây chính là thuật toán điển hình nhất mà chúng ta nên bắt đầu khi tiếp cận _phương pháp tăng cường_. Ngoài ra còn một số thuật toán hiện đại khác được xây dựng dựa trên _AdaBoost_, trong đó nổi bật nhất là _Gradient Boosting_. Trước tiên chúng ta sẽ cùng tìm hiểu về _AdaBoost_ theo nội dung bên dưới.

# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# Boosting.md
# ```
