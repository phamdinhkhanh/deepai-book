#!/usr/bin/env python
# coding: utf-8

# # 6. Sklearn Pipeline
# 
# **Pipeline là gì?**
# 
# Pipeline là một cách để mã hóa và tự động hóa các công đoạn và quy trình làm việc cần thiết để tạo ra một mô hình học máy. Pipeline bao gồm nhiều bước tuần tự thực hiện mọi thứ từ trích xuất dữ liệu (_data extraction_) và tiền xử lý dữ liệu (_preprocessing data_) cho đến huấn luyện và triển khai mô hình.
# 
# 
# ![](https://imgur.com/oFng9yE.png)
# 
# **Hình 1:** Các bước trong quá trình xây dựng và triển khai mô hình.
# 
# Đối với các hệ thống sử dụng mô hình ML, thì quy trình pipeline là phần trung tâm của sản phẩm. Nó đóng gói toàn bộ các phương pháp xử lý dữ liệu tốt nhất để tạo ra một mô hình học máy phù hợp nhất cho một bộ dữ liệu cụ thể. Ngoài ra pipeline còn cho phép mô hình thực thi trên quy mô lớn. Một thiết kế pipeline end-to-end sẽ cho phép hệ thống của bạn cập nhật một cách thường xuyên các mô hình học máy một cách nhanh chóng.
# 
# **Qui trình ML thủ công**
# 
# Ở thời điểm ban đầu của các mô hình ML hướng tới giải quyết một bài toán cụ thể mà doanh nghiệp đối diện, chẳng hạn bài toán gợi ý câu search hay bài toán phát hiện gian lận khách hàng. Team data science thông thường bắt đầu với một qui trình thủ công mà các bước trong qui trình xây dựng mô hình như: `thu thập dữ liệu, làm sạch dữ liệu, huấn luyện mô hình và đánh giá mô hình` sẽ được viết ngắn gọn trong một notebook. Notebook này được vận hành cục bộ để tạo ra phiên bản mô hình thô sơ chưa thế triển khai ngay. Sau đó chúng được chuyển giao sang cho Team engineer để chuyển hoá thành API và áp dụng vào sản phẩm.
# 
# ![](https://imgur.com/CDkSOm8.png)
# 
# **Hình 2:** Workflow của mô hình ML trong qui trình thủ công. Phần màu xám `Manual Experement Steps` là những bước trong qui trình thủ công được thực hiện bởi Team data science. Đầu ra của bước này là _trained model_, để triển khai được _trained model_ lên môi trường production dưới dạng API (chính là các _Model serving_ trên sơ đồ) thì _trained model_ cần được giao cho team Engineer.
# 
# Workflow trong qui trình thủ công thường mang tính đột xuất và bắt đầu bị phá vỡ khi Team data science tăng tốc chu kỳ mô hình. Các quy trình thủ công rất khó lặp lại vì các xử lý không được đóng gói. Do đó những khối mã lệnh được viết trên block code của notebook sẽ không còn phù hợp để tăng tốc chu kỳ của mô hình.
# 
# **Pipeline tự động**
# 
# Khi Team data science chuyển sang giai đoạn phát triển với qui mô lớn và cần cập nhật nhiều mô hình thường xuyên trên môi trường production, thì phương pháp tiếp cận theo pipeline đóng vai trò cực kì quan trọng. Trong workflow theo cách tiếp cận tự động hoá pipeline, bạn không phát triển và duy trì mỗi một mô hình là một sản phẩm mà thay vào đó, bạn phát triển một pipeline và pipeline chính là sản phẩm.
# 
# ![](https://imgur.com/vKvZapp.png)
# 
# **Hình 3:** Workflow của mô hình ML trong pipeline tự động. Phần màu xám `Automated Pipeline` thể hiện các thành phần trong qui trình pipeline tự động.
# 
# Một qui trình pipeline tự động bao gồm các thành phần được sắp đặt theo một bản thiết kế, trong đó qui định cách chúng được kết hợp với nhau để xây dựng và cập nhật toàn bộ mô hình.
# 
# Một hệ thống áp dụng pipeline tự động cung cấp khả năng thực thi, lặp lại dễ dàng và nhanh chóng. Nó cũng cho phép bạn xác định các đầu vào và đầu ra cần thiết được sử dụng trong mô hình. Những ưu điểm của pipeline đó là:
# 
# * Đóng gói theo qui trình: Qui trình xây dựng mô hình của một hệ thống được đóng gói lại trong một pipeline và có khả năng tái sử dụng khi cần thiết.
# 
# * Khả năng tự động hoá: Mọi bước trong qui trình của mô hình được tự động hoá mà không cần phải can thiệp vào code.
# 
# * Triển khai nhanh chu kỳ vòng lặp: Một chu kì từ thu thập dữ liệu tới triển khai mô hình có thể được triển khai ngay khi cập nhật phiên bản code mới của pipeline.
# 
# * Tự động hoá quá trình kiểm thử và đo lường hiệu suất mô hình: Mô hình sau khi được huấn luyện sẽ được triển khai ngay và kết quả đánh giá hiệu suất mô hình được thể hiện ở `Monitoring`. Team data science có thể dễ dàng thực hiện đối chiếu, so sánh, đánh giá hiệu năng mô hình.
# 
# * Kiểm soát version của pipeline: Những phiên bản mới của pipeline sẽ được tagging và caching trong một Code Repository.
# 
# Nhờ những hiệu quả và tính ưu việt mà khi huấn luyện và triển khai những mô hình ML trên production chúng ta hầu hết sẽ tìm cách thiết kế các pipeline.
# 
# 
# **Kiến thức được học ở bài này?**
# 
# Ở chương này bạn đọc sẽ học được những kiến thức mới về pipeline được liệt kê bên dưới:
# 
# * Tiền xử lý dữ liệu cho biến phân loại và liên tục.
# * Thiết kế một pipeline hoàn chỉnh bao gồm các bước tiền xử lý dữ liệu và dự báo.
# * Thực hiện cross validation các mô hình khác nhau trên một pipeline.
# * Cách lựa chọn metric phù hợp cho mô hình đối với bài toán phân loại và dự báo.
# * Kỹ thuật gridsearch trong tìm kiếm siêu tham số (_hyperparameter_) cho mô hình.
# 
# Hãy cũng tìm hiểu các nội dung như bên dưới.
# 
# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# appendix_pipeline
# ```
