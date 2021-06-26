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

# 6. Ứng dụng mô hình scorecard

Một trong những ứng dụng rất quan trọng trong các bài toán phân loại của học có giám sát trong machine learning là xếp hạng tín nhiệm khách hàng thông qua mô hình scorecard. Đây là một mô hình giúp lượng hóa mức độ tín nhiệm của cá nhân hoặc tổ chức thành một điểm tín nhiệm để làm cơ sở quyết định về hạn mức, lãi suất và kì hạn cho vay.

Đầu vào của mô hình Scorecard gồm các thông tin nằm trong hồ sơ cá nhân của khách hàng. Chúng được thu thập chủ yếu từ hai nguồn là cục tín dụng quốc gia và data warehouse của ngân hàng. Những trường thông tin này thường được khuyến nghị theo tiêu chuẩn của hội nghị Basel. Các thông tin được chia thành các nhóm cơ bản bao gồm:

* Nhân khẩu học (demographic): Là những thông tin liên quan đến đặc điểm cá nhân như trình độ học vấn, thu nhập, giới tính, độ tuổi, nghề nghiệp, trạng thái hôn nhân, qui mô gia đình, số người phụ thuộc,….

* Lịch sử tín dụng (credit history): Đây là những thông tin được quản lý tập trung tại cục tín dụng (bureau credit). Dữ liệu lịch sử vay của khách hàng được tổng hợp từ toàn bộ các ngân hàng hoạt động trên lãnh thổ của một quốc gia vào một data center. Như vậy ngân hàng có thể kiểm tra chéo thông tin tín dụng của khách hàng từ những ngân hàng khác.

* Thông tin giao dịch: Lịch sử giao dịch trên các thẻ tín dụng hoặc thẻ ATM sẽ đánh giá một phần nào đó về năng lực tài chính của khách hàng. Do đó thông tin này rất hữu ích đối với dự báo khả năng trả nợ.

* Thông tin tài sản đảm bảo: Đây là một thông tin đi kèm với các khoản vay thế chấp. Gía trị của tài sản đảm bảo sẽ là phương tiện cover rủi ro trong trường hợp khách hàng vỡ nợ.

Điểm nhấn trong kỹ thuật xây dựng mô hình scorecard đó là phương pháp tiền xử lý dữ liệu giúp biến đổi các biến đầu vào thành đặc trưng (_features_) có sức mạnh dự báo dựa trên phương pháp _trọng số chứng cứ_ (_weight of evidence - WOE_). Đây là một kỹ thuật tiền xử lý khá đặc trưng được áp dụng lâu đời trong `credit risk`.

Sau khi đi qua tiền xử lý dữ liệu WOE, chúng ta sẽ thu được những đặc trưng tốt và tiếp tục sử dụng hồi qui Logistic để dự báo xác suất vỡ nợ của chủ thể. Sau đó dựa trên mô hình hồi qui để xây dựng một thang điểm cho mỗi chủ thể dựa trên tổng điểm được đánh gía trên từng đặc trưng. Tổng điểm này được gọi là điểm tín nhiệm (credit score) có tác dụng lượng hoá mức độ tín nhiệm cho khách hàng.