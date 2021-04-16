# Làm sạch dữ liệu (WIP)

Sau bước EDA, ta có cái nhìn đầu tiên về phân bố của các trường dữ liệu.
Việc cần làm tiếp theo là làm sạch dữ liệu bằng cách xử lý các giá trị ngoại lệ hoặc giá trị bị khuyết.
Ngoài ra, do đặc tính cửa việc thu thập dữ liệu, các giá trị như nhau có thể được lưu trong cơ sở dữ liệu dưới dạng khác nhau hoặc có lỗi chính tả trong các dữ liệu dạng hạng mục.
Dữ liệu dạng số và dạng hạng mục cần có những cách xử lý khác nhau.

(sec_outlier_processing)=
## Xử lý các giá trị ngoại lệ

Giá trị ngoại lệ [^1] (_outliers_) trong dữ liệu là gì?

Với dạng số, dữ liệu ngoại lệ có thể là một giá trị phi thực tế như số tuổi âm, hoặc một giá trị khác xa với phần còn lại của các giá trị trong trường đó.
Với dạng hạng mục, dữ liệu ngoại lệ có thể là một giá trị phi thực tế như một hạng mục nằm ngoài những khả năng có thể xảy ra như một địa danh không có trên bản đồ.
Các giá trị có tần xuất xảy ra vô cùng thấp trong một cột dữ liệu cũng có _khả năng_ [^2] là một giá trị ngoại lệ.


### Dữ liệu số

## Dữ liệu hạng mục

(sec_missing_data)=
## Xử lý các giá trị bị khuyết

### Dữ liệu số

### Dữ liệu hạng mục

## Chuẩn hóa dữ liệu

[^1]: Đôi khi được gọi là "ngoại lai".

[^2] Giá trị đặc biệt này cũng có thể mang lại nhiều thông tin cho việc dự đoán. Cần kiểm tra kỹ mối tương quan giữa cột dữ liệu tương ứng và cột nhãn.