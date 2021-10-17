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

+++ {"id": "WY0fZSZWHju7"}

# 4. Matplotlib

Trong python có khá nhiều packages dựng đồ thị như `plotly, bokeh, ggplot, pygal, seaborn....`  nhưng phổ biến nhất là matplotlib. Sở dĩ matplotlib phổ biến nhất vì nó là một trong những package được phát triển đầu tiên trên python về dựng đồ thị. Những package được phát triển muộn hơn hầu hết đều kế thừa và phát triển lại các tính năng của nó. Bên cạnh đó, các package như pandas, seaborn thậm chí còn đóng gói các hàm của matplotlib vào bên trong các chức năng của mình. Đây đều là những packages mạnh về xử lý dữ liệu và visualization.

Ngoài ra, matplotlib là một công cụ dựng đồ thị rất mạnh hỗ trợ hầu hết các biểu đồ 2D, 3D phổ biến. Các biểu đồ được vẽ trên matplotlib có thể được tuỳ biến và can thiệp sâu để điều chỉnh style và định dạng. 

Tuy nhiên matplotlib vẫn tồn tại một số hạn chế so với những packages khác đó là: biểu đồ được vẽ trên matplotlib đó là không có khả năng public và chia sẻ đồ thị thông qua API như plotly; matplotlib chưa vẽ được các biểu đồ có thể tương tác được (như có nút ấn ẩn hiện, kéo thả, bộ lọc). Mặc dù vậy, với giới nghiên cứu và data scientist thì sự đa dạng của các biểu đồ của matplotlib đã đáp ứng được phần lớn các mục đích viết báo cáo khoa học, phân tích dữ liệu, phân tích kinh doanh.

Tiếp theo chúng ta sẽ cùng tìm hiểu những cách vẽ biểu đồ cơ bản trên matplotlib như `line, barchar, scatter, pie, boxplot, area`.

