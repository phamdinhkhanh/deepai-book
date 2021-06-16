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

# 2. Bài toán dự báo

Ở chương này chúng ta sẽ bắt đầu tìm hiểu về các thuật toán trong machine learning. Tôi khuyến nghị bạn đọc nắm vững kiến thức cơ bản ở các chương 1, 2, 3 về đại số tuyến tính, giải tích và xác suất bởi đây là những kiến thức bản lề để bạn nắm vững hơn kiến thức được trình bày tại những chương sau.

Như chúng ta đã biết các thuật toán của machine learning có thể được chia thành hai dạng cơ bản là bài toán học có giám sát (_supervised learning_) và bài toán học không giám sát (_unsupervised learning_). Đối với lớp bài toán học có giám sát, chúng ta sẽ tìm cách mô tả mối quan hệ giữa biến đầu vào (hoặc biến độc lập) với biến đầu ra (hoặc biến mục tiêu, biến phụ thuộc) trong khi bài toán học không giám sát sẽ tự động phân cụm dữ liệu dựa trên đặc trưng tiềm ẩn của dữ liệu đầu vào mà không yêu cầu phải có nhãn cho từng quan sát. Do đó chi phí chuẩn bị và gán nhãn dữ liệu cho những bài toán học có giám sát là tốn kém hơn nhiều so với học không giám sát.

Trong học có giám sát chúng ta lại chia thành lớp bài toán phân loại (đối với biến đầu ra rời rạc) và dự báo (đối với biến đầu ra liên tục). Những lớp mô hình phân loại giúp ta trả lời cho câu hỏi quan sát này có nhãn là gì? Trong khi mô hình dự báo sẽ trả lời cho câu hỏi giá trị của một quan sát được dự báo có độ lớn bằng bao nhiêu? Các tác vụ dự báo thường được sử dụng nhiều trong dự báo chuỗi thời gian, kinh tế lượng, dự báo giá cả, sản lượng, lợi suất,.... Đây là những bài toán có tính ứng dụng cao và thường đóng vai trò quan trọng trong rất nhiều các lĩnh vực khác nhau.

Trong một mô hình dự báo, mối quan hệ giữa biến độc lập và phụ thuộc được dựa trên chủ yếu là phương trình hồi qui tuyến tính. Chẳng hạn có dạng như: 

$$y = ax + b$$

Với $a, b$ là các hằng số.

Phương trình trên có **một** biến đầu vào $x$ nên được gọi là phương trình hồi qui tuyến tính **đơn biến**. Trên mặt phẳng hai chiều, các điểm $(x, y)$ biểu diễn bởi một đường thẳng.

Một trường hợp khác của phương trình tuyến tính cũng khá thường xuyên bắt gặp đó là:

$$y = a x_1 + b x_2 + c$$

Với $a, b, c$ là các hằng số.

Phương trình trên có **nhiều hơn một biến** đầu vào nên được gọi là phương trình hồi qui tuyến tính **đa biến**. Biểu diễn trong không gian ba chiều của nó là một mặt phẳng (_plane_).

Từ hai phương trình trên chúng ta có thể suy ra công thức tổng quát của phương trình hồi qui tuyến tính đó là:

$$y = a_0 + a_1 x _1 + a_2 x_2 + \dots + a_n x_n$$

Theo khái niệm toán học thì các phương trình dạng này được gọi là siêu phẳng (_hyperplane_).

Có một số lý do khiến cho phương trình tuyến tính được lựa chọn để biểu diễn mối quan hệ giữa biến độc lập (các biến $x_i$) và biến phụ thuộc ($y$) trong machine learning. 

* Phương trình tuyến tính có thể khái quát hoá được các phương trình nhân khi thực hiện phép logarith. Chẳng hạn nếu $y = x_1^{\alpha} x_2^{\beta}$ có thể biểu diễn thành 

$$\log{y} = \alpha \log{x_1} + \beta \log{x_2}$$

là một phương trình dạng tuyến tính. 

* Phương trình tuyến tính là định dạng định dạng dễ hiểu, dễ thực hiện. Ví dụ nếu bạn tìm cách biểu diễn $y$ và $x$ theo một phương trình dạng như:

$$y = \frac{sin(\sqrt{x^2+1}).e^x}{cos(x^{3/2})+x^3+2x+1}$$

Thì nó là một quan hệ rất phức tạp và không dễ thực hiện và tính toán.

* Phương trình tuyến tính có thể dễ dàng giải thích mối quan hệ giữa các biến độc lập và phụ thuộc. Thật vậy, trong phương trình $y = a_0 + a_1 x_1 + a_2 x_2 + \dots + a_n x_n$ thì $a_1$ thể hiện tác động biên của $x_1$ lên $y$. Khi $x_1$ tăng/giảm 1 đơn vị thì $y$ tăng/giảm $a_1$ đơn vị.

* Phương trình tuyến tính có thể biểu diễn được mọi mối quan hệ phức tạp của biến. Chúng ta có thể thấy phương trình $y = a x+b$ là một đường thẳng nhưng nếu ta thêm $x^2$ thì phương trình $y = a x^2+bx+c$ đã trở thành một đường cong phi tuyến. Chỉ với phương trình tuyến tính chúng ta có thể biểu diễn được hầu như mọi mối quan hệ dữ liệu phức tạp giữa $x$ và $y$. 

+++ {"id": "OvH2_a_CJR1N"}