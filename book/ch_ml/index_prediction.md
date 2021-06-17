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

Như chúng ta đã biết các thuật toán của machine learning có thể được chia thành hai dạng cơ bản là bài toán học có giám sát (_supervised learning_) và bài toán học không giám sát (_unsupervised learning_). Học có giám sát là lớp bài toán được xây dựng dựa trên dữ liệu có nhãn. Các mô hình được xây dựng trên bộ dữ liệu huấn luyện nhằm tìm ra mối quan hệ giữa biến đầu vào (hoặc biến độc lập) với biến đầu ra (hoặc biến mục tiêu, biến phụ thuộc). Trong khi đó lớp bài toán học không giám sát sẽ tự động phân cụm dữ liệu dựa trên đặc trưng tiềm ẩn của dữ liệu đầu vào mà không yêu cầu phải có nhãn cho từng quan sát. Chi phí cho việc chuẩn bị dữ liệu của những bài toán học có giám sát là tốn hơn nhiều so với học không giám sát.

Trong học có giám sát chúng ta lại chia thành lớp bài toán phân loại (_classification_) và dự báo (_prediction_), tuỳ thuộc vào biến đầu ra là rời rạc hay liên tục. Những lớp mô hình phân loại được áp dụng trên biến đầu ra rời rạc giúp trả lời cho câu hỏi quan sát này có nhãn là gì? Trong khi mô hình dự báo được áp dụng trên biến đầu ra liên tục sẽ trả lời cho câu hỏi giá trị của một quan sát được dự báo có độ lớn bằng bao nhiêu? 

Những lớp mô hình machine learning phức tạp thường có độ chuẩn xác cao nhưng chúng lại có mức độ tường minh thấp. Điều đó được thể hiện qua việc chúng ta không dễ dàng giải thích được tác động giữa biến độc lập lên biến phục thuộc. Trái lại mô hình hồi qui tuyến tính trong bài toán dự báo lại là lớp mô hình có phương trình biểu diễn cụ thể, đơn giản và giúp diễn giải và đánh giá tác động dễ dàng. Chính vì thế hồi qui tuyến tính được ưa chuộng và sử dụng trong rất nhiều lĩnh vực.

**Phương trình kiểu mẫu trong bài toán dự báo**

Trong trường hợp đơn giản nhất với một biến đầu vào phương trình tuyến tính có dạng:

$$y = ax + b$$

Với $a, b$ là các hằng số, $a \neq 0$.

Vì có một biến đầu vào nên phương trình trên còn được gọi là phương trình hồi qui tuyến tính **đơn biến**. Trên mặt phẳng hai chiều, quan hệ giữa $x$ và $y$ được thể hiện là một đường thẳng.

Trong trường hợp tổng quát phương trình tuyến tính có dạng:

$$y = a_0 + a_1 x _1 + a_2 x_2 + \dots + a_n x_n$$

Phương trình trên có **nhiều hơn một biến** đầu vào nên được gọi là phương trình hồi qui tuyến tính **đa biến**. Tập hợp những điểm $(x, y)$ tạo thành một siêu phẳng (_hyperplane_).

**Tại sao phương trình hồi qui lại được lựa chọn là tuyến tính?**

Có một số lý do khiến cho phương trình tuyến tính được lựa chọn để biểu diễn mối quan hệ giữa biến độc lập và biến phụ thuộc như sau:

* Phương trình tuyến tính có thể biểu diễn được mối quan hệ luỹ thừa và phép nhân thông qua logarith. Chắc hẳn bạn còn nhớ hàm Cobb–Douglas trong kinh tế vĩ mô biểu diễn mối quan hệ giữa sản lượng theo số lượng lao động $L$ và vốn $K$: $y = f(L, K) = C \times L^{\alpha} \times K^{\beta}$. Để đơn giản hoá, chúng ta sử dụng logarith hai vế để chuyển qua phương trình tuyến tính:

$$\log{y} = \alpha \log{L} + \beta \log{K} + \log{C}$$

* Phương trình tuyến tính là một định dạng định dạng đơn giản và tổng quát. Ví dụ nếu bạn tìm cách biểu diễn $y$ và $x$ theo một phương trình dạng như:

$$y = \frac{sin(\sqrt{x^2+1}).e^x}{cos(x^{3/2})+x^3+2x+1}$$

Thì nó là một quan hệ rất phức tạp và không đại diện.

* Phương trình tuyến tính có thể dễ dàng giải thích mối quan hệ giữa các biến độc lập và phụ thuộc. Thật vậy, trong phương trình $y = a_0 + a_1 x_1 + a_2 x_2 + \dots + a_n x_n$ thì $a_1$ thể hiện tác động biên của $x_1$ lên $y$. Khi $x_1$ tăng/giảm 1 đơn vị thì $y$ tăng/giảm $a_1$ đơn vị.

* Phương trình tuyến tính có thể biểu diễn được mối quan hệ phức tạp giữa biến độc lập và biến phục thuộc phụ thuộc vào đa thức bậc cao. Thật vậy, chúng ta có thể thấy phương trình $y = a x+b$ là một đường thẳng nhưng nếu ta thêm $x^2$ thì phương trình $y = a x^2+bx+c$ đã trở thành một đường cong phi tuyến. Khi tăng bậc của đa thức thì khả năng biểu diễn của phương trình hồi qui càng mạnh.