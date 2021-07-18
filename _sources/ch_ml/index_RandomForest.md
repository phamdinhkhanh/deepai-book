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

Thuật ngữ:

* mô hình rừng cây: Random Forest
* Cây quyết định: Decision Tree
* Học kết hợp: ensemble learning
* Phương pháp bỏ túi: bagging
* Lấy mẫu tái lập: Boostrapping
* Quá khớp: overfitting
* Mẫu nằm ngoài túi: out of bag
* Bầu cử: voting
* Dữ liệu ngoại lai: outlier
* Quá khớp: overfitting
* Bị chệch: bias

# 9. Giới thiệu về mô hình rừng cây (_Random Forest_)

Ở bài trước chúng ta đã tìm hiểu về cây quyết định. Cây quyết định là một mô hình khá nối tiếng hoạt động trên cả hai lớp bài toán phân loại và dự báo của học có giám sát. Ý tưởng chính của mô hình là xây dựng một đồ thị dạng câu hỏi để đưa ra dự báo.

Dù có độ chính xác khá cao nhưng cây quyết định tồn tại những hạn chế lớn đó là:

* Dễ xảy ra _quá khớp_ nếu số lượng các đặc trưng để hỏi lớn. Khi độ sâu của cây quyết định không bị giới hạn thì có thể tạo ra những node lá chỉ có một vài quan sát. Những kết luận dự báo từ chúng thường chỉ đúng trên tập huấn luyện mà không đúng trên tập kiểm tra. 

* Trong tình huống bộ dữ liệu có số lượng biến lớn. Một cây quyết định có độ sâu giới hạn (để giảm thiểu _quá khớp_) thường bỏ sót những biến quan trọng. 

* Cây quyết định chỉ tạo ra một kịch bản dự báo duy nhất cho mỗi một quan sát nên nếu model có hiệu suất kém thì kết quả sẽ bị chệch.

Nếu như sức mạnh của một cây quyết định là yếu thì hợp sức của nhiều cây quyết định sẽ trở nên mạnh mẽ hơn (một cây làm chẳng nên non, ba cây chụm lại nên hòn núi cao). Ý tưởng của sự hợp sức đã hình thành nên mô hình _rừng cây_ (_Random Forest_). 

Nếu bạn là một data scientist thì chắc hẳn bạn đã từng nghe qua và áp dụng mô hình _rừng cây_. Mô hình _rừng cây_  khá nổi tiếng, ở các cuộc thi trên kaggle mô hình này thường được sử dụng và nhiều lần dành chiến thắng. Vì có độ chính xác cao, giảm thiểu hiện tượng _quá khớp_ nên mô hình _rừng cây_ được sử dụng rộng rãi trong cả hai lớp bài toán phân loại và dự báo của học có giám sát.