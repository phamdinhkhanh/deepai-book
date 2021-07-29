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


+++ {"id": "p1JRUV44XAM8"}

Các thuật ngữ được sử dụng trong bài:

* Trường phái tần suất: Frequentist
* Trường phái bayes: Bayesian
* Suy diễn Bayes: Bayes Inference.
* Hợp lý: Likelihood.
* Hàm hợp lý: Likelihood
* Logarith hàm hợp lý: Log likelihood
* Xác suất hậu nghiệm: Posteriori
* Xác suất tiên nghiệm: Prior
* Ước lượng hợp lý tối đa: Maximum Likelihood Estimation (viết tắt MLE)
* Ước lượng tiên nghiệm tối đa: Maximum a Posteriori (viết tắt MAP)
* Không chắc chắn: Uncertainty

# 10. Bạn là _Tần suất_ (_Frequentist_) hay _Bayesian_?

_Tần suất_ và _Bayesian_ thực chất là hai quan niệm hoặc hai góc nhìn khác nhau về xác suất trong thống kê. Xuất phát từ đó hình thành nên hai trường phái lớn, cùng tồn tại song song, đấu tranh và phát triển để tạo nên những học thuyết mới về các mô hình thống kê và học máy hiện đại. Cùng phân tích ví dụ bên dưới để hiểu rõ hơn về hai góc nhìn đặc biệt này:

Giả sử trên tay bạn đang cầm một đồng xu đồng chất và thực hiện một phép gieo đồng xu với hai khả năng $\{S, N\}$ lần lượt tương ứng với mặt sấp và ngửa. Hãy loại bỏ trong đầu các kiến thức liên quan tới kết quả tung đồng xu đồng chất mà trước kia bạn đã từng được biết qua sách vở hoặc thực nghiệm. Ban đầu bạn thực hiện 3 lần tung và thu được kịch bản là $[S, S, N]$.

Khi thực hiện lượt tung thứ 4 bạn đưa ra một khẳng định khá chắc chắn rằng xác suất nhận được mặt sấp sẽ là $\frac{1}{3}$ dựa trên tỷ lệ thu được ở 3 lần tung trước đó. Điều này cho thấy bạn tin vào xác suất là sự thật, cố định và phụ thuộc vào tần suất của dữ liệu. Như vậy bạn là người theo trường phái tần suất (_Frequentist_), một trong những trường phái lâu đời của các mô hình thống kê.

Đối đầu với _tần suất_ là _Bayesian_ một trường phái cho phép chúng ta đo lường tốt hơn về mức độ không chắc chắn của các biến cố xác suất, nơi mà chúng ta có thể đưa vào những kinh nghiệm linh hoạt thay vì những sự thật tần suất khô khan.

Ở lượt tung thứ 4 bạn vẫn chưa tin lắm xác suất thu được mặt sấp là $\frac{1}{3}$ vì lý do số lượt tung quá ít. Bạn vẫn có niềm tin rằng tỷ lệ là cân bằng giữa hai lượt tung. Chẳng hạn căn cứ vào phân tích lý trí rằng đồng xu là đồng chất nên mặt sấp và mặt ngửa có vai trò bình đẳng và có thể hoán vị cho nhau. Do đó tổng xác suất của mặt tung và mặt xấp là 1 sẽ được chia đều về xác suất của từng mặt. Khi đưa ra phỏng đoán về lượt tung thứ 4 bạn không tin xác suất sẽ là $\frac{1}{3}$ là một sự thật. Dưới góc nhìn của Bayesian thì xác suất là một niềm tin hơn là một sự thật bị fix cứng. Nếu bạn đưa thêm niềm tin của mình về sự kiện tung đồng xu bằng phân tích lý trí ở trên. Khi đó bạn chính là người theo trường phái Bayesian.

Thực tế cho thấy đối với các sự kiện chỉ xảy ra ít lần thì niềm tin của bạn sẽ giúp ích nhiều hơn là tần suất. Thật vậy, một ví dụ kinh điển được thể hiện qua bầu cử tổng thống Mỹ năm 2016 giữa hai ứng cử viên là Donald Trumph và Hillary Clinton. Các mô hình từ chuyên gia tại thời điểm đó cho rằng kết quả thắng cử của bà Hillary Clinton lên tới 70% nhưng cuối cùng ông Donald Trumph vẫn trở thành tổng thống. Điều đó cho thấy xác suất từ mô hình chỉ là một niềm tin tương đối và trong các ước tính của nó luôn tồn tại một sự không chắc chắn (_uncertainty_). Sự không chắc chắn càng thể hiện rõ hơn ở những sự kiện chỉ xảy ra ít lần, đặc biệt là các sự kiện chỉ xảy ra một lần. 

Ưu điểm của trường phái _bayesian_ đó là hoạt động hiệu quả hơn trong các tác vụ dự báo với kích thước mẫu nhỏ. Trong khi _tần suất_ chỉ thực sự hữu ích nếu kích thước mẫu lớn. Chẳng hạn như trường hợp bạn tung đồng xu đồng chất không chỉ 3 lần mà lên tới 1000 lần và thu được số lượng mặt sấp và ngửa là gần bằng nhau.

+++ {"id": "xjCAUrhYi__e"}

**Xác suất Bayes**

Ở chương [Xác suất](https://phamdinhkhanh.github.io/deepai-book/ch_probability/appendix_probability.html#xac-suat-co-dieu-kien-va-dinh-ly-bayes) chúng ta đã làm quen với công thức xác suất Bayes. Xin nhắc lại, giả sử một mô hình có dữ liệu đầu vào là $\mathcal{D}$, nhãn là $y$ và $\mathcal{H}$ là giả thuyết quan hệ giữa $\mathcal{D}$ và $y$. Theo công thức bayes thì xác suất được ước lượng theo công thức:

$$P(y|\mathcal{D}, \mathcal{H}) = \frac{P(\mathcal{D}|y, \mathcal{H}) P(y | \mathcal{H})}{P(\mathcal{D}|\mathcal{H})} \tag{1}$$

* Xác suất $P(y|\mathcal{D}, \mathcal{H})$ trong công thức trên còn được gọi là xác suất hậu nghiệm (_posteriori_). Hậu nghiệm có nghĩa là được biết sau. Đây là một xác suất có điều kiện. Nó thể hiện một ước tính về khả năng xảy ra của $y$ khi đã biết $\mathcal{D}$ và giả thuyết $\mathcal{H}$.

* $P(\mathcal{D}|y, \mathcal{H})$ là _Likelihood_ đo lường tính vừa vặn (_goodness of fit_) của dữ liệu đối với các tham số mô hình được giải định trong giả thuyết $\mathcal{H}$. Đây là một đại lượng mà chúng ta thường muốn tối ưu trong các mô hình thống kê nhằm tìm ra một ước lượng tham số mô tả phân phối của một bộ dữ liệu cụ thể một cách phù hợp nhất. _Likelihood_ thể hiện sự hợp lý của dự liệu đầu vào, sự hợp lý này cũng góp phần giải thích xác suất hậu nghiệm. Ngoài ra, trong thống kê thì _hàm hợp lý_ (_Likelihood Function_) cũng là mục tiêu tối ưu của phương pháp _ước lượng hợp lý tối đa_ (_Maximum Likelihood Esitimation_)  được viết tắt là MAE. Đây là một phương pháp được sử dụng khá phổ biến trong ước lượng tham số của các mô hình suy diễn thống kê mà ta sẽ tìm hiểu biên dưới.

* $P(y| \mathcal{H})$ được gọi là xác suất tiên nghiệm (_prior_) thể hiện niềm tin của chúng ta vào xác suất. Trong ví dụ về tung đồng xu đồng chất thì 0.5 là xác suất tiên nghiệm. Vì là niềm tin nên xác suất này phụ thuộc vào quan điểm của người làm mô hình. Do đó giá trị của nó có thể thay đổi khi niềm tin về xác suất thay đổi. Đồng thời xác suất này cũng không bị phụ thuộc vào dữ liệu và thường được thiết lập bằng tỷ lệ _tần suất mẫu_ của nhãn $y$.

Như vậy theo trường phái suy diễn Bayes thì chúng ta kết hợp đồng thời xác suất từ dữ liệu thông qua xác suất của _Likelihood_ và xuất phát từ niềm tin thông qua xác suất _tiên nghiệm_.

Cuối cùng $P(\mathcal{D}|\mathcal{H})$ là chứng cớ (_evidence_) của dữ liệu. Đây là xác suất cố định đối với một bộ dữ liệu cụ thể. Do đó chúng ta có thể xem như giá trị này là không đổi và do đó xác suất hậu nghiệm là đồng dạng với _Likelihood_ và xác suất hậu nghiệm:

$$P(y|\mathcal{D}, \mathcal{H}) \propto P(\mathcal{D}|y, \mathcal{H}) P(y|\mathcal{H})$$

_Lưu ý:_ Ký hiệu $\propto$ có nghĩa là đồng dạng trong phân phối. Tức là phân phối ở vế trái và vế phải là chênh lệch theo một tỷ lệ cố định.

Ngoài ra chúng ta có một tính chất quan trọng về sự chuẩn hoá của phân phối xác suất _hậu nghiệm_:

$$\sum_{y} P(y|D, \mathcal{H}) = \sum_{y} \frac{P(D|y, \mathcal{H}) P(y|\mathcal{H})}{P(D|\mathcal{H})} = \sum_{y}\frac{P(D|y, \mathcal{H}) P(y|\mathcal{H})}{\sum_{y} P(D|y, \mathcal{H}) P(y|\mathcal{H})} = 1$$

Điều đó chứng tỏ xác suất _hậu nghiệm_ bản thân nó đã được chuẩn hoá để trở thành một phân phối xác suất.