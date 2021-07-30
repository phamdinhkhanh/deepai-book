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

+++ {"id": "_BMXC3bA40Ju"}

# 10.1. Ước lượng hợp lý tối đa (_Maximum Likelihood Function - MLE_)

Trong thống kê _ước lượng hợp lý tối đa_ MLE là một phương pháp nhằm ước lượng ra các **tham số** của một **phân phối xác suất** bằng cách tối đa hoá _hàm hợp lý_ sao cho dưới giả định mô hình thống kê thì dữ liệu quan sát được là phù hợp nhất. Tính phù hợp được đo lường thông qua một hàm số được gọi là _hàm hợp lý_ (_Likelihood Function_).

Giả sử chúng ta có một bộ dữ liệu gồm $N$ quan sát đầu vào là $\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N \}$. Có rất nhiều véc tơ tham số khác nhau có thể dùng để mô phỏng phân phối của dữ liệu, chúng ta gọi chung tập hợp đó là _không gian tham số_ (_parameter space_) $\mathcal{W}$. Giả định mỗi một phân phối được đặc trưng bởi một véc tơ tham số $\mathbf{w} = (w_0, w_1, \dots, w_k)^{\intercal}$. Ước lượng hợp lý tối đa sẽ tìm kiếm véc tơ tham số $\mathbf{w}$ trong không gian tham số $\mathcal{W}$ sao cho giá trị _hàm hợp lý_ là lớn nhất. Lớn nhất có nghĩa là phù hợp nhất. _Hàm hợp lý_ ký hiệu là $L(\mathbf{w})$ là một hàm đối với biến số là $\mathbf{w}$, nó có giá trị bằng phân phối xác suất đồng thời của bộ dữ liệu $\mathcal{D}$. 

$$L(\mathbf{w}) = P(\mathcal{D}|\mathbf{w})$$

Trong quá trình _ước lượng hợp lý tối đa_ thì $\mathcal{D}$ cũng giống như kết quả đã biết trước. Tìm ra ước lượng $\hat{\mathbf{w}}$ phù hợp nhất tức là đi tìm ra nguyên nhân giải thích tốt nhất kết quả đã biết. Trong trường hợp các quan sát ngẫu nhiên có phân phối _độc lập và xác định_ (_independent and identically distributed_) viết tắt là `iid`, thì hàm hợp lý sẽ bằng tích của xác suất trên từng quan sát:

$$L(\mathbf{w}) = P(\mathcal{D}|\mathbf{w}) = P(\mathbf{x}_1, \dots, \mathbf{x}_N |\mathbf{w}) = \prod_{i=1}^{N} P(\mathbf{x}_i|\mathbf{w})$$

Mục tiêu của ước lượng hợp lý tối đa là tìm ra véc tơ tham số $\mathbf{w}$ nhằm tối đa hoá hàm hợp lý.

$$\hat{\mathbf{w}} = \arg \max_{\mathbf{w}} L(\mathbf{w})$$

Giải bài toán tối ưu của tích là không dễ dàng. Do đó chúng ta thường sử dụng logarith để chuyển từ tối ưu hàm hợp lý sang tối ưu log của hàm hợp lý (_log likelihood_). Để phân biệt với hàm hợp lý thì hàm _log của hàm hợp lý_ được ký hiệu là một chữ $l$ viết thường.

$$\hat{\mathbf{w}} = \arg \max_{\mathbf{w}} l(\mathbf{w}) = \arg \max_{\mathbf{w}}\log {L(\mathbf{w})}$$

Nếu dữ liệu phân phối `iid` thì bài toán tối ưu sẽ đẹp hơn rất nhiều:

$$\begin{eqnarray}\hat{\mathbf{w}} & = & \arg \max_{\mathbf{w}} ~ \log \prod_{i=1}^{N} P(\mathbf{x}_i|\mathbf {w}) \\
& = & \arg \max ~ \sum_{i=1}^{N} \log P(\mathbf{x}_i|\mathbf {w}) 
\end{eqnarray}$$

Từ phương pháp ước lượng hợp lý tối đa chúng ta có thể chứng minh được ước lượng tham số của rất nhiều các phân phối khác nhau.

+++ {"id": "drvuGCH0Av2U"}

Thật vậy, chắc hẳn trong thống kê các bạn đã từng làm các dạng bài tập về ước lượng trung bình và phương sai của tổng thể dựa vào trung bình mẫu và kích thước mẫu. Ta có thể khái quát bài toán này thành một ví dụ như sau:

**Bài tập**:

Để ước lượng cân nặng trung bình của một người trưởng thành là một điều rất khó. Chúng ta không thể tìm ra con số chính xác về cân nặng trung bình của tất cả mọi người trưởng thành trên thế giới vì cân nặng luôn biến động và thực hiện quá trình này là tốn kém. Vì vậy chúng ta chỉ có thể tìm ra một ước lượng hợp lý nhất từ một mẫu nhỏ và lấy kết quả này đại diện cho tổng thể. Gỉa sử tiến hành đo mẫu gồm $N$ người trưởng thành có cân nặng là $\mathcal{D} = \{x_1, x_2, \dots, x_N \}$. Hãy ước lượng trung bình cân nặng của một người trưởng thành. 


**Lời giải**:

Chúng ta giả định rằng cân nặng tuân theo phân phối chuẩn với trung bình là $\mu$ và phương sai $\sigma^2$. Như vậy theo phân phối chuẩn thì giá trị có xác suất xuất hiện cao nhất sẽ ở vị trí trung bình của phân phối. Tuy nhiên ta không chắc chắn rằng trung bình của $N$ mẫu là ước lượng hợp lý nhất cho cân nặng. Chính vì thế chúng ta sử dụng phương pháp MLE để tìm ra ước lượng hợp lý tối đa. Bạn sẽ thấy ước lượng hợp lý nhất cho cân nặng chính là trung bình.

Thật vậy, vì cân nặng được giả định là tuân theo phân phối chuẩn nên xác suất tại một quan sát $x_i$ sẽ được tính theo hàm mật độ xác suất:

$$P(x_i|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp{-\frac{(x_i-\mu)^2}{2\sigma^2}}$$

Cân nặng của mọi người được giả định là `iid` nên xác suất xảy ra của bộ dữ liệu được tính thông qua tích xác suất trên từng điểm dữ liệu. Khi đó các tham số $\mu, \sigma$ được ước lượng thông qua phương pháp MLE.

$$\begin{eqnarray} \hat{\mu}, \hat{\sigma} & = & \arg \max_{\mu, \sigma} \sum_{i=1}^{N} \log P(x_i|\mu, \sigma) \\
& = & \arg \max_{\mu, \sigma} \sum_{i=1}^{N} \log \left[ \frac{1}{\sqrt{2\pi\sigma^2}} \exp{-\frac{(x_i-\mu)^2}{2\sigma^2}} \right]\\
& = & \arg \max_{\mu, \sigma} [\underbrace{-\frac{N}{2}\log 2\pi}_{C} -N \log \sigma - \sum_{i=1}^{N}\frac{(x_i-\mu)^2}{2\sigma^2}] \\
& = & \arg \max_{\mu, \sigma} [-N \log \sigma - \sum_{i=1}^{N}\frac{(x_i-\mu)^2}{2\sigma^2}] + C
\end{eqnarray}
$$


Đặt: 

$$J(\mu, \sigma) \triangleq \arg \max_{\mu, \sigma} [-N \log \sigma - \sum_{i=1}^{N}\frac{(x_i-\mu)^2}{2\sigma^2}]$$

Điều kiện cần của cực trị theo đạo hàm bậc nhất:

$$\begin{eqnarray}
\frac{\delta J(\mu, \sigma)}{\delta \mu} & = & -\sum_{i=1}^{N} \frac{(x_i-\mu)}{\sigma^2} = 0 \tag{1} \\
\frac{\delta J(\mu, \sigma)}{\delta \sigma} & = & -\frac{N}{\sigma}+\sum_{i=1}^{N} \frac{(x_i-\mu)^2}{\sigma^3} = 0 \tag{2}
\end{eqnarray}$$

Từ đẳng thức $(1)$ ta suy ra:

$$\hat{\mu} = \frac{1}{N}\sum_{i=1}^{N} x_i$$

Đẳng thức $(2)$ cho thấy:

$$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i-\mu)^2$$

Đây chính là những ước lượng hợp lý nhất về trung bình và phương sai của mẫu. Từ những ước lượng này chúng ta có thể suy ra những ước lượng về khoảng tin cậy cho biến.



+++ {"id": "BmkCbcpV5RYO"}

# 10.2. Ước lượng tiên nghiệm tối đa (_Maximum A Posteriori_)

Ở phương pháp _MLE_ chúng ta ước lượng ra phân phối của dữ liệu dựa trên _hàm hợp lý_. _Hàm hợp lý_ $P(\mathbf{x}_i|\mathbf{w})$ chỉ được tính trong điều kiện các tham số phân phối đã xác định. Điều đó có nghĩa rằng chúng ta không thể đưa thêm niềm tin của mình vào xác suất của phân phối dữ liệu. Đây là một hạn chế lớn, đặc biệt là trên những mô hình được hồi qui với kích thước mẫu nhỏ thì qui luật phân phối dựa trên tần suất không còn xác thực. Khi đó kết quả dự báo sẽ chuẩn xác hơn nếu chúng ta đưa thêm niềm tin vào xác suất.

Đó chính là lý do mà _ước lượng tiên nghiệm tối đa_ (_Maximum A Posteriori_), viết tắt là _MAP_ ra đời, cho phép ta đưa thêm niềm tin về phân phối tham số vào mô hình. Về bản chất đây cũng là một phương pháp ước lượng **tham số** của một **phân phối xác suất**, nhưng khác biết với _MLE_ đó là thay vì tối đa hoá hàm _hợp lý_ thì chúng ta tối đa hoá _xác suất hậu nghiệm_. Dựa vào công thức Bayes chúng ta có thể phân tích xác suất thành tích của hàm hợp lý với _xác suất tiên nghiệm_ và điều chỉnh niềm tin vào mô hình thông qua _xác suất tiên nghiệm_. Bài toán tối ưu _MAP_:

$$\begin{eqnarray}\hat{\mathbf{w}} & = & \arg \max_{\mathbf{w}}~ \log P(\mathbf{w}| \mathcal{D}) \\
& = & \arg \max~ \log \frac{P(\mathcal{D}|\mathbf{w}) P(\mathbf{w})}{P(\mathcal{D})} \\
& = & \arg \max~ \underbrace{\log P(\mathcal{D}|\mathbf{w})}_{\text{log likelihood}} + \underbrace{\log P(\mathbf{w})}_{\text{prior}} - \underbrace{\log P(\mathcal{D})}_{\text{evidence}} \\
& = & \arg \max~ \log P(\mathcal{D}|\mathbf{w}) + \log P(\mathbf{w})
\end{eqnarray}$$

Dòng thứ nhất suy ra dòng thứ hai là do công thức Bayes. Dòng thứ 3 suy ra dòng thứ 4 là do xác suất $P(\mathcal{D})$ chỉ phụ thuộc vào dữ liệu mà không phụ thuộc vào $\mathbf{w}$. Do đó trong bài toán tối ưu đối với $\mathbf{w}$ ta có thể loại bỏ thành phần này.

Ta nhận thấy hàm mục tiêu trong phương pháp _MAP_ có thêm _xác suất tiên nghiệm_ (_prior_) so với _MLE_. Thành phần này cũng gần tương tự như thành phần _điều chuẩn_ (_regularization term_) trong các mô hình hồi qui tuyến tính, hồi qui Logistic và SVM mà chúng ta đã học. Tác dụng của thành phần _điều chuẩn_ đó là giảm thiểu hiện tượng _quá khớp_ cho mô hình thông qua sự kiểm soát được áp đặt lên tham số của mô hình hồi qui.

Ưu điểm của phương pháp _MAP_ đó là chúng ta có thể đưa thêm vào niềm tin của mình về mô hình thông qua xác suất $P(\mathbf{w})$ để tối đa hoá hàm mục tiêu. Điều này là rất quan trọng vì thông qua những lượt huấn luyện mô hình trên những bộ dữ liệu khác nhau thì chúng ta có thể suy ra được phân phối $P(\mathbf{w})$ một cách chắc chắn hơn và thông qua đó làm giảm hiện tượng _quá khớp_.

Trong trường hợp chúng ta xem phân phối của $\mathbf{w}$ là đồng nhất thì $
\log P(\mathbf{w})$ là không đổi. Khi đó bài toán tối ưu _MAP_ trở thành tối ưu _MLE_. Như vậy chúng ta có thể coi _MAP_ là một bước phát triển mới, một phương pháp tổng quát hơn của _MLE_ cho phép chúng ta thể hiện niềm tin của mình đối với mô hình.


+++ {"id": "o94EBJLwNry4"}

# 10.3. Mô hình xác suất Naive Bayes

Mô hình xác suất Naive Bayes là mô hình mà xác suất dự báo được ước tính dựa trên công thức Naive Bayes.

Giả định bộ dữ liệu có biến đầu vào bao gồm $d$ biến $\mathbf{x} = \{x_1, x_2, \dots, x_d\}$. Những biến này được giả định là độc lập có điều kiện theo biến mục tiêu $y$. Trong đó biến mục tiêu $y$ có giá trị nằm trong tập hợp nhãn $\mathcal{C} = \{1, 2, \dots, C \}$. Giả định thêm rằng $\mathcal{H}$ là một giả thuyết về phân phối xác suất của biến đầu vào $\mathbf{x}$ tương ứng với từng giá trị của biến mục tiêu $y$. Khi đó phân phối của $\mathbf{x}$ sẽ phụ thuộc vào $y$ và $\mathcal{H}$ nhưng phân phối của $y$ không bị phụ thuộc vào $\mathcal{H}$. Một ước lượng điểm đổi với _xác suất hậu nghiệm_ theo công thức Bayes như sau:

$$\begin{eqnarray}P(y | \mathbf{x}, \mathcal{H}) & = & P(y | x_1, x_2, \dots, x_d, \mathcal{H}) \\
& = & \frac{P(x_1, x_2, \dots, x_d | y, \mathcal{H}) P(y|\mathcal{H})}{P(x_1, x_2, \dots, x_d | \mathcal{H})} \\
& = & \frac{P(x_1, x_2, \dots, x_d | y, \mathcal{H}) P(y)}{P(\mathbf{x} | \mathcal{H})} \\
& = & \frac{\underbrace{\prod_{i=1}^{d} P(x_i|y, \mathcal{H})}_{\text{likelihood}}) \underbrace{P(y)}_{\text{prior}}}{\underbrace{P(\mathbf{x} | \mathcal{H})}_{\text{evidence}}} \\
& \propto & \prod_{i=1}^{d} P(x_i|y, \mathcal{H}) P(y)  \tag{3}
\end{eqnarray}$$

$P(y| \mathbf{x}, \mathcal{H})$ chính là ước lượng xác suất từ giả thuyết $\mathcal{H}$ sau khi đã biết $\mathbf{x}$. Xác suất này là mục tiêu mà chúng ta cần tối ưu. Điều đó cũng có nghĩa rằng nếu ground truth là $y=c$ thì mô hình _Naive Bayes_ cần đưa ra dự báo cho khả năng xảy ra của nhãn $c$ càng lớn càng tốt. Xác suất này sẽ được tính theo khai triển từ công thức Bayes như chúng ta thấy ở $(3)$. Tiếp theo chúng ta cùng đi phân tích phép biến đổi _xác suất hậu nghiệm_.

Từ dòng 1 sang dòng 2 là do công thức Bayes. Mặt khác $\mathcal{H}$ là giả thuyết về phân phối của $\mathbf{x}$ nên giả thuyết này là độc lập với $y$. Điều này dẫn tới $P(y|\mathcal{H}) = P(y)$. Từ đó dòng 2 ta suy ra dòng 3. Tiếp theo các chiều dữ liệu đầu vào là độc lập có điều kiện theo $y$ nên: 

$$P(x_1, x_2, \dots, x_d | y, \mathcal{H}) = \prod_{i=1}^{d} P(x_i|y, \mathcal{H}) \tag{4}$$ 

Giả định $(4)$ ở trên là một sự ngây ngô vì đối với những bộ dữ liệu lớn gồm nhiều chiều thì rất ít khi đạt được điều kiện lý tưởng về sự độc lập. Vì thế mô hình mới có tên gọi là _Naive Bayes_ (tạm dịch là _Bayes ngây ngô_). Tuy nhiên thực nghiệm cho thấy giả định ngây ngô này lại khá hiệu quả trong nhiều lớp mô hình phân loại của học có giám sát mà chúng ta sẽ tìm hiểu về lý thuyết của chúng ở bài viết này.

TIếp theo công thức ở dòng 4 là một công thức quen thuộc phân rã _xác suất hậu nghiệm_ thành ba thành phần chính đó là likelihood, prior và evidence:

* _Likelihood_: Là phân phối xác suất của dữ liệu đầu vào $\mathbf{x}$ trong điều kiện đã biết biến mục tiêu $y$. Xác suất này thể hiện _tính phù hợp_ (_goodness of fit_) của tham số phân phối được giả định trong giả thuyết $\mathcal{H}$. Đồng thời, _Likelihood_ cũng cho biết mức độ đóng góp vào giải thích xác suất của $y$ từ phía dữ liệu đầu vào $\mathbf{x}$. Xác suất này phụ thuộc vào tham số phân phối nên quá trình tối ưu theo Naive Bayes chủ yếu là dựa vào tìm kiếm bộ tham số sao cho _Likelihood_ là tối đa.

* _Prior_: _Xác suất tiên nghiệm_ của biến mục tiêu $y$. Chúng ta có thể đưa vào niềm tin của người làm mô hình vào khả năng xảy ra của $y$, thông qua đó tác động tới _xác suất hậu nghiệm_ được dự báo. Thông thường đối với những bộ dữ liệu có kích thước lớn thì phân phối xác suất của $y$ được ước lượng chính là tỷ lệ giữa các nhãn trong tập huấn luyện.

* _Evidence_: Là phân phối xác suất của dữ liệu không phụ thuộc vào giá trị của $y$. Với mỗi một giả thuyết $\mathcal{H}$ thì $P(\mathbf{x}|\mathcal{H})$ là cố định nên chúng ta suy ra _xác suất hậu nghiệm_ sẽ đồng dạng với tích giữa _likelihood_ và _xác suất tiên nghiệm_. Tức là chúng ta có thể bỏ qua _Evidence_ trong quá trình tối ưu _xác suất hậu nghiệm_.




+++ {"id": "wd7OdFpgLevy"}

Chúng ta có một tính chất khá quan trọng về sự _chuẩn hoá xác suất_ của _xác suất hậu nghiệm_ trong công thức $(3)$. Tức là tổng xác suất của toàn bộ các trường hợp sẽ bằng 1:

$$\sum_{y}P(y | \mathbf{x}, \mathcal{H}) = \sum_{y}\frac{P(\mathbf{x}| y,\mathcal{H}) P(y|\mathcal{H})}{P(\mathbf{x} | \mathcal{H})} = \sum_{y}\frac{P(\mathbf{x}| y,\mathcal{H}) P(y|\mathcal{H})}{\sum_{y}P(\mathbf{x}| y,\mathcal{H}) P(y|\mathcal{H})} = 1$$

+++ {"id": "iLfv6RadLdbG"}

Điều đó cho thấy các ước lượng xác suất từ công thức Bayes của _xác suất hậu nghiệm_ bản thân nó đã được chuẩn hoá để trở thành một phân phối xác suất. $P(y | \mathbf{x}, \mathcal{H})$ là xác suất đối với một khả năng của $y$ nằm trong tập nhãn $\mathcal{C}$. Như vậy nhãn dự báo $\hat{y}$ phải là nhãn mà có khả năng xảy ra là lớn nhất. Điều đó có nghĩa rằng:

$$\begin{eqnarray}\hat{y} & = & \arg \max_{y \in \mathcal{C}} P(y| \mathbf{x}, \mathcal{H}) \\
& = &  \arg \max_{y \in \mathcal{C}} \prod_{i=1}^{d} P(x_i|y, \mathcal{H}) P(y)
\end{eqnarray}$$

Như vậy mô hình _Naive Bayes_ thực chất là ước lượng một _xác suất hậu nghiệm_ nên chúng ta có thể dựa trên _MAP_ để tìm ra các tham số phân phối cho dữ liệu.

Mô hình Naive Bayes dựa trên giả định khá _ngây ngô_ về sự độc lập có điều kiện giữa các chiều dữ liệu nhưng giả định ngây ngô này lại cho thấy hoạt động hiểu quả trong nhiều bài toán, đặc biệt là các bài toán về phân loại tin rác và phân loại văn bản. Chính nhờ giả định _ngây ngô_ mà bài toán tối ưu xác suất đã trở nên dễ dàng hơn nhờ xác suất được ước lượng trên từng chiều độc lập.

Chi phí huấn luyện cho bài toán Naive Bayes cũng ít tốn kém hơn so với các bài toán phân loại khác trong Machine Learning. Vì chúng ta không cần phải giải bài toán tối ưu trên dữ liệu nhiều chiều. Giả định _ngây ngô_ đã phân rã xác suất về những chiều đơn lẻ và độc lập. Dẫn tới thực chất tối ưu _xác suất hậu nghiệm_ là tối ưu phân phối xác suất trên từng chiều độc lập. Tuỳ thuộc vào dữ liệu là liên tục hoặc hạng mục mà ước lượng xác suất trên từng chiều sẽ được tính dựa trên hàm mật độ xác suất hoặc dựa trên tần suất. Nhưng nhìn chung thì những tối ưu này đều khá nhẹ nhàng.

+++ {"id": "Uc53RxrxjSj7"}

## 10.3.1. Gaussian Naive Bayes

Trong mô hình Gaussian Naive Bayes, xác suất của một chiều dữ liệu $x_i$ đối với một nhãn cụ thể $y=c$ được giả định dựa trên phân phối Gaussian và đặc trưng bởi hai tham số phân phối là trung bình $\mu_{ic}$ và phương sai $\sigma_{ic}^2$. Khi đó xác suất được ước lượng từ phân phối Gaussian:

$$P(x_i | y=c) = f(x_i; \mu_{ic}, \sigma_{ic}) = \frac{1}{\sqrt{2\pi\sigma_{ic}^2}} \exp\left(- \frac{(x_i-\mu_{ic})^2}{2\sigma_{ic}^2} \right)$$

Để ước lượng ra hai tham số $\mu_{ic}$ và $\sigma_{ic}$ chúng ta sử dụng phương pháp ước lượng MLE trên toàn bộ dữ liệu. Tức là $\mu_{ic}$ và $\sigma_{ic}$ phải là nghiệm của hàm _hợp lý_:

$$\hat{\mu}_{ic}, \hat{\sigma}_{ic} = \arg \max \prod_{j=1}^{N}P(x_i^{(j)}|y^{(j)}=c)$$

Trong đó $(j)$ chính là chỉ số của quan sát thứ $j$ trong bộ dữ liệu. Từ bài tập trong chương ước lượng hợp lý tối đa ta có thể dễ dàng suy ra giá trị ước lượng của hai tham số $\hat{\mu_{ic}}, \hat{\sigma_{ic}}$ tương ứng với trung bình và phương sai của các quan sát có nhãn là $c$.

![](https://i.imgur.com/Rf9UdPB.jpeg)

**Hình 1**: Các điểm chấm tròn thuộc về nhãn $y=0$ và dấu nhân thuộc về nhãn $y=1$. Trên hình vẽ chúng ta thực hiện các phép chiếu lên các trục $x_1$ và $x_2$ để thu được phân phối trên từng trục. Đường màu xanh thể hiện phân phối đối với nhãn 1 và màu vàng là phân phối của nhãn 0. Phép chiếu từ các điểm có nhãn 1 lên trục $x_1$ ta thu được một phân phối chuẩn $\mathbf{N}(\mu_{11}, \sigma_{11})$ như hình vẽ. Như vậy theo phân phối chuẩn thì những điểm càng gần tâm của nhãn $y=1$ thì xác suất $P(x_1|y=1) = f(x_1; \mu_{11}, \sigma_{11})$ càng lớn. Như vậy về bản chất xác suất trên từng chiều dữ liệu chính là một thước đo mức độ tương đồng đến tâm của nhãn. Xác suất này càng lớn thì thì các điểm dữ liệu sẽ càng gần tâm của nhãn và do đó khả năng cao chúng được phân loại về nhãn là chính xác.

Thông thường mô hình Gaussian Naive Bayes sẽ áp dụng trên dữ liệu đầu vào là những biến liên tục. Để xây dựng mô hình `Gaussian Naive Bayes` thì trong sklearn thì chúng ta sử dụng class `sklearn.naive_bayes.GaussianNB`. Bên dưới chúng ta sẽ thực hành huấn luyện mô hình này trên bộ dữ liệu iris.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 3D-UnC9SOAVu
outputId: 82682f8b-50a7-483c-f3bc-f9aaddad4bb3
---
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print(classification_report(y_pred, y_test))
```

+++ {"id": "cR2yzQq0PxOq"}

Như vậy trên tập kiểm tra mô hình dự báo có độ chính xác trung bình trên cả ba loài hoa đạt 96%. Đây không phải là một độ chính xác quá cao. Trên thực tế thì mô hình `Gaussian Naive Bayes` thường không phải là một lớp mô hình mạnh trong những bài toán phân loại có dữ liệu đầu vào là những biến liên tục. Ưu điểm của `Gaussian Naive Bayes` đó là có chi phí huấn luyện thấp, tốc độ tính toán nhanh và hoạt động trực tiếp trên những bài toán phân loại đa lớp mà không cần phải chuyển sang những bài toán `one-vs-one` hoặc `one-vs-rest`.

+++ {"id": "ypXYp1dfdKQ0"}

## 10.3.2. Multinormial Naive Bayes

Đây là phương pháp thường được sử dụng trong bài toán phân loại văn bản và thực nghiệm cho thấy là một phương pháp khá hiệu quả. Đầu tiên, chúng ta sẽ xây dựng một từ điển bao gồm toàn bộ các từ xuất hiện trong toàn bộ các văn bản. Gỉa sử từ điển này là tập $\mathcal{D}=\{x_1, x_2, \dots, x_d\}$, trong đó $x_i$ là một từ ở vị trí thứ $i$ trong từ điển. Từ điển $\mathcal{D}$ luôn có kích thước cố định là $d$. Thông qua $\mathcal{D}$, một văn bản $\mathbf{x}_j$ bất kì được đặc trưng bởi một véc tơ tần suất $(N_{1j}, N_{2j}, \dots, N_{dj})$ có độ dài bằng độ dài từ điển. Trong đó $N_{ij}$ đại diện cho tần suất của từ $x_i$ trong từ điển xuất hiện trong văn bản $\mathbf{x}_j$. Xác suất để văn bản $\mathbf{x}_j$ rơi vào lớp $y=c$ được tính theo công thức xác suất Bayes:

$$\begin{eqnarray}P(y=c|\mathbf{x}_j) & = & \frac{P(\mathbf{x}_j | y=c) P(y=c)}{P(\mathbf{x}_j)} \\
& \propto & \underbrace{P(y=c)}_{\text{prior}} \underbrace{\prod_{i=1}^{d} P(x_i| y=c)^{N_{ij}}}_{\text{likelihood}}  \tag{5}
\end{eqnarray}$$

_Xác suất tiên nghiệm_ (_prior_) được tính toán khá dễ dàng dựa trên thống kê tỷ lệ quan sát rơi vào từng lớp văn bản.

_likelihood_ thực chất là một phân phối [multinormial](https://phamdinhkhanh.github.io/deepai-book/ch_probability/appendix_probability.html#phan-phoi-multi-normial) về khả năng xuất hiện đồng thời các từ trong văn bản $\mathbf{x}_j$ với tần suất $(N_{1j}, N_{2j}, \dots, N_{dj})$. Để tính được _likelihood_ thì chúng ta phải tính được xác suất xuất hiện của từng từ trong một lớp văn bản có nhãn $y=c$. Ta kí hiệu xác suất này là $\lambda_{ic} = P(x_i| y=c)$. Đồng thời kí hiệu $\mathcal{C}$ là tập hợp indice của các văn bản thuộc lớp $y=c$. Dễ dàng nhận thấy:

$$\lambda_{ic} = \frac{\sum_{j \in \mathcal{C}}N_{ij}}{N_{c}}$$

Trong đó $N_{c}$ là toàn bộ các từ (tính cả lặp lại) xuất hiện trong các văn bản thuộc lớp $y=c$. Dễ nhận thấy: 

$$N_c = \sum_{i=1, j \in \mathcal{C}}^{d} N_{ij}$$

Trong một số tình huống khi một từ không xuất hiện trong văn bản thì sẽ có $\lambda_{ic} = 0$. Khi đó xác suất dự báo ở vế trái của $(5)$ sẽ bằng 0 bất kể các xác suất tương ứng với các từ còn lại xuất hiện trong văn bản có lớn như thế nào. Điều này dẫn tới đánh giá sai lệch về kết quả dự báo. Chính vì thể để khắc phục hiện tượng xác suất bị triệt tiêu về 0 do thiếu từ thì chúng ta sử dụng phương pháp _Laplace smoothing_:

$$\lambda_{ic} = \frac{\sum_{j \in \mathcal{C}}N_{ij} + \alpha}{N_{c} + \alpha d}$$

Hệ số $\alpha$ được lựa chọn là một số dương. Chúng ta nhân $\alpha d$ ở mẫu là để tổng $\sum_{i=1}^d \lambda_{ic} = 1$. Thông thường hệ số $\alpha$ được lựa chọn bằng 1.

Sau khi tính được xác suất của toàn bộ các từ trong bộ từ điển đối với nhãn $c$ ta thu được phân phối $[\lambda_{1c}, \lambda_{2c}, \dots, \lambda_{dc} ]$. Khi đó ta tính được xác suất dự báo:

$$\begin{eqnarray}P(y=c|\mathbf{x}_j) & \propto & P(y=c) \prod_{i=1}^{d} \lambda_{ic}^{N_{ij}}
\end{eqnarray}$$

Cách tính xác suất theo `Multinormial Naive Bayes` là khá đơn giản bởi chúng ta hoàn toàn ước lượng xác suất dựa trên thống kê về tần suất. Trong sklearn chúng ta sử dụng module `sklearn.naive_bayes.MultinomialNB` để xây dựng mô hình `Multinormial Naive Bayes`. Tiếp theo chúng ta sẽ phân loại văn bản thông qua module này.

**Bộ dữ liệu:**

Bộ dữ liệu huấn luyện được trích lọc từ `fetch_20newsgroups`. `fetch_20newsgroups` bao gồm 20 chủ đề khác nhau. Tuy nhiên ở đây ta chỉ lấy ra 1183 văn bản thuộc hai chủ để là _cơ đốc giáo_ có nhãn `soc.religion.christian` và _đồ hoạ máy tính_ có nhãn `comp.graphics`.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: MtrOKIO9wXa5
outputId: 809ade94-3ee6-4973-932e-acab961f7d08
---
from sklearn.datasets import fetch_20newsgroups

# Download bộ dữ liệu phân loại văn bản gồm 2 chủ đề  tôn giáo: 'soc.religion.christian', 'comp.graphics'].
categories = ['soc.religion.christian', 'comp.graphics']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
print('Total document: {}'.format(len(twenty_train.data)))
```

+++ {"id": "bIbq9iYoyQ7l"}

Nội dung của object `twenty_train` sẽ bao gồm dữ liệu là văn bản được chứa trong list `twenty_train.data` và nhãn được chứa trong list `twenty_train.target`.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Uk9nu7Fawyje
outputId: 6302ccf0-fc83-42f0-bf97-ff4780507495
---
for i in range(5):
  print('----------------------------------> \n')
  print('Content: {} ...'.format(twenty_train.data[i][:100]))
  print('Target label: {}'.format(twenty_train.target[i]))
```

+++ {"id": "nu_UEzpVykq6"}

Chúng ta không thể đưa trực tiếp dữ liệu là văn bản vào để huấn luyện mô hình mà cần phải mã hoá chúng dưới dạng véc tơ.

Trong sklearn để xử lý văn bản, mã hoá kí tự sang số (_tokenizing_) và lọc bỏ _từ dừng_ (_stopwords_) chúng ta hoàn toàn có thể sử dụng `sklearn.feature_extraction.text.CountVectorizer`. Class này sẽ giúp xây dựng một từ điển và biến đổi một văn bản thành một véc tơ đặc trưng theo tần suất xuất hiện các từ trong từ điển.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: lqd-gqFNzr4J
outputId: 40e94c7e-a56c-482c-a9cd-56ec173c75b9
---
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
```

+++ {"id": "HSML_Sq90V3N"}

Ma trận `X_train_counts` thu được là một ma trận tần suất có số dòng bằng số lượng văn bản và số cột bằng kích thước của từ điển. Mỗi một dòng là một véc tơ tần suất xuất hiện của các từ trong từ điển. Những tần suất này được sắp xếp theo thứ tự khớp với thứ tự của các từ mà nó thống kê trong từ điển.

**Huấn luyện mô hình Multinormial Naive Bayes**

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: x4XJ7Y8L1EK-
outputId: 20d79aa1-d76d-4367-e933-6d28dd043de1
---
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np

# Khởi tạo mô hình
mnb_clf = MultinomialNB()

# Cross-validation với số K-Fold = 5 và thực hiện 1 lần cross-validation
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
scores = cross_val_score(mnb_clf, X_train_counts, twenty_train.target, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: {:.03f}, Standard Deviation Accuracy: {:.03f}'.format(np.mean(scores), np.std(scores)))
```

+++ {"id": "gdyVCStl2DYm"}

Độ chính xác đạt được 98.9% là rất cao, đồng thời độ biến động của độ chính xác chỉ 0.6% khi thực hiện cross-validation. Điều đó cho thấy mô hình `Multinomial Naive Bayes` khá hiệu quả trong tác vụ phân loại văn bản.

+++ {"id": "jTtlb5W91OdV"}

**Dự báo cho một văn bản mới**

Để dự báo cho một nội dung văn bản mới chúng ta cần phải trải qua hai bước:

* Mã hoá văn bản sang véc tơ tần suất.
* Dự báo trên véc tơ tần suất.

Tất cả những bước này được thực hiện khá dễ dàng trên sklearn.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: jdMEognl2898
outputId: 6c7de2af-6f03-4d6c-94a1-4cccd8afd520
---
# Mã hoá câu văn sang véc tơ tần suất
docs_new = ['God is my love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)

# Dự báo 
mnb_clf.fit(X_train_counts, twenty_train.target)
predicted = mnb_clf.predict(X_new_counts)

for doc, category in zip(docs_new, predicted):
  print('%r => %s' % (doc, twenty_train.target_names[category]))
```

+++ {"id": "ih1ku38NjLBz"}

# 10.4. Tổng kết

Như vậy qua bài viết này chúng ta đã được tìm hiểu thêm về sự khác biệt giữa hai trường phái _tần suất_ và _bayesian_ trong suy diễn thống kê. Điểm khác biệt giữa hai trường phái này đó là _tần suất_ cho rằng xác suất là cố định, bất biến và phụ thuộc vào dữ liệu trong khi _bayesian_ tạo ra sự linh hoạt hơn cho xác suất bằng cách đưa thêm vào niềm tin của người dự báo vào xác suất.

Phương pháp ước lượng tham số phân phối của dữ liệu dựa trên tối đa hoá hàm _hợp lý_ được gọi là _MLE_. _MLE_ trên thực tế là một trường hợp đặc biệt của _MAP_ nếu phân phối của tham số được cho là đồng nhất. 

Trong ước lượng _MAP_ chúng ta tìm cách tối đa hoá _xác suất hậu nghiệm_ thông qua phân rã chúng thành _xác suất tiên nghiệm_ và _likelihood_. Thành phần _xác suất tiên nghiệm_ có ý nghĩa như một _thành phần điều chuẩn_ (_regularization term_) giúp kiểm soát giá trị của tham số ước lượng, thông qua đó giúp giảm thiểu hiện tượng _quá khớp_ cho mô hình.

_Naive Bayes_ là mô hình phân loại mà xác suất dự báo được tính dựa trên công thức Bayes. Trong mô hình _Naive Bayes_ chúng ta dựa trên một giả thuyết _ngây ngô_ đó là các biến đầu vào là độc lập có điều kiện theo biến mục tiêu. Như vậy quá trình tối ưu _xác suất tiên nghiệm_ trở nên đơn giản hơn rất nhiều thông qua tối ưu trên từng chiều đặc trưng. Đối với biến đầu vào liên tục chúng ta ước lượng _likelihood_ theo phân phối Gaussian trong khi các bài toán mà biến đầu vào dạng văn bản hoặc thứ bậc thì phân phối Multinormial được sử dụng. Mô hình _Naive Bayes_ có chi phí tính toán thấp và tỏ ra khá hiệu quả đối với lớp các bài toán liên quan tới phân loại văn bản.

+++ {"id": "QeYpgRxl-epb"}

# 10.5. Bài tập

1. Trường phái _Bayesian_ khác với _tần suât_ (_Frequentist_) trong suy diễn thống kê như thế nào?

2. Ước lượng hợp lý tối đa _MLE_ sẽ tìm ra ước lượng của tham số phân phối dựa trên hàm mục tiêu là gì?

3. Phương pháp _MAP_ có mục tiêu là tối đa hoá hàm mục tiêu là gì? 

4. Ưu điểm của _MAP_ so với _MLE_ là gì?

5. Trong mô hình xác suất _Naive Bayes_ giả định nào được đặt ra và được xem là _ngây ngô_?

6. Làm thế nào để ước lượng ra tham số $\mu, \sigma$ trong phân phối _Gaussian_ cho các biến đầu vào trong mô hình _Gaussian Naive Bayes_.

7. Phân phối _Multinormial Naive Bayes_ thường được sử dụng trên dữ liệu dạng như thế nào?

8. Dựa vào bộ dữ liệu `fetch_20newsgroups`, hãy xây dựng mô hình phân loại chủ đề theo `Naive Bayes` cho 4 nhóm chủ đề là `'alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian'`.

+++ {"id": "wlNFYwa2-h2H"}

# 10.6. Tài liệu

+++ {"id": "Eh8FNtdcQZJk"}

https://scikit-learn.org/stable/modules/naive_bayes.html

https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf

https://deeplearningtheory.com/PDLT.pdf

https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1

https://wiseodd.github.io/techblog/2017/01/01/mle-vs-map/

https://towardsdatascience.com/mle-map-and-bayesian-inference-3407b2d6d4d9
