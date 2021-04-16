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

+++ {"id": "qXi39Jz7RzPO"}

# 1. Xác suất

Xác suất là một đại lượng linh hoạt dùng để đo lường sự chắc chắn của một sự kiện. Khi nói đến xác suất, chúng ta nghĩ đến khả năng xảy ra. Lấy ví dụ trong tác vụ phân loại ảnh chó và mèo. Nếu bạn rất chắc chắn rằng bức ảnh đó là một con chó thì bạn sẽ đưa ra xác suất là một giá trị nào đó rất gần 1, chẳng hạn 0.95. Nhưng giả sử bức ảnh bị nhoè và chụp ở khoảng cách rất xa thì bạn không chắc chắn đó là một con chó. Do đó xác suất sẽ giúp bạn đưa ra một lựa chọn lưỡng lự hơn, đó là 0.5 (có thể là chó, mà cũng có thể là mèo).

Các sự kiện trong tự nhiên thì hoàn toàn không bao giờ chắc chắn. Chắc bạn còn nhớ trong bầu cử tổng thống Mỹ năm 2016 giữa ông Donald Trumph và bà Hillary Clinton. Trước khi bầu cử rất nhiều nhận định cho rằng bà Clinton sẽ thắng cử nhưng cuối cùng ông Trumph vẫn trở thành tổng thống. Chính vì thế các nhà khoa học sẽ gán cho các sự kiện không chắc chắn một xác suất để cho thấy sự tin cậy của quyết định.

Việc chúng ta dự báo xác suất có rất nhiều ý nghĩa trong thực tiễn. Các công ty thương mại điện tử muốn dự đoán khả năng khách hàng mua sản phẩm nào là cao nhất dựa trên xác suất. Từ đó họ tối ưu lại thiết kế hệ thống recommendation của mình để gợi ý cho khách hàng sao cho họ mua hàng nhiều nhất. Trong ngành bài bạc, các nhà cái muốn tìm cách chiến thắng người chơi dựa trên tính toán về khả năng chiến thắng/thất bại là bao nhiêu để tìm ra kỳ vọng về lợi nhuận. Các công ty bảo hiểm muốn tính toán khả năng xảy ra rủi ro đối với khách hàng của mình để đưa ra mức phí bảo hiểm phù hợp. Trong nông nghiệp chúng ta quan tâm nhiều hơn tới khả năng mưa, nắng, độ ẩm, gió, các cơn bão để tìm cách điều tiết mùa màng,.... Mục tiêu của các mô hình phân loại trong học máy đều là tìm ra một mô hình ước lượng xác suất tốt nhất để mang lại lợi ích cho tác vụ huấn luyện.

Chính vì vai trò quan trọng như vậy nên có những ngành học dành toàn bộ cho xác suất như xác suất thống kê, định giá tài sản tài chính, định giá bảo hiểm,.... Không thể phủ nhận rằng đây là một mảng rất rộng và tất nhiên chương này tôi cũng không tìm cách bao quát toàn bộ kiến thức về xác suất mà chỉ giới thiệu đến các bạn những khái niệm nền tảng được ứng dụng nhiều trong học máy. Từ đó bạn đọc sẽ có thêm kiến thức để tự nghiên cứu và ứng dụng các mô hình trong thực tiễn.


+++ {"id": "QIZAiZvqYKwR"}

## 1.1. Không gian mẫu

Các xác suất chính là một độ đo được xác định trên một không gian mẫu. Không gian mẫu được ký hiệu là $S$ cho biết tất cả các khả năng có thể xảy ra của một sự kiện. Ví dụ khi chúng ta gieo một xúc sắc 6 mặt thì các mặt ${1, 2, 3, 4, 5, 6}$ chính là một không gian mẫu. Khi chúng ta tung đồng xu 2 mặt đồng chất thì các mặt ${S, N}$ chính là một không gian mẫu.

Xác suất của một sự kiện $i$ bất kỳ nằm trong không gian mẫu được ký hiệu bằng $P(X=i)$ hoặc chúng ta có thể viết tắt $P(i)$.

Chúng ta cũng có thể sử dụng ký hiệu $P(1 \leq X \leq 4)$ để chỉ ra xác suất rơi vào các khả năng ${1, 2, 3, 4}$. Ký hiệu $X$ ở trên được gọi là biến ngẫu nhiên.





+++ {"id": "p7ZyghaJZFrw"}

## 1.2. Biến ngẫu nhiên

Biến ngẫu nhiên là giá trị ngẫu nhiên đại diện cho một đại lượng trong thực tế. Biến ngẫu nhiên có thể liên tục hoặc rời rạc tuỳ theo đại lượng mà nó biểu diễn. Trong trường hợp tung xúc sắc 6 mặt thì biến ngẫu nhiên chính là một trong các khả năng ${1, 2, 3, 4, 5, 6}$. Đây là biến rời rạc vì tập hợp của chúng có số lượng quan sát cố định. Nếu chúng ta đo lường cân nặng của một người thì giá trị đó là một biến ngẫu nhiên liên tục. Lý do nó liên tục là vì cân nặng có thể là một số hữu tỷ bất kỳ, ví dụ như 55.0293102311 mà không nhất thiết phải là một số nguyên. Và chắc chắc rằng cân nặng giữa 2 người bất kỳ trên trái đất là khác nhau. Khi chúng ta nói hai người có cân nặng bằng nhau là ta đang giả định rằng cân nặng của họ cùng nằm trên một khoảng rất nhỏ ví dụ như từ $52-53$.

Biến ngẫu nhiên liên tục và rời rạc có sự khác biệt nhau về giá trị có thể nhận được nên trong một số công thức chúng ta tách rời thành những trường hợp cho biến ngẫu nhiên và biến rời rạc riêng. Cụ thể như các đặc trưng của biến cho hai trường hợp biến ngẫu nhiên liên tục và rời rạc bên dưới.

## 1.3. Đặc trưng của biến

### 1.3.1 Kì vọng

Trong một mẫu có rất nhiều các quan sát thì chúng ta không biết để chọn ra mẫu nào làm đại diện. Giá trị kỳ vọng của một biến ngẫu nhiên sẽ được sử dụng là giá trị đại diện cho toàn bộ mẫu. Giá trị kỳ vọng này được tính theo hai trường hợp:

* Nếu $\text{x}$ là biến ngẫu nhiên rời rạc.

$$\text{E(x)} = \sum_{i=1}^{n} x_i p(x_i)$$

Trong đó $p(x_i)$ là xác suất xảy ra biến cố $x = x_i$. Khi khả năng xảy ra của các biến cố ngẫu nhiên $x_i$ là như nhau thì giá trị của kỳ vọng: 

$$\text{E(x)} = \bar{\text{x}} = \frac{\sum_{i=1}^{n}x_i}{n}$$

* Nếu $\text{x}$ là một đại lượng ngẫu nhiên liên tục:

$$\text{E(x) }= \bar{\text{x}} = \int xp(x) dx$$

Một số tính chất của kì vọng:

$$\begin{eqnarray}\text{E(ax)} & = & a\text{E(x)} \\
\text{E(ax+by)} & = & a\text{E(x)} + b\text{E(y)} \\
\text{E(xy)} & = & \text{E(x)}\text{E(y)}, ~ \text{if} ~ \text{x, y} ~ \text{independent}
\end{eqnarray}$$


### 1.3.2 Hiệp phương sai 

Là đại lượng đo lường mối quan hệ cùng chiều hoặc ngược chiều giữa 2 biến ngẫu nhiên. Đây là đại lượng được sử dụng nhiều trong kinh tế lượng và thống kê học để giải thích mối quan hệ tác động giữa các biến. Khi hiệp phương sai giữa 2 biến lớn hơn 0, chúng có quan hệ đồng biến và ngược lại. Hiệp phương sai chỉ được tính trên 2 chuỗi có cùng độ dài.

$$\text{cov(x, y)} = \text{E}[(\text{x}-\bar{\text{x}})(\text{y}-\bar{\text{y}})] = \frac{\sum_{i=1}^{n} (x_i-\bar{\text{x}})(y_i-\bar{\text{y}})}{n}$$

Gía trị của hiệp phương sai giữa 2 chuỗi số $\text{x,y}$ được kí hiệu là $\text{cov(x,y)}$ hoặc $\sigma_{\text{xy}}$ và được tính bằng kì vọng của tích chéo độ lệch so với trung bình của 2 biến như công thức trên.

Như vậy ta có thể rút ra các tính chất của hiệp phương sai:

* tính chất giao hoán: 
$$\text{cov(x, y) = cov(y, x)}$$
* tính chất tuyến tính: 
$$\text{cov(ax, by) = ab.cov(x, y)}$$

Khai triển công thức hiệp phương sai ta có: 

$$\begin{eqnarray}\text{cov(x, y)} & = & \text{E(xy)}-\mu_\text{x}\text{E(y)}-\mu_\text{y}\text{E(x)} + \mu_\text{x}\mu_\text{y}\end{eqnarray}$$

Trong đó $\mu_\text{x}, \mu_\text{y}$ lần lượt là kì vọng của $\text{x, y}$.

### 1.3.3. Phương sai 

Là trường hợp đặc biệt của hiệp phương sai. Phương sai chính là hiệp phương sai giữa một biến ngẫu nhiên với chính nó. Giá trị của phương sai luôn lớn hơn hoặc bằng 0 do bằng tổng bình phương sai số của từng mẫu so với kỳ vọng. Trong trường hợp phương sai bằng 0, đại lượng là một hằng số không biến thiên. Phương sai của một đại lượng thể hiện mức độ biến động của đại lượng đó xung quanh giá trị kỳ vọng. Nếu phương sai càng lớn, miền biến thiên của đại lượng càng cao và ngược lại.

Phương sai được kí hiệu là $\text{Var}(x)$, $\sigma_x^2$ hoặc $s_x^2$. Công thức phương sai được tính như sau:

  * Nếu $x$ là đại lượng ngẫu nhiên rời rạc:

$$\text{Var}(x) = \sum_{i=1}^{n} (x_i-\mu)^2 p(x_i) dx$$

Trong đó $\text{E}(x) = \mu$. Khi các biến cố xảy ra với cùng xác suất bằng $\frac{1}{n}$, phương sai chính là trung bình $\text{Var}(x) = \frac{\sum_{i=1}^{n} (x_i-\mu)^2}{n}$

  * Nếu $x$ là đại lượng ngẫu nhiên liên tục:

$$\text{Var}(x) = \int (x-\mu)^2 p(x) dx$$

Phương sai của một biến có thể được tính toán thông qua kì vọng của biến:

$$\begin{eqnarray}
\text{Var}(x) & = & \text{E}((x-\mu)^2) \\
& = & \text{E}((x^2-2\mu x+\mu^2)) \\ 
& = & \text{E}(x^2)-2\mu \text{E}(x)+\text{E}(\mu^2) \\ 
& = & \text{E}(x^2)-2\mu^2+\mu^2 \\ 
& = & \text{E}(x^2)-\mu^2 \\
& = & \text{E}(x^2)-\text{E}(x)^2 \end{eqnarray}$$

Đây là một trong những tính chất rất thường được sử dụng trong tính toán nhanh phương sai mà bạn đọc cần nhớ. Đồng thời từ công thức trên ta cũng suy ra một bất đẳng thức quan trọng đó là kỳ vọng của bình phương luôn lớn hơn bình phương của kỳ vọng: $\text{E}(x^2) \geq \text{E}(x)^2$

+++ {"id": "02SB_DVb38Ji"}


### 1.3.4. Độ lệch chuẩn 

Độ lệch chuẩn của một đại lượng có giá trị bằng căn bậc 2 của phương sai. Nó đại diện cho sai số của đại lượng so với trung bình.

$$\sigma_x = \sqrt{\text{Var}(x)}$$

Trong trường hợp các biến rời rạc phân phối đều với xác suất $\frac{1}{n}$:

$$\sigma_x = \sqrt{\frac{\sum_{i=1}^{n}(x-\bar{x})^2}{n}}$$

Trong thống kê chúng ta thường xác định các giá trị outliers dựa trên nguyên lý 3 sigma bằng cách xem những giá trị nằm ngoài khoảng $[\mu-3\sigma, \mu+3\sigma]$ như là outliers. Ta có thể xử lý outliers bằng cách đưa về đầu mút gần nhất $\mu-3\sigma$ hoặc $\mu+3\sigma$ hoặc loại bỏ luôn outliers.

6.Hệ số tương quan: Là một chỉ số có quan hệ gần gũi với hiệp phương sai. Hệ số tương quan đánh giá mối quan hệ đồng biến hay nghịch biến giữa 2 đại lượng ngẫu nhiên. Tuy nhiên khác với hiệp phương sai, hệ số tương quan cho biết thêm mối quan hệ tương quan tuyến tính giữa 2 biến là mạnh hay yếu.

Hệ số tương quan giao động trong khoảng [-1, 1]. Tại 2 giá trị đầu mút -1 và 1, hai biến hoàn toàn tương quan tuyến tính. Tức ta có thể biểu diễn $\text{y}=a\text{x}+b$. Trường hợp hệ số tương quan bằng 0, hai đại lượng là độc lập tuyến tính. Phương trình biểu diễn tương quan được tính như sau:

$$\rho_{\text{xy}} = \frac{\text{cov(x,y)}}{\sigma_{x}\sigma_{y}}$$

Trong hồi qui tuyến tính và logistic, hệ số tương quan thường được dùng để ranking mức độ quan trọng của biến trước khi thực hiện hồi qui. Trong các mô hình timeseries như ARIMA, GARCH chúng ta cũng xác định các tham số bậc tự do của phương trình hồi qui dựa trên hệ số tương quan giữa các chuỗi với độ trễ của nó.

+++ {"id": "67HAjwMZ4c-B"}

## 1.4. Qui luật số lớn

Qui luật số lớn cho rằng khi một mẫu con có kích thước càng lớn được rút ra từ tổng thể thì các đại lượng đặc trưng của nó như trung bình, phương sai càng tiệm cận tới giá trị của của tổng thể. Phát biểu toán học của qui luật số lớn:

Xét $n$ mẫu ngẫu nhiên $X_1, X_2,..., X_n$ độc lập cùng tuân theo phân phối $\mathbf{N}(\mu, \sigma^2)$. Khi đó với mọi số thực dương $\epsilon$ ta có:

$$\lim_{n \rightarrow \infty} P(|\frac {X_{1}+X_{2}+...+X_{n}}{n} - \text{E(X)}| \geq \epsilon) = \lim_{n \rightarrow \infty} P(|\bar{\text{X}} - \text{E(X)}| \geq \epsilon) = 0$$

Công thức trên có ý nghĩa rằng xác suất để $\bar{\text{X}}$ hội tụ về $\text{E(X)}$ là rất lớn. Do đó mặc dù $\epsilon$ rất nhỏ thì khả năng để khoảng cách $|\bar{\text{X}} - \text{E(X)}| \geq \epsilon$ cũng rất nhỏ và gần bằng 0.

+++ {"id": "Azd9E8zH6lIr"}

## 1.5. Chứng minh qui luật số lớn

Đây là một phần dành cho những bạn nào yêu thích toán. Những bạn đọc chỉ quan tâm tới ứng dụng có thể bỏ qua.

Để chứng minh luật số lớn ta cần sử dụng đến bất đẳng thức Markov đó là: xác suất để một biến ngẫu nhiên $\text{X}$ không âm lớn hơn $a$ ($a > 0$) luôn nhỏ hơn kì vọng của biến ngẫu nhiên đó chia cho $a$. $$P(\text{X}\geq a) \leq \frac{\text{E(X)}}{a}$$

* **Chứng minh bất đẳng thức markov:**

Do $x$ không âm nên

$$\begin{eqnarray}\text{E(X)} &=& \int_{0}^{\infty} xf(x)dx \\ &=& \int_{0}^{a} xf(x)dx + \int_{a}^{\infty} xf(x)dx \\ 
&\geq& \int_{a}^{\infty} xf(x)dx \\
&\geq& \int_{a}^{\infty} af(x)dx \\
&=& a\int_{a}^{\infty}f(x)dx \\ 
&=& a.P(\text{X} \geq a)\end{eqnarray}$$

Từ đó suy ra $$P(\text{X}\geq a) \leq \frac{\text{E(X)}}{a}$$

* **Chứng minh qui luật số lớn:**

$$P(|\frac {X_{1}+X_{2}+...+X_{n}}{n} - \text{E(X)}| \geq \epsilon) = P((\frac {X_{1}+X_{2}+...+X_{n}}{n} - \text{E(X)})^2 \geq \epsilon^2)$$

Đặt $Y_n = \frac{X_1+X_2+...+X_n}{n}$ và $\text{Z} = (Y_{n}-\text{E(X)})^2$. Áp dụng bất đẳng thức markov cho đại lượng không âm $\text{Z}$, ta có: $$P(\text{Z} \geq \epsilon^2) \leq \frac{\text{E(Z)}}{\epsilon^2} \tag{1}$$

Ở đây ta coi $X_1, X_2, \dots, X_n$ là các biến độc lập. Khi đó:

$$\text{Var}(Y_{n}) = \text{Var}(\frac{X_1 + X_2 + \dots + X_n}{n}) = \frac{n\text{Var(X)}}{n^2} = \frac{\text{Var(X)}}{n}$$

Do đó:

$$\begin{eqnarray}\lim_{n \rightarrow \infty}\text{E(Z)} & = & \lim_{n \rightarrow \infty}\text{E}(Y_{n}-\text{E(X)})^2 
\end{eqnarray}$$

Mặt khác:

$$\lim_{n \rightarrow \infty} \text{E}(Y_n) = \text{E(X)}$$

Do đó:

$$\begin{eqnarray}\lim_{n \rightarrow \infty}\text{E(Z)} & = & \lim_{n \rightarrow \infty}\text{E}(Y_{n}-\text{E(X)})^2 \\ 
&=& \lim_{n \rightarrow \infty}\text{E}(Y_{n}-\text{E}(Y_n))^2 \\ 
&=& \lim_{n \rightarrow \infty}\text{Var}(Y_{n}) \\ 
&=& \lim_{n \rightarrow \infty} \frac{\text{Var(X)}}{n} = 0 \end{eqnarray}$$

Từ đó thế vào (1) ta suy ra:

$$\begin{eqnarray}\lim_{n \rightarrow \infty} P(\text{Z} \geq \epsilon^2) &\leq& \lim_{n \rightarrow \infty}\frac{\text{E(Z)}}{\epsilon^2} \ &=& \lim_{n \rightarrow \infty} \frac{\text{Var(X)}}{n\epsilon^2} = 0\end{eqnarray}$$

Mặt khác $P(\text{Z} \geq \epsilon^2) \geq 0$ nên suy ra $\lim_{n \rightarrow \infty}P(\text{Z} \geq \epsilon^2) = 0$. Suy ra điều phải chứng minh. Mấu chốt của chứng minh bất đẳng thức này là chúng ta phải phát hiện được tính chất $\text{Var}(Y_{n}) = \frac{\text{Var(X)}}{n}$ là một đại lượng tiến dần về 0 khi $n$ tiến tới vô cùng.

+++ {"id": "XW7pzbq5-ACI"}

## 1.6. Hàm mật độ và hàm khối xác suất

Một số biến ngẫu nhiên có tính chất phân phối đều như tung đồng xu hai mặt đồng chất, tung xúc sắc 6 mặt đồng chất. Nhưng hầu hết các biến ngẫu nhiên không bao giờ có tính chất phân phối đều. Hẳn bạn còn nhớ qui luật pareto nổi tiếng về 20% nguyên nhân là sự giải thích của 80% các sự kiện? Rất nhiều các sự kiện trong cuộc sống tuân theo qui luật tưởng chừng như vô lý nhưng lại rất hợp lý này. Chẳng hạn như 20% người giaù sẽ nắm giữ 80% tổng tài sản của thế giới. quyết định đến từ 20% nhân sự cấp cao của công ty quyết định 80% năng suất lao động toàn công ty. Trong cuộc sống, việc tìm ra phân phối của biến sẽ có rất nhiều ý nghĩa vì nó giúp ta biết được đâu là **bản chất của vấn đề** trong mọi khía cạnh cuộc sống. 

Để biết được một biến ngẫu nhiên sẽ có phân phối trông như thế nào? Ở miền nào thì tập trung nhiều giá trị của biến, miền nào thì ít hơn. Câu trả lời đó được cung cấp thông qua hàm phân phối xác suất.

* **Hàm khối xác suất** (_probability mass function_): được viết tắt là _pmf_ và ký hiệu là $p(x)$ của một biến ngẫu nhiên $\mathbf{x}$ rời rạc là một hàm số đo lường xác suất xảy ra sự kiện $p(\mathbf{x} = x)$ của một biến cố. Như vậy $1 \geq p(x) \geq 0$ và tổng xác suất của toàn bộ các khả năng trong không gian biến cố bằng 1, hay:

$$\sum_{x \in \mathcal{S}} p(x) = 1$$

Trong đó $\mathcal{S}$ là không gian biến cố, chẳng hạn trường hợp tung đồng xu thì $\mathcal{S} = {1,2,3,4,5,6}$.

* **Hàm mật độ xác suất** (_probability density function_): Khi biến ngẫu nhiên liên tục sẽ có vô số các giá trị có thể của $x$. Vì vậy ta không thể biểu diễn khả năng xảy ra của toàn bộ sự kiện dưới dạng tổng xác suất rời rạc. Khi đó tích phân sẽ được sử dụng thay thế.

$$\int p(x) dx = 1$$

Trong trường hợp này thuật ngữ hàm mật độ xác suất (_probability density function_) và ký hiệu là _pdf_ sẽ được sử dụng để thể hiện $p(x)$.

Như chúng ta đã biết tích phân của một hàm số $f(x)$ chính là diện tích nằm giữa đường cong đồ thị $y = f(x)$ và trục hoành. Như vậy, phần diện tích nằm dưới hàm mật độ xác suất $p(x)$ và trên trục hoành luôn có giá trị là 1. Chẳng hạn như đồ thị hàm mật độ xác suất của phân phối chuẩn như hình bên dưới:


![](https://ds055uzetaobb.cloudfront.net/image_optimizer/1dbcc5a80e3fb541aa4678fcff58bb26ca717902.png)

**Hình 1**: đồ thị hàm mật độ xác suất của phân phối chuẩn.

> Hàm mật độ xác suất của phân phối chuẩn có phương trình $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^\frac{-(x-\mu)^2}{2\sigma^2}$ là đường cong có hình quả chuông đối xứng 2 bên. Giá trị hàm mật độ xác suất tại những điểm lùi về phía 2 đuôi trái và phải nhỏ dẫn và giá trị hàm mật độ xác suất tại vị trí trung tâm $x=\mu$ là lớn nhất. Phần diện tích màu hồng nằm dưới đường cong hàm mật độ xác suất và trục hoành có giá trị bằng 1.


+++ {"id": "NN85cFGJaE7c"}

## 1.8. Hàm phân phối xác suất tích luỹ

Hàm phân phối xác suất tích luỹ (_cumulative distribution function_) được viết tắt là _cdf_ là một hàm số cho biết xác suất xảy ra của biến ngẫu nhiên nhỏ hơn một giá trị $x$ xác định. Hàm số này được định nghĩa như sau:

$$F_X(x) = P(X \leq x)$$

Do đó $F_X(x) \in [0, 1]$ và là một hàm đơn điệu tăng. 

Vậy hàm _cdf_ được biểu thị trên đồ thị như thế nào? Hẳn chúng ta còn nhớ khái niệm về tích phân đã từng học tại THPT, đây chính là phần diện tích nằm dưới đồ thị của hàm số và nằm trên trục hoành. 

![](http://work.thaslwanter.at/Stats/html/_images/PDF_CDF.png)

Chẳng hạn trong phân phối ở hình trên ta có giá trị của phân phối xác suất tích luỹ $F_X(x=1) = CDF(x)$ và chính là diện tích phần diện tích được tô xám bên trái. Khi đồ thị hóa hàm phân phối xác suất tích lũy ta thu được đồ thị của $F_X(x)$ như hình bên phải. Trục $x$ là giá trị của biến ngẫu nhiên $X$ và trục $y$ là giá trị của hàm phân phối xác suất tích luỹ $F_X(x)$. Đây là một hàm đơn điệu tăng theo $x$.

+++ {"id": "z2gJTKi9ZIUq"}

## 1.7. Xác suất đồng thời (_join distribution_)

Trường hợp trên là đối với không gian xác suất chỉ gồm một biến cố. Trên thực tế có nhiều biến cố xảy ra có mối liên hệ với nhau và đòi hỏi phát xét đến những không gian xác suất đồng thời của nhiều biến cố. Chúng ta sẽ thể hiện các xác suất đồng thời thông qua hàm phân phối xác suất đồng thời $p(x, y)$ biểu thị khả năng xảy ra đồng thời của cả 2 sự kiện $x$ và $y$.

**Nếu x, y rời rạc:**

$$\sum_{x, y} p(x, y) = 1$$

**Nếu x, y liên tục:**

$$\int {p(x, y)} dx dy = 1$$

**Nếu x rời rạc, y liên tục:**

$$\sum_{x}\int p(x, y) dy = 1$$


+++ {"id": "3py9VerNEApv"}

## 1.8. Xác suất biên (margin distribution)

Nếu chúng ta cố định một biến cố và tính tổng (đối với biến rời rạc) hoặc tích phân (đối với biến liên tục) các xác suất chung $p(x, y)$ theo biến cố còn lại thì ta sẽ thu được hàm phân phối xác suất của theo một biến. Hàm phân phối xác suất này được gọi là xác suất biên (marginal probability). Chúng ta có thể coi xác suất biên là một dạng giảm chiều dữ liệu của xác suất. Chẳng hạn trong bản bên phân phối giữa điểm toán và điểm văn bên dưới thì muốn tính phân phối xác suất biên của điểm toán chúng ta sẽ tính tổng xác suất của mỗi điểm toán từ $0$ tới $10$ theo toàn bộ các điểm văn tương ứng:

![](https://imgur.com/rYYb5OJ.png)

Như vậy từ phân phối xác suất hai chiều ta đã thu được phân phối xác suất biên một chiều.

Công thức phân phối xác suất biên được tính như sau:

**Biến rời rạc:**

$$p(x) = \sum_{y} p(x, y)$$

$$p(y) = \sum_{x} p(x, y)$$

**Biến liên tục:**

$$p(x) = \int_{y} p(x, y) dy$$

$$p(y) = \int_{x} p(x, y) dx$$


+++ {"id": "vgUBFsQ0ZNsH"}

## 1.9. Xác suất có điều kiện và định lý bayes.

Xác suất của $y$ theo điều kiện của $x$ kí hiệu là $p(y|x)$ còn được gọi là xác suất hậu nghiệm (_posterior probability_). Xác suất này được tính dựa trên định lý bayes như sau:

$$p(y|x) = \frac{p(x, y)}{p(x)}$$

Xác suất hậu nghiệm cho ta biết khả năng xảy ra của một biến cố $y$ trong điều kiện đã xét đến khả năng xảy ra của các biến cố $x$.

Hậu nghiệm ở đây có nghĩa là chúng ta chỉ biết xác suất xảy ra của $y$ sau khi đã biết $x$. Trái nghịch với xác suất hậu nghiệm là xác suất tiên nghiệm _prior probability_ $p(y)$. Xác suất này là xác suất được đưa ra mà không cần có data. Vậy tại sao không có data mà vẫn đưa ra được xác suất. Đó là bởi nó dựa trên kinh nghiệm có từ trước. Chẳng hạn như khi tung đồng xu đồng chất thì không cần biết đồng xu đó như thế nào thì chúng ta vẫn biết xác suất xảy ra của nó là $1/2$ vì kinh nghiệm cho chúng ta biết được qui luật này.

Từ định lý bayes ta có thể suy ra:

$$p(x, y) = p(x|y)p(y) = p(y|x)p(x)$$

Ví dụ xác suất có điều kiện: Một người tham gia một trò chơi tung xúc sắc 6 mặt đồng chất. Người đó chơi 100 lượt thì có 18 lượt thu được mặt 6. Trong các ván có mặt sáu thì có 15 ván chiến thắng. Hỏi xác suất người đó chiến thắng là bao nhiêu nếu tung được mặt 6?

Chúng ta ký hiệu biến cố chiến thắng là $y$ và gieo được mặt $6$ là $x$. Từ các giữ kiện ta suy ra $p(x=6)=\frac{18}{100} = 0.18$, $p(y, x=6)=\frac{15}{100} = 0.15$. Khi đó theo định lý bayes thì xác suất người đó chiến thắng trong điều kiện tung được xúc sắc mặt 6 là:

$$p(y|x = 6) = \frac{p(y, x=6)}{p(x = 6)} = \frac{0.15}{0.18} = 0.833$$

Thậm chí không cần biết số lần tung được mặt $6$ là bao nhiêu. Dựa trên xác suất tiên nghiệm thì $p(x = 6)$ thông thường sẽ bằng $\frac{1}{6}$ nếu khối xúc sắc là đồng chất. Khi đó ta có thể tính ra được xác suất chiến thắng tương đối chính xác là: 


$$p(y|x = 6) = \frac{p(y, x=6)}{p(x = 6)} = \frac{0.15}{1/6} = 0.9$$


Xác suất $p(y |x=6)$ là xác suất hậu nghiệm cho biết khả năng chiến thắng trong điều kiện đã biết tung được mặt $x=6$.

Trong các mô hình classification, xác suất dự báo đối với input là quan sát $X$ sẽ là xác suất hậu nghiệm $P(Y=1 |X)$ trong điều kiện mẫu có các đặc trưng mẫu là $X$.

+++ {"id": "MmLUVuIAZc5W"}

## 1.10. Định lý bayes mở rộng

Chúng ta có thể triển khai $p(x) = \sum_{y} p(x, y)$
Khi đó công thức bayes mở rộng sẽ có dạng:

$$p(y|x) = \frac{p(x, y)}{\sum_{y} p(x, y)}$$

Công thức bayes mở rộng sẽ hữu ích trong trường hợp chúng ta chưa có ngay xác suất tiên nghiệm $p(x)$ mà xác suất này chỉ được tính thông qua các trường hợp xảy ra đồng thời của cả $x$ và $y$.

Chẳng hạn quay trở lại ví dụ về tung xúc sắc. Nếu ta biết các dữ kiện đó là: Số lần _chiến thắng_ khi tung được xúc sắc 6 mặt là 15 và số lần _hoà_ khi tung được xúc sắc 6 mặt là 3. Điểm 6 là cao nhất nên sẽ không có lần nào _thua_. Khi đó xác suất chiến thắng khi tung vào mặt 6 đó là:

$$p(y|x=6) = \frac{p(x=6, y)}{\sum_{y} p(x, y)} = \frac{p(x=6, y)}{p(x=6, y)+p(x=6, \bar{y})}=\frac{15}{18} = 0.8333$$

+++ {"id": "FpVIHfbAZktr"}

# 2. Phân phối xác suất

Phân phối xác suất là một khái niệm liên quan tới biến ngẫu nhiên. Trong thống kê có một số hình dạng phân phối nhất định của biến có thể kể đến như: Phân phối chuẩn, phân phối đều, phân phối possion, phân phỗi nhị thức, phân phối category, phân phối dirichlet.

Những phân phối này đều dựa trên hai kiểu là biến ngẫu nhiên hoặc biến rời rạc.

+++ {"id": "34F3itqPZw1M"}

## 2.1. Phân phối chuẩn (_gaussian distribution_)

Phân phối chuẩn là phân phối nổi tiếng nhất trong thống kê. Nó được tìm ra bởi nhà toán học Gaussian (ông vua của các nhà toán học), một nhà toán học rất nổi tiếng người Đức. Người ta từng ví rằng việc tìm ra qui luật phân phối chuẩn quan trọng giống như việc tìm ra 3 định luật của Newton trong vật lý cổ điển. Người Đức tự hào về phân phối chuẩn đến mức đã cho in hình quả chuông chuẩn trên tờ tiền của họ. Ngoài ra Gaussian còn rất nổi tiếng bởi khả năng tính toán của mình. Ngay từ lớp 2 ông đã làm thầy giáo của mình ngạc nhiên bởi việc tính nhanh tổng các số từ 1 tới 100 (công thức mà ngày nay ai cũng biết). Sau này ông còn tìm lại được một tiểu hành tinh khi các nhà thiên văn mất dấu nó khi nó đi lẫn vào ánh sáng của các hành tinh khác.

![](https://imgur.com/YJ3dTKX.png)

> Hình ảnh của gaussian và phân phối chuẩn trên tờ 10 mark của Đức.

Quay trở lại lý thuyết, phân phối này được mô tả bởi hai tham số: trung bình $\mu$ và phương sai $\sigma^2$. Giá trị của $\mu$ là vị trí trung tâm của đáy phân phối có giá trị của hàm mật độ xác suất là cao nhất. Phân phối có độ rộng đáy càng lớn khi $\sigma^2$ lớn, điều này chứng tỏ khoảng giá trị của biến biến động mạnh, và ngược lại. Hàm mật độ xác suất của phân phối này được định nghĩa là:

$$f(x) = \frac{1}{\sqrt{2\pi \sigma^2}}\exp \left( -\frac{(x - \mu)^2}{2\sigma^2}\right)$$

Trong trường hợp $\mu = 0, \sigma^2 = 1$ thì phân phối chuẩn được gọi là phân phối chuẩn hoá. Đây là trường hợp thường xuyên được sử dụng nhất vì khi đó giá trị của phân phối đối xứng qua trục tung. Điều này tạo thuận lợi cho quá trình tính toán.

Phân phối chuẩn thường được sử dụng nhiều trong thống kê để ước lượng khoảng tin cậy, tính toán xác suất và kiểm định giả thuyết thống kê. Trong pytorch thì chúng ta khởi tạo một chuỗi phân phối chuẩn thông qua hàm `torch.normal()` và phân phối chuẩn hoá (_standard normal distribution_) thông qua hàm `torch.randn()`. Chữ `n` trong đuôi của `randn` đại diện cho `normalization`. Chúng ta cần phân biệt nó với `torch.rand()` của phân phối đều.


```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 404
executionInfo:
  elapsed: 853
  status: ok
  timestamp: 1617014367535
  user:
    displayName: khanhblog AI
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GhNCi9Qnch9sWXSuvX4N5yijAGEjX1IvfmN-95m=s64
    userId: 06481533334230032014
  user_tz: -420
id: 7aLmi8FDiOsE
outputId: e227d120-5be2-4832-d37a-58f87f43deab
---
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def _plot(x, title, figsize=(10, 6)):
  plt.figure(figsize=figsize)
  sns.kdeplot(x)
  plt.xlabel("Value of x")
  plt.title(title)

x = torch.randn(1000)
_plot(x, "Standard Normal Distribution")
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 404
executionInfo:
  elapsed: 1011
  status: ok
  timestamp: 1617014347721
  user:
    displayName: khanhblog AI
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GhNCi9Qnch9sWXSuvX4N5yijAGEjX1IvfmN-95m=s64
    userId: 06481533334230032014
  user_tz: -420
id: P3iix4AtjzHe
outputId: 243c6a18-96e2-443f-a682-3ec3200e2f99
---
x = torch.normal(mean=1, std=2, size=(1000,))
_plot(x, "Normal Distribution")
```

+++ {"id": "nYniwm-zinkN"}

Phân phối chuẩn có mật độ tập trung cao ở khoảng quanh $\mu$ và thấp về phía hai đuôi. Phân phối chuẩn được đặc trưng bởi hình dạng đuôi dày hoặc đuôi mỏng. Nếu đuôi dày thì mật độ quan sát tập trung về đuôi nhiều hơn và do đó chuỗi có hiện tượng phân tán mạnh.

+++ {"id": "AGz3QcK1ZzV7"}

## 2.2. Phân phối đều (_uniform distribution_)

Phân phối đều là một phân phối liên tục trên một đoạn $[a, b]$. Nó có giá trị _pdf_ bằng nhau trên khắp mọi nơi thuộc $[a, b]$.


$$f(x)={\begin{cases}\frac{1}{b-a}&{\text{if }}x \in [a,b]~~\\
0&{\text{ifelse }}\end{cases}}$$

Trong pytorch chúng ta khởi tạo giá trị của phân phối đều bằng hàm `torch.rand()` bên trong hàm này cần khai báo kích thước véc tơ.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 404
executionInfo:
  elapsed: 890
  status: ok
  timestamp: 1617013926374
  user:
    displayName: khanhblog AI
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GhNCi9Qnch9sWXSuvX4N5yijAGEjX1IvfmN-95m=s64
    userId: 06481533334230032014
  user_tz: -420
id: qgRcccFEfHwk
outputId: 601b417d-0f0b-4b98-b1ca-0ceb0e6e2f68
---
x = torch.rand(1000)
_plot(x, "Uniform Distribution")
```

+++ {"id": "ISwib5Gghqpp"}

Ta thấy đỉnh của phân phối đều khá bằng phẳng cho thấy các giá trị được phân phối đều khắp mọi miền trong khoảng xác định của $x$.

+++ {"id": "VUCRil0SaaOx"}

## 2.3. Phân phối possion

Giả sử bạn đang ngồi ở một bệnh viện. Xác suất để phía trước bạn có $x$ người là bao nhiêu? Trung bình khi tới bệnh viện bạn phải chờ trong thời gian bao lâu. Phân phối possion sẽ có tính ứng dụng cao trong trường hợp này. Phân phối possion được đặc trưng bởi hai tham số là $k$ đại diện cho số lượt sự kiện xảy ra và $\lambda$ là kỳ vọng của phân phối.

$$f(k; \lambda) = P(X=k) = \frac{\lambda^{k}e^{-\lambda}}{k!}$$

Phân phối possion trên pytorch được khởi tạo thông qua hàm `torch.possion()`. Bạn cần điền vào một 

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 404
executionInfo:
  elapsed: 1008
  status: ok
  timestamp: 1617015916623
  user:
    displayName: khanhblog AI
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GhNCi9Qnch9sWXSuvX4N5yijAGEjX1IvfmN-95m=s64
    userId: 06481533334230032014
  user_tz: -420
id: KWOn2xNupcuQ
outputId: a2080d33-bc92-4bf2-ed74-3b022ee098a1
---
rates = torch.rand(1000)
x = torch.poisson(rates)
_plot(x, "Possion Distribution")
```

+++ {"id": "tiFqFOekqnIb"}

Bạn có thể hình dung hơn ý nghĩa của phân phối possion thông qua ví dụ sau đây: Thống kê cho thấy độ dài hàng chờ khám bệnh ở bệnh viện tuân theo phân phối possion với kỳ vọng trung bình là $\lambda = 6.5$. Hỏi xác suất để có 2 người đứng trước bạn khi xếp hàng ở bệnh viện là bao nhiêu ?

$$f(k=2; \lambda) = P(X = 3) = \frac{\lambda^{k}e^{-\lambda}}{k!} = \frac{6.5^3e^{-6.5}}{3!}=0.069$$

Chúng ta cũng có thể tính xác suất cho độ dài hàng chờ tuỳ ý.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
executionInfo:
  elapsed: 851
  status: ok
  timestamp: 1617017333557
  user:
    displayName: khanhblog AI
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GhNCi9Qnch9sWXSuvX4N5yijAGEjX1IvfmN-95m=s64
    userId: 06481533334230032014
  user_tz: -420
id: UZLHIDg_p9_h
outputId: 849bf1f6-4260-431b-dbda-694f82f21fa8
---
import numpy as np

def _possion(k , lab):
  arr = torch.arange(1, k+1)
  prod = 1
  for i in arr:
    prod*=i
  poss = lab**k*np.e**(-lab)/prod
  return poss

# Lập phân phối possion với lambda = 6.5 và độ dài hàng chờ từ 1 tới 20.
lab=6.5
poss = []
for k in torch.arange(1, 21):
  poss_i = _possion(k, lab)
  poss.append(poss_i)

for k, p in enumerate(poss):
    print("Probability of the waiting length equal {}: {}%".format(k+1, round(p.numpy()*100, 2)))

print("Sum of probabilities: {}".format(sum(poss).numpy()))
```

+++ {"id": "VT-kjrjhvVxv"}

Ta thấy tổng xác suất của các độ dài hàng chờ từ $1$ tới $20$ là gần bằng 100%. Khả năng độ dài hàng chờ cao nhất là 6 với xác suất là 15.75%. Đây cũng chính là giá trị sát với kỳ vọng của phân phối possion.

+++ {"id": "8KT71w_dZ3Je"}

## 2.4. Phân phối bernoulli

Phân phối bernoulli tính xác xuất để 1 biến cố xảy ra trong 1 lần thử có hàm phân phối xác xuất:

$$f_{p}(k):=P(X=k)={\begin{cases}p&{\text{if }}k=1,\\[6pt]1-p&{\text{if }}k=0.\end{cases}}$$

Giá trị của phân phối bernoulli thường được rút ra từ quá trình thực hiện phép thử với một số lượng lớn để con số đạt được đáng tin cậy hơn. Ví dụ như nếu ta tung đồng xu đồng chất 5 lần thì có khả năng thu được 4 lần sấp 1 lần ngửa vẫn rất cao. Nhưng nếu tung 1000 lần thì tỷ lệ giữa mặt sấp và ngửa sẽ gần bằng 1:1. Đây chính là phân phối xác suất được rút ra từ qui luật số lớn. 

+++ {"id": "DqSBeQDzZ7f-"}

## 2.5. Phân phối category

Phân phối categorical là trường hợp tổng quát khác của phẩn phối bernoulli cho $K$ trường hợp. Khi đó biểu diễn của phân phối là một véc tơ $\lambda = [\lambda_1, \lambda_2, \dots, \lambda_K]$ với $\lambda_i$ là các số không âm có tổng bằng 1. Khi đó xác xuất để 1 biến cố rơi vào nhóm $k$ chính là:


$$f_{p}(k):=P (X=k)= \lambda_k$$

+++ {"id": "V1sOW4X1dej5"}

## 2.6. Phân phối nhị thức

Phân phối nhị thức cho ta biết xác xuất để 1 biến cố kiện xảy ra với tần suất là $k$ khi thực hiện một số lượng phép thử $n$, $n \geq k$. Phân phối này là trường hợp tổng quát của phân phối bernoulli cho tần suất nhiều hơn 1 biến cố xảy ra và có hàm phân phối xác xuất là:
$$f_{p}(k) := P(X=k)=\binom{k}{n}p^{k}(1-p)^{n-k}$$

$\binom{k}{n}$ chính là xác suất để lấy ngẫu nhiên từ $n$ mẫu ra $k$ phần tử mà không xét đến tính thứ tự. Nó còn gọi là tổ hợp chập $k$ của $n$.

$$\binom{k}{n} = \frac{n!}{k!(n-k)!}$$

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
executionInfo:
  elapsed: 890
  status: ok
  timestamp: 1617018575307
  user:
    displayName: khanhblog AI
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GhNCi9Qnch9sWXSuvX4N5yijAGEjX1IvfmN-95m=s64
    userId: 06481533334230032014
  user_tz: -420
id: ypW0B4IXxs1v
outputId: f3c68770-c661-4538-9773-dbc1c80f2fda
---
import numpy as np

# Công thức tính luỹ thừa
def _mul_accum(k):
  assert k>=0
  prod = 1
  for i in np.arange(1, k+1):
    prod*=i
  return prod

# Công thức tính tổ hợp
def _combine(k, n):
  assert (k>=0) & (n>=k)
  acc_k = _mul_accum(k)
  acc_n = _mul_accum(n)
  acc_nk = _mul_accum(n-k)
  c_k_n = acc_n/(acc_k*acc_nk)
  return c_k_n

# Tính phân phối nhị thức
def _binary_dist(n, k, p):
  assert (k>=0) & (n>=k)
  c_k_n = _combine(k, n)
  prob = c_k_n*p**k*(1-p)**(n-k)
  return prob

# k = 2, n = 10, p = 0.5
k = 2
n = 10
p = 0.5
print("Probability of binary distribution with k = 2, n = 10, p = 0.5:")
_binary_dist(n, k, p)
```

+++ {"id": "aaCg8SbMdmpn"}

## 2.7. Phân phối beta

Beta distribution là một phân phối liên tục xác định dựa trên biến ngẫu nhiên $p \in [0, 1]$ nhằm mô tả sự biến động của tham số $p$ trong phân phối bernoulli. Phân phối beta được đặc tả bởi 2 tham số $\alpha, \beta$ không âm theo phương trình hàm mật độ xác xuất:

$$f(p; \alpha, \beta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}p^{\alpha-1}(1-p)^{\beta-1}$$

Trong đó hàm $\Gamma(.)$ được gọi là gamma function có mối liên hệ với giai thừa trong điều kiện tham số của nó là số nguyên.

$$\Gamma(n) = (n-1)!$$

Trong trường hợp tổng quát:

$$\Gamma(z) = \int_0^{\infty} t^{z-1}e^{-t} dt$$

+++ {"id": "jcXNM6loaKST"}

## 2.8. Phân phối dirichlet

Phân phối dirichlet là trường hợp tổng quát của phân phối beta cho $K$ trường hợp. Phân phối sẽ được xây dựng dựa trên một phân phối categorical có véc tơ phân phối dạng $\lambda = [\lambda_1, \lambda_2, \dots, \lambda_K]$ sao cho tổng các phần tử bằng 1. Một véc tơ $\alpha = [\alpha_1, \alpha_2, \dots, \alpha_K]$ sẽ là các tham số đặc trưng tương ứng với mỗi phần tử của véc tơ $\lambda$. Khi đó hàm mật độ xác xuất của véc tơ phân phối $\lambda$ sẽ có dạng:

$$f(\lambda; \alpha) = \frac{\Gamma(\sum_{i=1}^{K}\alpha_i)}{\prod_{i=1}^{K}\Gamma(\alpha_i)} \prod_{i=1}^{K}{\lambda_i^{\alpha_i-1}(1-\lambda_i)^{\alpha_i-1}} $$

Hoặc chúng ta có thể viết gọn dưới dạng:

$$f(\lambda; \alpha) = \text{Dir}_{\lambda}(\alpha)$$

Trên thực thế phân phối Dirichlet chính là phân phối liên hợp của phân phối categorical và phân phối benourlli.

+++ {"id": "tvsNMJuHduMN"}

## 2.9. Phân phối multi-normial

Là một phân phối rất tổng quát vì nó có thể khái quát được nhiều phân phối bao gồm phân phối bernoulli, phân phối categorical, phân phối nhị thức.

Đây là phân phối rời rạc thường xuyên bắt gặp trong thực tế. Bạn sẽ hiểu nó thông qua hai ví dụ như sau:

- Tính xác suất lặp lại $x$ lần một mặt của xúc xắc 6 mặt nếu gieo nó $n$ lần. 

- Trong 1 rổ có 10 bóng đèn trong đó có 4 bóng xanh va 5 bóng đỏ. Tính xác suất để lấy ngẫu nhiên 5 bóng thì thu được 3 bóng xanh và 2 bóng đỏ nếu không phân biệt thứ tự các bóng cùng màu.

Bạn đọc đã hình dung ra phân phối multinomial rồi chứ? Tóm lại, đây là phân phối nhằm tính ra xác suất để có $n_1+n_2+\dots+n_k$ phép thử thành công khi thực hiện $n$ phép thử độc lập. Trong đó $n_i$ là số lần chúng ta thử thành công nhóm $i$.

Trong trường hợp số nhóm $k=2$ và số phép thử $n=1$ chính là phân phối bernoulli. Khi $k>2$ và $n=1$ ta thu được phân phối categorical. Khi $k=2$ và $n>1$ là phân phối nhị thức.

Một lưu ý nhỏ: Trong lĩnh vực xử lý ngôn ngữ tự nhiên, phân phối _multi-normial_ đôi khi còn được gọi là _categorical_.

Gỉa định rằng chúng có ta một chiếc túi đựng $n$ quả bóng của $k$ màu sắc khác nhau. Ta sẽ thực hiện các phép thử lấy ngẫu nhiên các quả bóng từ túi sao cho sau mỗi lần thử các quả bóng được đặt trở lại túi. Kí hiệu biến $X_i = x_i$ là số lượng bóng thực tế đã lấy được từ nhóm màu thứ $i$. Xác xuất để lấy được 1 quả bóng màu $i$ là $p_i$. Khi đó hàm phân phối xác xuất của Multi-nomial có dạng:

$$f_p(x) := P(X_1 = x_1, X_2 = x_2, \dots, X_k = X_k)={\begin{cases}\frac{n!}{x_1!x_2! \dots x_k!}p_1^{x_1}p_2^{x_2}\dots p_k^{x_k} &{\text{when }}\sum_{i=1}^{k}x_i=n,\\ ~~ 0&{\text{if other}}\end{cases}}$$


+++ {"id": "J7qviHn9Z_Dq"}

## 3. Bài tập

1. Tính phân phối xác suất biên của bảng phân phối xác suất sau:




```{code-cell}
:id: saGzBE_MfakM

import torch
A = torch.tensor([[1, 2, 3, 4],
                  [3, 4, 5, 6],
                  [2, 3, 5, 6],
                  [1, 3, 5, 7]])
```

+++ {"id": "_chrO8mBflQ1"}

Biết mỗi một dòng là phân phối xác suất của thời tiết gồm `nắng, mưa, âm u, lạnh` khi cố định địa điểm. Các cột là phân phối xác suất của địa điểm gồm `rất gần, gần, xa, rất xa` theo thời tiết. 

+++ {"id": "CgsncUlFfZi6"}

2. Viết code để tính giá trị xác suất của phân phối multi-normial khi thực hiện $n$ phép thử trong đó có $k$ phép thử thành công. Biết đầu vào là véc tơ xác suất $\mathbf{p}$.

+++ {"id": "Iewjxcnngvj5"}

3. Xác suất để bạn tán đổ thành công một cô gái là 0.1. Hỏi bạn cần phải tán ít nhất bao nhiêu cô gái để xác suất có bạn gái là trên 50%?

4. Một học sinh ngồi đợt xe bus. Biết thời gian khi ngồi chờ xe bus là một biến ngẫu nhiên tuân theo phân phối poission và có gía trị trung bình là $\lambda = 7$ phút. Hỏi xác suất học sinh đó cần phải ngồi chờ xe bus dưới 5 phút là bao nhiêu phần trăm?
