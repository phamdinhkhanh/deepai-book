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

# 11.1. Feature Engineering
## 11.1.1. Trích lọc đặc trưng (_Feature Extraction_)

Ở những bộ dữ liệu cao chiều thì huấn luyện mô hình và dự báo cần tiêu tốn rất nhiều chi phí tính toán. Chính vì thế _trích lọc đặc trưng_ là một kĩ thuật giúp giảm chiều giữ liệu mà ở đó cho phép chúng ta lựa chọn hoặc kết hợp các biến đầu vào thành những _đặc trưng_ dự báo nhưng vẫn thể hiện một cách chính xác và nguyên vẹn của dữ liệu gốc. _Trích lọc đặc trưng_ được áp dụng trong nhiều bài toán khác nhau của machine learning.

  * Autoendcoder: Là kĩ thuật khá hiệu quả trong _self - supervised learning_. Kĩ thuật này sẽ tự mã hoá dữ liệu đầu từ không gian cao chiều sang một không gian thấp chiều (quá trình _encoder_). Sau đó giải mã ngược lại từ không gian thấp chiều sang không gian cao chiều (quá trình _decoder_) sao cho thông tin đầu ra của quá trình giải mã và đầu vào phải gần bằng nhau.
![](https://dothanhblog.files.wordpress.com/2020/01/autoencoder.png?w=467&h=354)
  * Bag-of-Words: Hay còn gọi là _thuật toán túi từ_ thường được sử dụng trong _xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP)_ và _trích lọc thông tin (information retrieval)_. Thuật toán cho phép chúng ta trích lọc thông tin từ các đoạn văn bản, mẩu tin, trang web bằng cách xây dựng một _túi từ_ và tìm cách mã hoá nội dung văn bản thành một véc tơ tần suất của từ mà không quan tâm đến thứ tự của từ và cấu trúc ngữ pháp.
  * Image Processing: Đây là những thuật toán được sử dụng để phát hiện đặc trưng trên ảnh như hình dạng (_shaped_) và cạnh (_edges_). Đó có thể là những phương pháp trích lọc đặc trưng trên ảnh thủ công như [HOG](https://phamdinhkhanh.github.io/2019/11/22/HOG.html) và [SHIFT](http://luthuli.cs.uiuc.edu/~daf/courses/ComputerVisionTutorial2012/EdgesOrientationHOGSIFT-2012.pdf) hoặc sử dụng bộ trích lọc đặc trưng thông qua tích chập [CNN](https://phamdinhkhanh.github.io/2019/08/22/convolutional-neural-network.html).

+++ {"id": "NgtLHNIk0nHZ"}

## 11.1.2. Biến đổi đặc trưng (_Feature Transformation_)

_Biến đổi đặc trưng_ là những kĩ thuật giúp biến đổi dữ liệu đầu vào thành những dữ liệu phù hợp với mô hình nghiên cứu. Những dữ liệu này thường có tương quan cao đối với biến mục tiêu và do đó giúp cải thiện độ chính xác của mô hình. Bên dưới là một số phương pháp chính được áp dụng trong biến đối đặc trưng:
  * **Chuẩn hóa biến**: Chuẩn hoá biến nhằm mục đích tạo ra sự đồng nhất đơn vị giữa các biến đầu vào và giảm thiểu những tác động xấu lên mô hình do sự khác biệt về độ lớn giữa các biến. Các kĩ thuật liên quan đến chuẩn hoá đơn vị cho biến đầu vào còn được gọi là _Feature Scaling_ bao gồm: _Chuẩn hoá MinMax (Minmax scaling_), _chuẩn hoá độ dài đơn vị_ (_Unit length scaling_), _chuẩn hoá phân phối chuẩn_ (_Standardization_).
  * **Biến đổi biến theo hàm**: Trong trường hợp dữ liệu có _phương sai thay đổi (heteroscedasticity)_ thì chúng ta có thể sử dụng một số hàm biến đổi biến đầu vào để tạo ra những biến có phương sai ổn định và dạng phân phối gần với phân phối chuẩn hơn như `logrith, căn bậc 2, căn bậc 3`.
  * **Tạo biến tương tác**: Các biến tương tác là những biến kết hợp từ nhiều biến đầu vào chẳng hạn như $x_1x_2, x_1^2x_2, x_1x_2x_3^2, \dots$ Biến tương tác có thể là tích của hai hoặc nhiều biến. Trong một mô hình có ít biến đầu vào thì sử dụng biến tương tác có thể giúp tạo ra nhiều biến giải thích mới giúp ích cho mô hình.
  * **Tạo biến bậc cao**: Biến bậc cao là những biến được tạo thành từ biến đầu vào bằng cách luỹ thừa với giá trị bậc cao, có thể là bậc 2, 3,.... Chẳng hạn với biến đầu vào là $x_1$ thì biến bậc cao của nó là $x_1^2, x_1^3,....$.
  * **Dữ liệu về vị trí địa lý**: Từ vị trí địa lý có thể suy ra vùng miền, thành thị, nông thôn, mức thu nhập trung bình, các yếu tố về nhân khẩu,....
  * **Dữ liệu thời gian**: Các dữ liệu chuỗi thời gian thường tồn tại tính chu kì và mùa vụ. Chính vì vậy, các kĩ thuật biến đổi biến thời gian thành đặc trưng ghi nhận tính chất chu kì và mùa vụ sẽ giúp tăng cường khả năng giải thích của mô hình đối với biến mục tiêu. Chúng ta có thể lựa chọn chu kì của thời gian là buổi sáng/chiều/tối trong ngày; ngày trong tháng; tuần trong tháng; tháng trong năm hoặc quí trong năm tuỳ theo qui luật mùa vụ được thể hiện ở _biến mục tiêu_.

+++ {"id": "KIxF_KIr0paQ"}

## 11.1.3. Lựa chọn đặc trưng (_Feature Selection_) 

_Lựa chọn đặc trưng_ là một phần rất quan trọng trong Machine Learning với mục tiêu chính là loại bỏ những đặc trưng không thực sự chứa thông tin hữu ích cho bài toán phân loại hoặc dự báo. Kĩ thuật _lựa chọn đặc trưng_ có thể được sử dụng để cải thiện tốc độ huấn luyện và dự báo (khi có ít đặc trưng hơn có nghĩa là mô hình được huấn luyện và dự báo nhanh hơn) và thậm chí giảm hiện tượng _quá khớp_.

Các kĩ thuật lựa chọn đặc trưng khá đa dạng:
  * Sử dụng hệ số tương quan với biến mục tiêu: Những biến tương quan cao với biến mục tiêu là những biến có khả năng giải thích tốt. Mức độ quan trọng của biến có thể được xếp hạng thông qua sử dụng tương quan Pearson Correlation,
  * Sử dụng chỉ số AIC: AIC (_Akaike information criterion_) là chỉ số được sử dụng để đánh giá chất lượng của mô hình thống kê. Chỉ số này được tính toán thông qua giá trị logarith của hàm hợp lý (_Log Likelihood Function_). Để xếp hạng mức độ quan trọng của biến thì đầu tiên chúng ta sẽ tính AIC cho mô hình được hồi qui trên toàn bộ các biến. Sau đó thực hiện các thử nghiệm huấn luyện mà mỗi lượt bỏ bớt đi một biến để xem giá trị AIC của mô hình nào là nhỏ nhất. AIC càng nhỏ thì mô hình có sai số càng thấp trên tập huấn luyện và từ đó đưa ra xếp hạng biến.
  * Sử dụng chỉ số IV: IV (_Information Value_) là chỉ số được sử dụng trong các bài toán phân loại nhị phân trong thống kê. Chỉ số này thường được đo lường để đánh giá sức mạnh phân loại của biến đầu vào.
  * Lựa chọn đặc trưng bằng sử dụng mô hình: Random Forest, Lasso Regression, Neural Network, SVD.
  * Lựa chọn thông qua mức độ biến động phương sai: Những biến ít biến động hoặc thậm chí không thay đổi giá trị sẽ không có tác dụng phân loại và dự báo. Chính vì vậy chúng ta có thể lọc bỏ những biến này thông qua xác định độ lớn của phương sai phải lớn hơn một ngưỡng cho trước.

+++ {"id": "LitAgjBBIDJH"}

Tiếp theo chúng ta sẽ lần lượt phân tích những kĩ thuật này về lý thuyết, trường hợp áp dụng thông qua các ví dụ thực hành.

+++ {"id": "lDa9SWPAbpIp"}

# 11.2. Trích lọc đặc trưng (feature extraction)

Trong thực tế dữ liệu thường ở dạng thô và đến từ nhiều nguồn khác nhau như văn bản, hình ảnh, âm thanh, các phiếu điều tra, các hệ thống lưu trữ, website, app,.... Nên đòi hỏi người xây dựng mô hình phải thu thập và tổng hợp lại các nguồn dữ liệu có liên quan đến vấn đề đang nghiên cứu. Dữ liệu sau đó phải được làm sạch và biến đổi thành dạng dữ liệu cấu trúc (_structure data_) để tiến hành xây dựng mô hình. 

Đối với các dữ liệu dạng văn bản, hình ảnh hoặc âm thanh chúng ta sẽ cần đến các kĩ thuật trích lọc đặc trưng để biến dữ liệu từ dạng chưa mã hoá sang dạng số học thì mới có thể huấn luyện được mô hình. Một trong những kiểu dữ liệu phổ biến áp dụng kĩ thuật trích lọc này là dữ liệu dạng văn bản sẽ được trình bày bên dưới.

+++ {"id": "eleZxW5obvsK"}

## 11.2.1. Trích lọc đặc trưng cho văn bản

Dữ liệu văn bản có thể tồn tại ở nhiều dạng khác nhau như chữ cái thường, chữ cái hoa, dấu câu, các kí tự đặc biệt,.... Các ngôn ngữ khác nhau cũng có mẫu kí tự khác nhau và cấu trúc ngữ pháp khác nhau.



+++ {"id": "vteyIKkeb5yK"}

Vấn đề chính của dữ liệu dạng văn bản đó là làm thể nào để mã hoá được kí tự về dạng số? Kĩ thuật _tokenization_ sẽ giúp ta thực hiện điều này. _tokenization_ là việc chúng ta chia văn bản theo đơn vị nhỏ nhất và xây dựng một từ điển đánh dấu index cho những đơn vị này. Có hai kiểu mã hoá chính là mã hoá theo từ và mã hoá theo kí tự. 

* Đối với mã hoá theo từ thì các từ trong câu sẽ là đơn vị nhỏ nhất. Trong Tiếng Anh thì từ chủ yếu tồn tại ở dạng từ đơn trong khi Tiếng Việt tồn tại các từ ghép. Khi mã hoá theo từ thì kích thước của từ điển sẽ rất lớn, tuỳ thuộc vào số lượng các từ khác nhau xuất hiện trong toàn bộ các văn bản.

* Mã hoá theo kí tự thì chúng ta sẽ sử dụng các kí hiệu trong bảng chữ cái để làm từ điển mã hoá từ. Kích thước của bộ từ điển khi mã hoá theo kí tự sẽ nhỏ hơn so với mã hoá theo từ.

+++ {"id": "4wLVJnT8tgfi"}

### 11.2.1.1. Phương pháp _bag-of-words_

_bag-of-words_, viết tắt là BoW, có nghĩa là bỏ túi các từ. Theo phương pháp _bag-of-word_ chúng ta sẽ mã hoá các từ trong câu thành một véc tơ có độ dài bằng số lượng các từ trong từ điển và đếm tần suất xuất hiện của các từ. Tần xuất của từ thứ $i$ trong từ điển sẽ chính bằng phần tử thứ $i$ trong véc tơ.


![](https://imgur.com/UuynVys.jpeg)


**Hình 1:** Văn bản ở bên trái được mã hoá thành véc tơ tần suất từ ở bên phải. Các từ `I` và `have` lặp lại 2 lần nên có tần suất là 2. Những từ không xuất hiện trong câu nhưng có trong từ điển như `deep, is, this, machine, learning` thì có giá trị là 0.


Như vậy theo phương pháp _bag-of-words_ thì mỗi từ sẽ trở thành một chiều biểu diễn trong không gian của véc tơ đầu ra. Khi số lượng các từ rất lớn thì kết quả mã hoá có thể tạo thành một véc tơ có độ dài rất lớn. Thông thường đây sẽ là một véc tơ thưa (_sparse vector_) có hầu hết các giá trị bằng 0. Số lượng chiều lớn khiến việc biểu diễn các véc tơ mã hoá trên không gian gặp khó khăn. Nếu như ta muốn biểu diễn trên đồ thị thì phải tìm cách giảm chiều véc tơ xuống còn 2 hoặc 3 chiều.

+++ {"id": "SNt6cXAnb7Or"}

Bên dưới là code minh hoạ cho phương pháp túi từ. Để xây dựng phương pháp túi từ chúng ta trải qua hai bước:

1. Xây dựng từ điển.
2. Mã hoá văn bản sang véc tơ tần suất của từ.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: bpDX0eu-b0WF
outputId: 3fc6db64-2718-434e-c346-f1e951e2eb82
---
from functools import reduce
import numpy as np

# Đầu vào là một texts bao gồm 3 câu văn:
texts = [['i', 'have', 'a', 'cat'], 
        ['he', 'has', 'a', 'dog'], 
        ['he', 'has', 'a', 'dog', 'and', 'i', 'have', 'a', 'cat']]

# B1: Xây dựng từ điển
dictionary = list(enumerate(set(reduce(lambda x, y: x + y, texts))))

# B2: Mã hoá câu sang véc tơ tần suất
def bag_of_word(sentence):
    # Khởi tạo một vector có độ dài bằng với từ điển.
    vector = np.zeros(len(dictionary))
    # Đếm các từ trong một câu xuất hiện trong từ điển.
    for i, word in dictionary:
        count = 0
        # Đếm số từ xuất hiện trong một câu.
        for w in sentence:
            if w == word:
                count += 1
        vector[i] = count
    return vector
            
for i in texts:
    print(bag_of_word(i))
```

+++ {"id": "8UbOTTwc23VA"}

Nếu muốn sử dụng thư viện để tìm biểu diễn _bag-of-words_ của từ thì trong sklearn chúng ta sử dụng package như sau:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: M_fFNLnc3EsF
outputId: d1d8b850-8bc1-416d-8fd9-aeebece5c2aa
---
from sklearn.feature_extraction.text import CountVectorizer

texts = ['i have a cat', 
        'he has a dog', 
        'he has a dog and i have a cat']

vect = CountVectorizer()
X = vect.fit_transform(texts)
print('words in dictionary: ', vect.get_feature_names())
print(X.toarray())
```

+++ {"id": "DHw1USjhcB63"}

Quá trình này có thể được mô tả bởi biểu đồ bên dưới:

![](https://imgur.com/JgITfRU.jpeg)

Các biểu diễn theo túi từ có hạn chế đó là chúng ta không phân biệt được 2 câu văn có cùng các từ bởi túi từ không phân biệt thứ tự trước sau của các từ trong một câu. Chặng như 'you have no dog' và 'no, you have dog' là 2 câu văn có biểu diễn giống nhau mặc dù có ý nghĩa trái ngược nhau.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: PYCZVJom2xdQ
outputId: 6aa97456-c809-4f6d-8545-2e712390cf63
---
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(ngram_range = (1, 1))
vect.fit_transform(['you have no dog', 'no, you have dog']).toarray()
```

+++ {"id": "jV2fg0877xoC"}

Chính vì thế phương pháp _bag-of-n-gram_ sẽ được sử dụng thay thế.

+++ {"id": "GDcWPViA02-P"}

### 11.2.1.2. Phương pháp _bag-of-n-gram_

Phương pháp _bag-of-n-grams_ là phương pháp mở rộng của _bag-of-words_. Một _n-grams_ là một chuỗi bao gồm $n$ tokens. Trong trường hợp $n=1$ từ ta gọi là _unigram_, đối với 2 từ là _bigram_ và 3 từ là _trigram_. Khi thực hiện tokenization với _n-grams_ thì trong từ điển sẽ xuất hiện những cụm $n-grams$ từ nếu chúng xuất hiện trong các văn bản. Chẳng hạn như câu `I have a dog` sẽ được tokenize thành `I have, have a, a dog`. Như vậy số lượng các từ trong từ điển sẽ gia tăng một cách đáng kể. Nếu chúng ta có $k$ từ đơn thì có thể lên tới $k^2$ từ trong bigram. Nhưng thực tế không phải hầu hết các từ đều có thể ghép đôi với nhau nên véc tơ biểu diễn của câu trong _bigram_ là một véc tơ rất thưa và có số chiều lớn. Điều này dẫn tới tốn kém về chi phí tính toán và lưu trữ.

Trong `sklearn`, để sử dụng _bigram_ thì trong `CountVectorizer` chúng ta thay đổi `ngram_range = (2, 2)`. Giá trị đầu tiên là độ dài nhỏ nhất và giá trị sau là độ dài lớn nhất được phép của các _ngrams_. Ở đây ta khai báo độ dài nhỏ nhất và lớn nhất là 2 nên thu được _ngrams_ là _bigram_.


```{code-cell}
:id: bRVxmd2fcXHO

from sklearn.feature_extraction.text import CountVectorizer

# bigram
bigram = CountVectorizer(ngram_range = (2, 2))
n1, n2, n3 = bigram.fit_transform(['you have no dog', 'no, you have dog', 'you have a dog']).toarray()

# trigram
trigram = CountVectorizer(ngram_range = (3, 3))
n1, n2, n3 = trigram.fit_transform(['you have no dog', 'no, you have dog', 'you have a dog']).toarray()
```

+++ {"id": "UH6nN2UKBDN3"}

Sau khi mã hoá các câu văn chúng ta cũng có thể tính toán được khoảng cách giữa các véc tơ trong không gian euclidean:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: -oXABBA6BD5K
outputId: 9c950f58-c3ae-4930-8807-260e1142052c
---
from scipy.spatial.distance import euclidean
print(euclidean(n1, n2), euclidean(n2, n3), euclidean(n1, n3))
```

+++ {"id": "sRe4wNzkB4C8"}

### 11.2.1.3. Phương pháp _TF-IDF_

+++ {"id": "FfCtelTGcMF9"}

Giả sử chúng ta có một _bộ văn bản_ (_corpus_) bao gồm rất nhiều các văn bản con. Những từ hiếm khi được tìm thấy trong bộ văn bản (_corpus_) nhưng có mặt trong một số chủ đề nhất định có thể chiếm vai trò quan trọng hơn. Ví dụ đối với chủ đề gia đình thì các từ như `cha mẹ, ông bà, con cái, anh em, chị em` xuất hiện nhiều hơn so với các chủ đề khác.

Ngoài ra cũng có những từ xuất hiện rất nhiều trong văn bản nhưng chúng xuất hiện ở hầu như mọi chủ đề, mọi văn bản chẳng hạn như `the, a, an`. Những từ như vậy được gọi là  _stopwords_ vì chúng không có nhiều ý nghĩa đối với việc phân loại văn bản. Khi mã hoá ngôn ngữ thì chúng ta sẽ tìm cách loại bỏ những từ _stopwords_ bằng cách sử dụng từ điển có sẵn các từ _stopwords_ quan trọng.

Phương pháp TF-IDF là một phương pháp mà chúng ta sẽ đánh trọng số cho các từ mà xuất hiện ở một vài văn bản cụ thể lớn hơn thông qua công thức:

$$\begin{eqnarray}\text{idf}(t,D) & = & \log\frac{\mid D \mid}{|\{d \in D; t \in d \}|+ 1} = \log \frac{\mid D\mid}{\text{df}(d, t)+ 1} \\
\text{tfidf}(t,d,D) & = & \text{tf}(t,d) \times \text{idf}(t,D)
\end{eqnarray}$$

trong đó:

* $\mid D \mid$ là số lượng các văn bản trong _bộ văn bản_.
* $\text{df}(d, t) = |\{d \in D; t \in d \}|$ là tần suất các văn bản $d \in D$ mà từ $t$ xuất hiện. 
* $\text{tf}(t,d)$ là tần suất xuất hiện của từ $t$ trong văn bản $d$.

Như vậy $\text{idf}(t, D)$ là chỉ số _nghịch đảo tần suất văn bản_ (_inverse document frequency_) chỉ số này bằng logarith của nghịch đảo số lượng văn bản chia cho số lượng văn bản chứa một từ cụ thể $t$. Một từ cụ thể có $\text{idf}(t,D)$ lớn chứng tỏ rằng từ đó chỉ xuất hiện trong một số ít các văn bản.

$\text{tfidf}(t, d, D)$ tỷ lệ thuận với _tần suất của từ xuất hiện trong văn bản_ và _nghịch đảo tần suất văn bản_. Ta có thể giải thích ý nghĩa của $\text{tfidf}$ đối với đánh giá mức độ quan trọng của từ như sau: Khi một từ càng quan trọng thì nó sẽ có tần suất xuất hiện trong một văn bản cụ thể, chẳng hạn văn bản $d$ lớn, tức là $\text{tf}(t,d)$ lớn; Đồng thời từ đó phải không là _stopwords_, tức là số lượng văn bản mà nó xuất hiện trong toàn bộ bộ văn bản nhỏ, suy ra $\text{idf}(t, D)$ phải lớn.

Để mã hoá văn bản dựa trên phương pháp tfidf chúng ta sử dụng package `sklearn` như sau:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: ai99v4_ZdIRh
outputId: 40c52488-e592-48b8-db02-5bf1bf1defe5
---
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
 	'tôi thích ăn bánh mì nhân thịt',
	'cô ấy thích ăn bánh mì, còn tôi thích ăn xôi',
	'thị trường chứng khoán giảm làm tôi lo lắng',
	'chứng khoán sẽ phục hồi vào thời gian tới. danh mục của tôi sẽ tăng trở lại',
  'dự báo thời tiết hà nội có mưa vào chiều và tối. tôi sẽ mang ô khi ra ngoài'
]

# Tính tfidf cho mỗi từ. max_df để loại bỏ stopwords xuất hiện ở hơn 90% các câu
vectorizer = TfidfVectorizer(max_df = 0.9)
# Tokenize các câu theo tfidf
X = vectorizer.fit_transform(corpus)
print('words in dictionary:')
print(vectorizer.get_feature_names())
print('X shape: ', X.shape)
```

+++ {"id": "N8phDY0udM9b"}

Ta có thể thấy từ `tôi` xuất hiện ở toàn bộ các câu và không mang nhiều ý nghĩa của chủ đề của câu nên có thể coi là một _stopword_. Bằng phương pháp lọc cận trên của tần suất xuất hiện từ trong văn bản là 90% ta đã loại bỏ được từ này khỏi dictionary.

Các phương pháp bỏ túi có thể tìm được một số cuộc thi trên kaggle như [Catch me if you can competition](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking), [bag of app](https://www.kaggle.com/xiaoml/bag-of-app-id-python-2-27392), [bag of event](http://www.interdigital.com/download/58540a46e3b9659c9f000372):

+++ {"id": "evCPheQlJjjX"}

### 11.2.1.4. Word2vec

Word2vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located close to one another in the space.[1]

word2vec là một nhóm các mô hình sử dụng để tạo ra biểu diễn nhúng cho từ. Những mô hình này tương đối nông, chỉ bao gồm những mạng neural 2 layers được huấn luyện để tái tạo lại bối cảnh ngôn ngữ cho từ. Thông qua mô hình word2vec mỗi một từ trong một _bộ văn bản_ được biểu diễn thông qua một véc tơ trong không gian cao chiều, có thể lên tới hàng trăm chiều, sao cho các từ có chung ngữ cảnh sẽ được đặt gần nhau hơn trong không gian.

Chẳng hạn dưới đây là một ví dụ sau khi thực hiện mã hoá từ thông qua mô hình word2vec thì các từ `king, queen, man, woman` có mối liên hệ theo công thức: king - man + woman = queen

![](https://camo.githubusercontent.com/7acb5beb08711a6e75b6eadb90fdf48fb67c67d87f45812dc8cbd8426c1ee44f/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f3830302f312a4b3558344e2d4d4a4b743846474674725448776964672e676966)

**Hình 2**: Mô hình word2vec đã định vị véc tơ biểu diễn cho những từ có chung ngữ cảnh thì được đặt gần nhau hơn. Để thực hiện được những biểu diễn từ chính xác, các mô hình cần được đào tạo trên các tập dữ liệu rất lớn để bao quát được đa dạng các ngữ cảnh khác nhau của từ. Các mô hình pretrained cho xử lý ngôn ngữ tự nhiên có thể được tải về tại [word2vec - api](https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-models).

Các phương pháp tương tự được áp dụng trong các lĩnh vực khác như trong tin sinh. Một ứng dụng khác nữa là [food2vec](https://jaan.io/food2vec-augmented-cooking-machine-intelligence/).


+++ {"id": "ejsoOzi0nhYm"}

Tại một vị trí cụ thể trong câu văn chúng ta sẽ xác định được một từ mục tiêu và các từ bối cảnh. Từ mục tiêu là từ ở vị trí được lựa chọn còn từ bối cảnh là những từ ở vị trí xung quanh giúp tạo ra bối cảnh ngữ nghĩa cho từ mục tiêu.

Giả sử chúng ta có một câu văn như sau: "Tôi muốn một chiếc cốc màu xanh". Nếu lựa chọn một _context window_ bao gồm 3 từ liền kề thì chúng ta sẽ lần lượt thu được các bộ 3 từ: `tôi muốn một, muốn một chiếc, một chiếc cốc, chiếc cốc màu, cốc màu xanh`. Đối với những bộ 3 từ này thì các từ ở giữa sẽ là từ mục tiêu và từ bối cảnh là những từ ở đầu và ở cuối. Như vậy chúng ta sẽ có các cặp từ mục tiêu và bối cảnh như sau:

`[(('tôi', 'một'), 'muốn'), (('muốn', 'chiếc'), 'một'), (('một', 'cốc'), 'chiếc'), (('chiếc', 'màu'), 'cốc'), (('cốc', 'xanh'), 'màu')]`


Mô hình word2vec có 2 phương pháp chính là skip-grams và CBOW như sau:

![](https://imgur.com/41qQJ2u.jpeg)

**Hình 3:** Mô hình CBOW và Skip-gram trong word2vec.

+++ {"id": "f7kctHXpqZiI"}

#### 11.2.1.4.1. Phương pháp CBOW

Đối với mô hình CBOW chúng ta sẽ xây dựng một mô hình học có giám sát sử dụng đầu vào là các từ bối cảnh, chẳng hạn như trong hình là các từ $\mathbf{w}_{t-2}, \mathbf{w}_{t-1}, \mathbf{w}_{t+1}, \mathbf{w}_{t+2}$ để giải thích từ mục tiêu ở vị trí hiện tại là $\mathbf{w}_t$.

Các từ $\mathbf{w}_t$ đã được mã hoá dưới dạng véc tơ one-hot trong không gian $\mathbb{R}^{d}$ chiều để có thể đưa vào huấn luyện. Ở đây $d$ chính là kích thước của từ điển. Như vậy ở phương pháp CBOW chúng ta có 5 véc tơ one-hot đầu vào với số chiều bằng với số lượng từ trong từ điển. Sau đó những véc tơ này được giảm chiều dữ liệu thông qua một phép chiếu lên không gian thấp chiều, chẳng hạn 200 chiều, bước này chính là projection trên hình vẽ. Kết quả thu được là một véc tơ embedding $\mathbf{e}_c \in \mathbb{R}^{200}$. Sau cùng, phân phối xác suất của từ mục tiêu được dự báo thông qua một hàm softmax áp dụng lên véc tơ $\mathbf{e}_c$. Quá trình huấn luyện mô hình sẽ dựa trên hàm softmax dạng cross-entropy:

$$\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{i=1}^{d} y_i\log(\hat{y}_i)$$

Trong đó $\hat{y}_i$ là xác suất dự báo từ mục tiêu tương ứng với từ ở vị trí index thứ $i$ trong từ điển, được tính theo công thức softmax:

$$\hat{y_i} = \frac{\exp(\mathbf{w}_{:i}^{\intercal}\mathbf{e}_c)}{\sum_{i=1}^{d}\exp(\mathbf{w}_{:i}^{\intercal}\mathbf{e}_c)}$$

$\mathbf{w}_{:i} \in \mathbb{R}^{200}$ chính là véc tơ tham số kết nối toàn bộ các node thuộc $\mathbf{e}_c$ tới vị trí node thứ $i$ của layer cuối cùng.

Sau quá trình lan truyền thuận và lan truyền ngược, các hệ số của mô hình sẽ được cập nhật và chúng ta sẽ thu được biểu diễn từ dần chuẩn xác hơn. Một từ đầu vào sẽ có biểu diễn thông qua phương pháp CBOW chính là véc tơ $\mathbf{e}_c$.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 7nIH6q3q5w9p
outputId: 348ce5e6-d719-4bb3-c661-6fa56e3274cd
---
from tensorflow.keras.preprocessing import text
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from nltk.corpus import gutenberg
from string import punctuation
import nltk

# download bộ văn bản gutenberg
nltk.download('gutenberg')
nltk.download('punkt')
norm_bible = gutenberg.sents('bible-kjv.txt') 
norm_bible = [' '.join(doc) for doc in norm_bible]

# tokenize văn bản
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(norm_bible)
word2id = tokenizer.word_index

# khởi tạo từ điển cho bộ văn bản
word2id['PAD'] = 0
id2word = {v:k for k, v in word2id.items()}
vocab_size = len(word2id)

print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(word2id.items())[:10])
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 9aNT99Vh54ZI
outputId: 562e677c-5445-4586-9739-9492c32ed420
---
# Mã hoá câu văn bằng index
wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in norm_bible]
print('Embedding sentence by index: ', wids[:5])
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: pKii69Bt6QOl
outputId: 667ea7c6-40c0-4c9f-d44e-347cfbbdcb34
---
%%script echo skipping
# Xác định context and target
import numpy as np
def generate_context_word_pairs(corpus, window_size, vocab_size):
    context_length = window_size*2
    for words in corpus:
        sentence_length = len(words)
        # print('words: ', words)
        for index, word in enumerate(words):
            context_words = []
            label_word   = [] 
            # Start index of context
            start = index - window_size
            # End index of context
            end = index + window_size + 1
            # List of context_words
            context_words.append([words[i] for i in range(start, end) if 0 <= i < sentence_length and i != index])
            # List of label_word (also is target word).
            # print('context words {}: {}'.format(context_words, index))
            label_word.append(word)
            # Padding the input 0 in the left in case it does not satisfy number of context_words = 2*window_size.
            x = sequence.pad_sequences(context_words, maxlen=context_length)
            # print('context words padded: ', x)
            # Convert label_word into one-hot vector corresponding with its index
            y = to_categorical(label_word, vocab_size)
            yield (x, y)
            
            
# Test this out for some samples
i = 0
window_size = 2 # context window size
for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
    if 0 not in x[0]:
        print('Context (X):', [id2word[w] for w in x[0]], '-> Target (Y):', id2word[np.argwhere(y[0])[0][0]])
    
        if i == 10:
            break
        i += 1
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: _4qLOFaz6Ypa
outputId: aa446297-4629-46ce-bf5a-991b3cb13f00
---
# Xây dựng mô hình CBOW là một mạng fully connected gồm 3 layers
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Lambda
embed_size = 100
window_size=2
# build CBOW architecture
cbow = Sequential()
cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size*2))
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
cbow.add(Dense(vocab_size, activation='softmax'))
cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# view model summary
print(cbow.summary())
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: x6B827tm6oGb
outputId: 3dbc6612-44dc-4719-b1a0-8fab55d0a5d2
---
%%script echo skipping
# Huấn luyện model với 5 epochs với 100 quan sát đầu tiên

for epoch in range(1, 6):
    loss = 0.
    i = 0
    for x, y in generate_context_word_pairs(corpus=wids[:100], window_size=window_size, vocab_size=vocab_size):
        i += 1
        loss += cbow.train_on_batch(x, y)
        if i % 500 == 0:
            print('Processed {} (context, word) pairs'.format(i))

    print('Epoch:', epoch, '\tLoss:', loss)
```

+++ {"id": "QmpkT45z66uT"}

#### 11.2.1.4.2. Phương pháp skip-gram

Phương pháp skip-gram thực chất là một phiên bản đảo ngược của phương pháp CBOW. Chúng ta sẽ sử dụng đầu vào là các từ mục tiêu và dự báo các từ bối cảnh dự vào từ mục tiêu. Như thể hiện ở _hình 3_ thì $\mathbf{w}_t$ chính là từ mục tiêu được sử dụng làm đầu vào, các từ $\mathbf{w}_{t-2}, \mathbf{w}_{t-1}, \mathbf{w}_{t+1}, \mathbf{w}_{t+2}$ là những từ bối cảnh cần được dự đoán. Những từ này đều được mã hoá thành véc tơ one-hot trong không gian $\mathbb{R}^{d}$. Sau đó véc tơ one-hot sẽ được chiếu lên không gian nhằm giảm chiều dữ liệu xuống còn chẳng hạn $200$ chiều. Đầu ra thu được là véc tơ $\mathbf{e}_c$ có kích thước 200, đây cũng chính là biểu diễn nhúng của từ trong skip-gram. Cuối cùng chúng ta sử dụng một sigmoid layer để dự đoán xem từ mục tiêu $\mathbf{w}_t$ và từ bối cảnh $\mathbf{w}_j$ ($\mathbf{w}_j$ được lựa chọn ngẫu nhiên từ từ điển) có cùng bối cảnh hay không?


```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: S3Qn8clu-G3t
outputId: 36939f4e-a368-4ff2-983d-d91a25442ded
---
from tensorflow.keras.preprocessing.sequence import skipgrams

window_size=2
# generate skip-grams
skip_grams = [skipgrams(wid, vocabulary_size=vocab_size, window_size=window_size) for wid in wids[:100]]

# view sample skip-grams
pairs, labels = skip_grams[0][0], skip_grams[0][1]
for i in range(10):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
          id2word[pairs[i][0]], pairs[i][0], 
          id2word[pairs[i][1]], pairs[i][1], 
          labels[i]))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: fAuzsMFg-F-T
outputId: 185f0272-57ba-4deb-8f1e-d7da91ea3007
---
from tensorflow.keras.layers import dot, concatenate
from tensorflow.keras import Input
from tensorflow.keras.layers import Dot, Dense, Reshape, Embedding
from tensorflow.keras.models import Sequential, Model

# build skip-gram architecture
word_input = Input(shape = (1,))
word_embed = Embedding(vocab_size, embed_size,
                         embeddings_initializer="glorot_uniform",
                         input_length=1, name = 'word_embedding')(word_input)
word_output = Reshape((embed_size, ))(word_embed)
word_model = Model(word_input, word_output)

print('word_model: \n', word_model.summary())
context_input = Input(shape = (1,))
context_embed = Embedding(vocab_size, embed_size,
                  embeddings_initializer="glorot_uniform",
                  input_length=1, name = 'context_embedding')(context_input)
context_output = Reshape((embed_size,))(context_embed)
context_model = Model(context_input, context_output)
print('context_model: \n', context_model.summary())

concate = dot([word_output, context_output], axes = -1)
dense = Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid")(concate)
model = Model(inputs = [word_input, context_input], outputs = dense)
model.compile(loss="mean_squared_error", optimizer="rmsprop")

# view model summary
print('model merge word and context: \n', model.summary())
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: js_-ZxQF-dg_
outputId: ef62c8f1-5dcc-4810-d5f2-289c21b65c13
---
%%script echo skipping
# Để cho nhanh thì mình sẽ training trên 100 skip_grams đầu tiên.
for epoch in range(1, 6):
    loss = 0
    for i, elem in enumerate(skip_grams[:100]):
        pair_first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
        pair_second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
        labels = np.array(elem[1], dtype='int32')
        X = [pair_first_elem, pair_second_elem]
        Y = labels
        if i % 500 == 0:
            print('Processed {} (skip_first, skip_second, relevance) pairs'.format(i))
        loss += model.train_on_batch(X,Y)  

    print('Epoch:', epoch, 'Loss:', loss)
```

+++ {"id": "GQQyhbBO_tZ2"}

#### 11.2.1.4.3. Sử dụng gensim huấn luyện mô hình word2vec

Huấn luyện mô hình word2vec sử dụng mạng nơ ron là để chúng ta hiểu rõ hơn về cấu trúc mạng nơ ron và cách thức hoạt động của mạng. Trên thực tế để huấn luyện mô hình word2vec chúng ta có thể thông qua package gensim như sau:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: IqgEsBLw_1vo
outputId: 54f1f7f3-68ef-4ed4-a1bf-9d2e0b6a8511
---
from gensim.models import Word2Vec
# Training model với 1000 câu đầu tiên trong kinh thánh
sentences = [[item.lower() for item in doc.split()] for doc in norm_bible[:1000]]
model = Word2Vec(sentences, min_count = 1, vector_size = 150, window = 10, sg = 1, workers = 8)
model.train(sentences, total_examples = model.corpus_count, epochs = 10)
```

+++ {"id": "K2mgxKyqAMoH"}

Tìm biểu diễn véc tơ nhúng của một từ:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: tL09kmgiASqz
outputId: 121aba28-4aa3-4486-f9f5-4e532a3c8d34
---
print('embedding vector shape: ', model.wv['king'].shape)
model.wv['king'][:5]
```

+++ {"id": "zA6BCPt9d98i"}

## 11.2.2. Trích lọc đặc trưng trong xử lý ảnh

Trong quãng thời gian trước đây khi tài nguyên tính toán còn hạn chế và "thời kì phục hưng của mạng thần kinh" vẫn chưa thực sự quay trở lại, khai phá đặc trưng cho dữ liệu hình ảnh là một lĩnh vực phức tạp. Người ta phải thiết kế những bộ trích lọc thủ công để trích lọc các đặc trưng như góc, cạnh, đường nét ngang, dọc, chéo,.... Những thuật toán như [HOG](https://phamdinhkhanh.github.io/2019/11/22/HOG.html), [SHIFT](http://luthuli.cs.uiuc.edu/~daf/courses/ComputerVisionTutorial2012/EdgesOrientationHOGSIFT-2012.pdf) là phương pháp thường được sử dụng để trích lọc đặc trưng. Nhược điểm của những phương pháp này đó là tách rời bộ trích lọc đặc trưng (_feature extractor_) và bộ phân loại (_classifier_) nên mô hình có tốc độ huấn luyện và dự báo chậm.

Thời kì tan băng của deep learning đã khiến mạng CNN phát triển mạnh mẽ. Những kiến trúc mạng CNN hiện đại ngày càng trở nên sâu hơn và đạt độ chính xác cao. Đây là những kiến trúc end-to-end cho phép các bộ trích lọc đặc trưng gắn liền với bộ phân loại trong một pipeline duy nhất. Các bộ trích lọc cũng không cần khởi tạo một cách thủ công mà trái lại chúng được sinh ngẫu nhiên theo các phân phối giả định. 

Nhờ các nguồn tài nguyên gồm các mô hình pretrained sẵn có mà bạn không cần phải tìm ra kiến trúc và huấn luyện mạng từ đầu. Thay vào đó, có thể tải xuống một mạng hiện đại đã được huấn luyện với trọng số từ các nguồn đã được công bố. Các nhà khoa học dữ liệu thường thực hiện điều chỉnh để thích ứng với các mạng này theo nhu cầu của họ bằng cách "tách" các lớp kết nối đầy đủ (fully connected layers) cuối cùng của mạng, thêm các lớp mới được thiết kế cho một nhiệm vụ cụ thể, và sau đó đào tạo mạng trên dữ liệu mới. Nếu nhiệm vụ của bạn chỉ là vector hóa hình ảnh, bạn chỉ cần loại bỏ các lớp cuối cùng và sử dụng kết quả đầu ra từ các lớp trước đó:

![](https://camo.githubusercontent.com/ee00962051042ac56919da91c4b9d3209e6cb4f0c6fb30e80dda67e229495fca/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f3830302f312a49775f634b46774c6b54564f325350724f5a553272512e706e67)

**Hình 4**: Đây là một mô hình phân lớp được huấn luyện trên một bộ dữ liệu từ trước hay còn gọi là mô hình pretrained. Lớp cuối cùng của mạng được tách ra và sử dụng để huấn luyện lại trên tập dữ liệu mới nhằm điều chỉnh để dự báo cho bộ dữ liệu mới.

Tuy nhiên, chúng ta sẽ không tập trung quá nhiều vào kỹ thuật mạng nơ ron. Thay vào đó các feature được tạo thủ công vẫn rất hữu ích: ví dụ đối với bài toán trong cuộc thi [Rental Listing Inquiries - Kaggle Competition](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries), để dự đoán mức độ phổ biến của danh sách cho thuê, ta có thể giả định rằng các căn hộ có ánh sáng sẽ thu hút nhiều sự chú ý hơn và tạo một feature mới như "giá trị trung bình của pixel".


+++ {"id": "dH2jSPgbGgXf"}

**Trích lọc thông tin văn bản trên hình ảnh**

_OCR_ (_Optical character recognition_) là dạng bài toán trích lọc thông tin văn bản trên hình ảnh. Chúng có tính ứng dụng cao và thường mang lại nhiều thông tin khi xử lý dữ liệu dạng hình ảnh.

Chằng hạn nếu có văn bản trên hình ảnh, bạn có thể đọc nó để khai thác một số thông tin thông qua gói phát hiện văn bản trong hình ảnh [pytesseract](https://github.com/madmaze/pytesseract).

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 87QtxenldT7P
outputId: 9ec3cfd1-f1ab-46b2-b78e-0b8a614b9a35
---
%%script echo skipping
!sudo apt-get install tesseract-ocr
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 231
id: 9wv8pEQSe3Kc
outputId: 4a48c28a-f4be-4fc6-8c9a-ba3a44402bbe
---
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

%matplotlib inline

##### Just a random picture from search
img = 'http://ohscurrent.org/wp-content/uploads/2015/09/domus-01-google.jpg'
img = requests.get(img)

img = Image.open(BytesIO(img.content))

# show image
img_arr = np.array(img)
plt.imshow(img_arr)
```

+++ {"id": "qJIo4JOOey4T"}

Đọc một hình ảnh thiết kế căn hộ thông qua link.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: JKp2iynQe6_z
outputId: 872788aa-29d8-474d-af61-e290e49a8f11
---
%%script echo skipping
import cv2
from PIL import Image
import pytesseract

img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
print(image_to_string(img_rgb))
```

+++ {"id": "xILpNQH_e-hy"}

## 11.2.3. Thông tin địa lý

Trong python chúng ta có một package khá phổ biến trong việc khai thác các thông tin địa lý đó là `reverse_geocoder`. Có 2 dạng bài toán chính với thông tin địa lý gồm 

* geocoding: mã hóa một tọa độ địa lý từ một địa chỉ.
* revert geocoding: từ thông tin cung cấp về kinh độ và vĩ độ trả về địa chỉ của địa điểm và các thông tin có liên quan. 

Cả hai bài toán đều có thể giải quyết thông qua API của google map hoặc OpenStreetMap. Sau đây là ví dụ trích xuất thông tin địa lý từ một địa điểm thông qua kinh độ và vĩ độ.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: TfY4whNIfIfy
outputId: 73721dad-0a81-46d3-9287-450354d5b817
---
%%script echo skipping
# install package reverse_geocoder
!pip install reverse_geocoder
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: SKdXGZysfL_D
outputId: 4ad51f5c-c596-47d6-95e8-39cca122fb48
---
import reverse_geocoder as revgc

# truyền vào latitude, longitude
revgc.search((21.0364466, 105.8450788))
```

+++ {"id": "Ri_y70oAfMtJ"}

Như chúng ta thấy, từ tọa độ có thể biết được căn hộ này nằm ở quận Hoàn Kiếm, Hà Nội, là một nơi phát triển và có mức sống cao. Như vậy mức giá của nó khả năng sẽ cao hơn. Từ quận và huyện ta xác định được căn hộ có nằm ở trung tâm hay không, các tiện nghi xung quan nó. Những thông tin trên rất quan trọng trong việc đánh giá khả năng bán được của căn hộ. Mặc dù trong bộ dữ liệu gốc không hề xuất hiện nhưng chúng có thể được trích xuất từ tọa độ địa lý.

+++ {"id": "kk0IiXDjfQC3"}

## 11.2.4. Dữ liệu thời gian

Trong dự báo, các dữ liệu thường có trạng thái thay đổi. Trạng thái của ngày hôm qua có thể khác biệt so với ngày hôm nay. Chẳng hạn như chiều cao, cân nặng của một người hay giá thị trường của các cổ phiếu. Chính vì thế thời gian là một thông tin có ảnh hưởng lớn tới biến mục tiêu. Từ một mốc thời gian biết trước chúng ta có thể phân rã thông tin thành giờ trong ngày, ngày trong tháng, tháng, quí, năm,.... Sẽ có rất nhiều điều thú vị được khám phá từ các thông tin này. Chẳng hạn như các qui luật của một số chuỗi số thay đổi theo mùa vụ: Nhiệt độ các tháng thay đổi theo mùa, GDP thay đổi theo qui luật quí, doanh số tiêu thụ kem thay đổi theo mùa,.... Yếu tố thời gian còn giúp xác định xu hướng biến đổi của một biến theo thời gian và kết hợp với tính mùa vụ sẽ trở thành một chỉ số quan trọng để ước lượng chuỗi thời gian.


+++ {"id": "OAgtuLhFfYvU"}

Biến đổi one-hot coding là một phương pháp quan trọng được sử dụng để mã hóa các biến chu kì thời gian. One-hot coding sẽ biến đổi một biến thành các vector có phần tử là 0 hoặc 1, trong đó 1 đại diện cho sự xuất hiện của đặc trưng và 0 đại diện cho các đặc trưng mà biến không có. 

Ví dụ: Chúng ta có 1 ngày trong tuần có thể rơi vào các thứ từ 2 đến chủ nhật. Như vậy một biểu diễn one-hot encoding của ngày thứ 2 sẽ là một véc tơ có phần tử đầu tiên bằng 1 và các phần tử còn lại bằng 0. Biểu diễn này cũng tương tự như với mã hóa dữ liệu văn bản thành các _sparse vector_. 

Trong python chúng ta có thể sử dụng hàm weekday() để xác định thứ tự của một ngày trong tuần. Thuộc tính weekday() chỉ tồn tại đối với dữ liệu dạng datetime. Do đó ta cần chuyển đổi các biến ngày đang ở dạng string về dạng datetime thông qua strftime (string format time). Bảng string format time có thể xem [tại đây](https://strftime.org/).

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 0FnARA-8fZke
outputId: 6a32c093-3e51-4d9a-9279-53fbf9678587
---
from datetime import datetime
import pandas as pd

dataset = pd.DataFrame({'created': ['2021-08-13 00:00:00', '2021-08-12 00:00:00', '2021-08-11 00:00:00', 
                                    '2021-08-10 00:00:00', '2021-08-09 00:00:00', '2021-08-08 00:00:00', '2021-08-07 00:00:00']})

def parser(x):
    # Để biết được định dạng strftime của một chuỗi kí tự ta phải tra trong bàng string format time
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

dataset['created'] = dataset['created'].map(lambda x: parser(x))
print(dataset['created'].dtypes)
```

+++ {"id": "UdtxxtipfkTe"}

Như vậy biến created đã được chuyển về dạng datetime. Chúng ta có thể tạo ra một one-hot encoding dựa vào hàm weekday().

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 48RptEozflP6
outputId: 2eb43832-3fb6-4fd2-90fe-ba4410d9ce11
---
dataset['weekday'] = dataset['created'].apply(lambda x: x.date().weekday())
dataset['weekday']
```

+++ {"id": "bv6El19Lfsmd"}

Ta có thể tạo ra một biến trả về trạng thái ngày có phải là cuối tuần bằng kiểm tra weekday() có rơi vào [5, 6] là những ngày cuối tuần hay không.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Wd_YGJIyftef
outputId: ee017337-3c51-4fe3-b6d4-89bb5f5fc2d3
---
dataset['is_weekend'] = dataset['created'].apply(lambda x: 1 if x.date().weekday() in [5, 6] else 0)
dataset['is_weekend']
```

+++ {"id": "ZDSyPntUfxdJ"}

Trong một số bài toán dữ liệu có thể bị phụ thuộc vào thời gian. Chẳng hạn như lịch trả nợ của thẻ tín dụng sẽ rơi vào kì sao kê là một ngày cụ thể trong tháng. Khi làm việc với dữ liệu chuỗi thời gian chúng ta nên lưu ý tới danh sách các ngày đặc biệt trong năm như nghỉ tết âm lịch, quốc khánh, quốc tế lao động,.... Bởi những ngày này thường sẽ có biến động lớn về dữ liệu kinh doanh.

+++ {"id": "70xnMGYzd84V"}

## 11.2.5. Dữ liệu từ website, log

Các hệ thống website lớn sẽ tracking lại các session của người dùng. Những thông tin được tracking bao gồm thông tin thiết bị, loại event, customer ID, ... Từ customer ID chúng có thể link tới database người dùng để biết được các thông tin về giới tính, độ tuổi, tài khoản, hành vi giao dịch,.... Trong một số trường hợp một khách hàng có thể thay đổi thiết bị truy cập, do đó không phải hầu hết các trường hợp chúng ta đều map được session với Customer ID trên dữ liệu local. Tuy nhiên từ các thông tin được lưu trong Cookie về người dùng (còn gọi là user agent) cũng cung cấp cho chúng ta khá nhiều điều. Chẳng hạn như: Thiết bị truy cập, trình duyệt, hệ điều hành,... Từ thiết bị di động chúng ta cũng ước đoán được người dùng có mức thu nhập như thế nào: Sử dụng Iphone X thì khả năng cao là người có thu nhập cao, sử dụng điện thoại xiaomi khả năng là người thu nhập trung bình và thấp,.... Để phân loại các thông tin về người dùng chúng ta có thể sử dụng package user_agents trong python.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: aT8GEuZZgfFY
outputId: cf2ebf28-bcee-4f1c-eb5c-62e67e1e52db
---
%%script echo skipping
!pip install user_agents
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 1JZf9vrLgops
outputId: 749c259d-a7e1-4aca-be5c-111150647253
---
import user_agents
# Giả định có một user agent như bên dưới
ua = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/56.0.2924.76 Chrome/56.0.2924.76 Safari/537.36'
# Parser thông tin user agent
ua = user_agents.parse(ua)
# Khai thác các thuộc tính của user
print('Is a bot? ', ua.is_bot)
print('Is mobile? ', ua.is_mobile)
print('Is PC? ',ua.is_pc)
print('OS Family: ',ua.os.family)
print('OS Version: ',ua.os.version)
print('Browser Family: ',ua.browser.family)
print('Browser Version: ',ua.browser.version)
```

+++ {"id": "TlsGsnYXgutV"}

# 11.3. Biến đổi đặc trưng (feature transformation)

Các chiều dữ liệu thường có sự khác biệt về đơn vị (_scale_) và phân phối (_distribution_) và điều đó gây ảnh hưởng tới mô hình ở những khía cạnh sau:

* Khi huấn luyện thường dẫn tới hiện tượng _bùng nổ gradient_ (_exploding gradient_). Hiện tượng _bùng nổ gradient_ là một hiện tượng phổ biến khiến cho giá trị dự báo bị quá lớn (thường là giá trị _nan_).

* Không chuẩn hoá dữ liệu đầu vào có thể khiến cho quá trình huấn luyện thiếu ổn định và có thể không vượt qua được các _điểm cực trị địa phương_ để đi tới _cực trị toàn cục_.

* Giá trị dự báo của biến mục tiêu trở nên nhạy cảm hơn. Đối với những biến có đơn vị lớn và trọng số huấn luyện lớn thì một sự thay đổi nhỏ về giá trị của biến sẽ dẫn tới sự thay đổi lớn của giá trị dự báo.

Trong quá trình huấn luyện mô hình, sử dụng các phép biến đổi dữ liệu sẽ luôn giúp ích cho mô hình huấn luyện. Xin trích dẫn:

"In practice it is nearly always advantageous to apply pre-processing transformations to the input data before it is presented to a network. Similarly, the outputs of the network are often post-processed to give the required output values."

Page 296, [Neural Networks for Pattern Recognition](https://www.amazon.com/Networks-Recognition-Advanced-Econometrics-Paperback/dp/0198538642/ref=as_li_ss_tl?ie=UTF8&qid=1540160671&sr=8-2&keywords=Neural+Networks+for+Pattern+Recognition&linkCode=sl1&tag=inspiredalgor-20&linkId=991aca4ff0fc6769d5dad40a86092458&language=en_US), 1995.

Có hai phương pháp chính để biến đổi dữ liệu đó là _scaling_ và _chuẩn hoá_ (_standardize_) mà ta sẽ tìm hiểu bên dưới.

+++ {"id": "C4gHiPeHo_f_"}

## 11.3.1. Chuẩn hoá (_standardization_)

Kĩ thuật _chuẩn hoá_ được áp dụng đối với những biến không có phân phối chuẩn. Biến được biến đổi theo kì vọng và độ lệch chuẩn như sau:

$$\mathbf{x}' = \frac{\mathbf{x}-\overline{\mathbf{x}}}{\sigma(\mathbf{x})}$$

Từ đó suy ra giá trị của biến sau khi biến đổi ngược lại:

$$\mathbf{x} = \mathbf{x}'*\sigma(\mathbf{x})+\bar{\mathbf{x}}$$

Các biến sau khi được chuẩn hoá sẽ có cùng một dạng phân phối chuẩn hoá với trung bình bằng 0 và phương sai bằng 1. Nhờ đó quá trình huấn luyện sẽ trở nên ổn định và hội tụ tới nghiệm tối ưu nhanh hơn.

![](https://i.imgur.com/Kql3MEH.jpeg)

**Hình 5**: Phương pháp _chuẩn hoá_ (_standardization_). Sau chuẩn hoá biến có phần phối chuẩn với trung bình bằng 0 và phương sai bằng 1. 

Ngoài ra ta còn chứng minh được rằng phương pháp _chuẩn hoá_ còn là một phép co trong không gian mà ở đó khoảng cách giữa 2 điểm bất kì luôn cùng một tỷ lệ so với không gian gốc.

Thật vậy. Giả sử ta xét hai điểm là $\mathbf{x}_1$ và $\mathbf{x}_2$ trong không gian gốc. Toạ độ hai điểm này sau khi MinMax Scaling lần lượt là $\mathbf{x}_1'$ và $\mathbf{x}_2'$. Chúng có mối liên hệ với giá trị gốc theo phương trình:

$$\begin{eqnarray}\mathbf{x}_1 & = & \mathbf{x}_{1}' * \sigma(\mathbf{x}) + \bar{\mathbf{x}}  \\
\mathbf{x}_2 & = & \mathbf{x}_{2}' * \sigma(\mathbf{x}) + \bar{\mathbf{x}}
\end{eqnarray}$$

Suy ra:

$$\frac{\mathbf{x}_1-\mathbf{x}_2}{\mathbf{x}_1' - \mathbf{x}_2'} = \sigma(\mathbf{x}) = \alpha$$

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 390
id: zowjQcL5qcvj
outputId: f75db35c-2d03-44c4-e66e-3f99b27d4ed9
---
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Khởi tạo một biến X ngẫu nhiên
X = np.random.randn(1000, 1)*6 + 5
# Standardization
X_std = StandardScaler().fit_transform(X.reshape(-1, 1))

def _plot_dist(x, bins=10, xlim=(-1, 1), varname='x'):
  sns.histplot(x, bins = bins, kde = True)
  plt.title('histogram of {}'.format(varname))
  plt.xlim(xlim)
  plt.legend([varname])


# Visualization
fig = plt.figure(figsize=(20, 6))

ax_1 = fig.add_subplot(1, 2, 1)
ax_1 = _plot_dist(X, bins=10, xlim=(-15, 25), varname='original data')


ax_2 = fig.add_subplot(1, 2, 2)
ax_2 = _plot_dist(X_std, bins=10, xlim=(-15, 25), varname='standardized data')
```

+++ {"id": "mpLXY8pqz5IM"}

**Hình 5:** Biến bên trái chưa được chuẩn hoá, khoảng biến thiên của biến trong khoảng từ -15 đến 20 và trung bình là 5. Sau khi chuẩn hoá ta thu được biến bên phải có phân phối chuẩn với trung bình là 0 và khoảng biến thiên từ -2 tới 2. Như vậy chuẩn hoá theo phân phối chuẩn đã giúp giảm độ lớn trung bình và phương sai của biến.

+++ {"id": "R1YjFZ8to9W0"}

## 11.3.2. Kĩ thuật scaling

Đối với kĩ thuật _scaling_ thì chúng ta thường áp dụng trên những biến đã tuân theo phân phối chuẩn. Thông qua _scaling_, toàn bộ giá trị của biến sẽ được đưa về một miền giá trị bị giới hạn trong khoảng $[0, 1]$. Khi đó đối với các điểm outliers xuất hiện trên tập _kiểm tra_ (_test dataset_) thì giá trị của chúng có thể nằm ngoài miền này (tức tồn tại giá trị lớn hơn 1 và nhỏ hơn 0). Dựa trên kĩ thuật _scaling_ chúng ta sẽ cân nhắc thiết lập lại outliers về các điểm đầu mút của miền giới hạn. 

Trong kĩ thuật scaling thì chúng ta có các phương pháp chính:


+++ {"id": "5m4fBTJSSDJ4"}

### 11.3.2.1. Minmax Scaling

Biến được đưa về các range [0,1] theo công thức:

$$\mathbf{x'} = \frac{\mathbf{x}-\min({\mathbf{x})}}{\max({\mathbf{x}}) - \min({\mathbf{x}})}$$

Như vậy giá trị của biến sau khi biến đổi ngược lại từ scaling sẽ là:

$$\mathbf{x} = \mathbf{x}' * (\max(\mathbf{x}) - \min(\mathbf{x})) + \min(\mathbf{x})$$

+++ {"id": "d0fpDGvPJEp1"}

MinMax Scaling còn là một phép co trong không gian mà ở đó tỷ lệ khoảng cách giữa 2 điểm bất kì được bảo toàn so với khoảng cách của chúng trong không gian gốc.

Thật vậy. Giả sử ta xét hai điểm là $\mathbf{x}_1$ và $\mathbf{x}_2$ trong không gian gốc. Toạ độ hai điểm này sau khi MinMax Scaling lần lượt là $\mathbf{x}_1'$ và $\mathbf{x}_2'$. Chúng có mối liên hệ với giá trị gốc theo phương trình:

$$\begin{eqnarray}\mathbf{x}_1 & = & \mathbf{x}_{1}' * (\max(\mathbf{x}) - \min(\mathbf{x})) + \min(\mathbf{x}) \\
\mathbf{x}_2 & = & \mathbf{x}_{2}' * (\max(\mathbf{x}) - \min(\mathbf{x})) + \min(\mathbf{x}) 
\end{eqnarray}$$

Suy ra:

$$\frac{\mathbf{x}_1-\mathbf{x}_2}{\mathbf{x}_1' - \mathbf{x}_2'} = \max(\mathbf{x}) - \min(\mathbf{x}) = \alpha$$

![](https://i.imgur.com/Txn9nAD.jpeg)

**Hình 6**: Hình minh hoạ phương pháp MinMax Scaling.

+++ {"id": "V_WNdwh7Qwtz"}

Trên sklearn để thực hiện MinMaxScaler như sau:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 390
id: LA3_MiojQ0eh
outputId: 38c72788-dd70-471e-8ab8-3cca164e0957
---
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# Khởi tạo một biến X ngẫu nhiên
X = np.random.randn(1000, 1)+5
# minmax scaler của X
X_minmax = MinMaxScaler().fit_transform(X.reshape(-1, 1))

def _plot_dist(x, bins=10, xlim=(-1, 1), varname='x'):
  sns.histplot(x, bins = bins, kde = True)
  plt.title('histogram of {}'.format(varname))
  plt.xlim(xlim)
  plt.legend([varname])


# Visualization
fig = plt.figure(figsize=(20, 6))

ax_1 = fig.add_subplot(1, 2, 1)
ax_1 = _plot_dist(X, bins=10, xlim=(0, 10), varname='original data')

ax_2 = fig.add_subplot(1, 2, 2)
ax_2 = _plot_dist(X_minmax, bins=10, xlim=(0, 10), varname='minmax scaling data')
```

+++ {"id": "HdyQszSzRkiS"}

**Hình 7**: Bên trái là biến gốc có giá trị nằm trong khoảng từ 2 đến 8. Sau khi chuẩn hoá MinMax Scaling thì giá trị của biến thu hẹp về miền $[0, 1]$.

+++ {"id": "LmR56TZ-RzfA"}

### 11.3.2.2. Unit Length

Theo phương pháp này giá trị của biến sẽ được chuẩn hoá bằng cách chia cho norm chuẩn $L_2$.

$$\mathbf{x'} = \frac{\mathbf{x}}{||\mathbf{x}||_2}$$

Khi đó trong không gian Euclidean thì biến $\mathbf{x}'$ sẽ là một véc tơ có độ dài là 1 đơn vị. Chính vì vậy phương pháp chuẩn hoá này còn gọi là chuẩn hoá _độ dài đơn vị_ (_Unit Length_). Ta cũng dễ dàng nhận thấy các giá trị sau khi chuẩn hoá theo _Unit Length_ sẽ nằm trong khoảng $[0, 1]$. So với các phương pháp chuẩn hoá khác như _Standardization_ và _MinMax Scaling_ thì chuẩn hoá theo _Unit Length_ thường có mức độ thu hẹp độ biến động của biến là lớn nhất. Dường như biến sẽ bị co cụm về một miền giá trị rất nhỏ gần 0. Chúng ta có thể kiểm chứng điều này thông qua ví dụ về chuẩn hoá _Unit Length_ trên sklearn bên dưới:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 390
id: FDpOm5NtUToN
outputId: 7e877604-1f83-4396-cae7-27e3a7b66b3f
---
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Khởi tạo một biến X ngẫu nhiên
X = np.random.randn(1000, 1)+5
# Unit Length scaling
X_un = X/np.linalg.norm(X)

def _plot_dist(x, bins=10, xlim=(-1, 1), varname='x'):
  sns.histplot(x, bins = bins, kde = True)
  plt.title('histogram of {}'.format(varname))
  plt.xlim(xlim)
  plt.legend([varname])

# Visualization
fig = plt.figure(figsize=(20, 6))

ax_1 = fig.add_subplot(1, 2, 1)
ax_1 = _plot_dist(X, bins=10, xlim=(0, 10), varname='original data')

ax_2 = fig.add_subplot(1, 2, 2)
ax_2 = _plot_dist(X_un, bins=10, xlim=(0, 1), varname='unit length scaling data')
```

+++ {"id": "e96UuYyepxdu"}

**Hình 8**: Bên trái là phân phối của biến gốc và bên phải là phân phối của biến sau khi thực hiện _Unit Length_ scaling. So với các phương pháp phân phối khác thì _Unit Length_ trả về giá trị có khoảng biến thiên hẹp hơn và gần sát với 0.

+++ {"id": "_5DZqyrcXQAK"}

### 11.3.2.3. Robust Scaling

Trong trường hợp dữ liệu tồn tại outliers thì các phương pháp chuẩn hoá dựa trên _Standardization_, _MinMax Scaling_ sẽ thường không mang lại hiệu quả. Sự xuất hiện của các outliers thường nằm ở rìa phân phối của biến và chúng có xác suất xảy ra thấp. Điều đó khiến cho phân phối bị lệch sang một bên (hiện tượng _skewness_ cao) và sử dụng các phương pháp chuẩn hoá thông thường như _Standardization_ trở nên khó khăn hơn do bản thân trung bình và độ lệch chuẩn được tính ra cũng đã bị méo bởi sự xuất hiện của những điểm outliers.

Một trong những cách tiếp cận để chuẩn hoá dữ liệu khi xuất hiện outliers đó là loại bỏ outliers khỏi tính toán trung bình và độ lệch chuẩn, sau đó sử dụng những giá trị được tính toán để _scaling_ biến.

Phương pháp này chính là _Robust Scaler_. Chúng được thực hiện bằng cách tính toán các khoảng phân vị trung vị $Q_2$ (50% percentile), và các khoảng phân vị $Q_1$ (25% percentile), phân vị $Q_3$ (75% percentile). Giá trị của một biến sẽ được trừ đi trung vị $Q_2$ và sau đó chia cho độ dài _khoảng liên phân vị_ _(interquartile range - IQR_,$~\text{IQR} = Q_3 - Q_1$). Công thức như sau:

$$\mathbf{x}' = \frac{\mathbf{x}-Q_2(\mathbf{x})}{Q_3(\mathbf{x})-Q_1(\mathbf{x})}$$

Sử dụng _Robust Scaler_ có thể giúp loại bỏ các _outliers_ và sau đó chúng ta có thể tiếp tục thực hiện các phương pháp chuẩn hoá khác sau đó như _Standardization_, _MinMax Scaling_.


Phương pháp _Robust Scaler_ được phát triển trong sklearn thông qua class RobustScaler:

```
sklearn.preprocessing.RobustScaler(
with_centering=True,
with_scaling=True,
quantile_range=(25.0, 75.0),
copy=True,
)
```

Trong đó:

* _with_centering_: Quyết định xem giá trị có được chuẩn hoá bằng cách trừ đi $Q_2$. Mặc định là True.

* _with_scaling_: Có thực hiện scale bằng cách chia cho $\text{IQR}$ hay không ? Mặc định được thiết lập là True.

Chúng ta có thể thay đổi độ dài khoảng $\text{IQR}$ thông qua thay đổi giá trị của đối số _quantile_range_. Nó nhận giá trị là một khoảng là tập con của $[0, 100]$. Thay đổi giá trị này sẽ thay đổi định nghĩa về outlier và độ lớn scaling.

Tiếp theo chúng ta sẽ phân tích kĩ hơn về _Robust Scaler_ thông qua ví dụ bên dưới.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 350
id: cLwu5OIChfTz
outputId: 469cf5c9-7fb8-4bbf-f7d0-be0cafcf8710
---
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
  
# data
x = pd.DataFrame({
    # Distribution with lower outliers
    'x1': np.concatenate([np.random.normal(20, 2, 1000), np.random.normal(-100, 2, 25)]),
    # Distribution with higher outliers
    'x2': np.concatenate([np.random.normal(30, 2, 1000), np.random.normal(90, 2, 25)]),
})
  
# robust scaler  
scaler = RobustScaler()
robust_df = scaler.fit_transform(x)
robust_df = pd.DataFrame(robust_df, columns =['x1', 'x2'])

# standard scaler
scaler = StandardScaler()
standard_df = scaler.fit_transform(x)
standard_df = pd.DataFrame(standard_df, columns =['x1', 'x2'])

# minmax scaler
scaler = MinMaxScaler()
minmax_df = scaler.fit_transform(x)
minmax_df = pd.DataFrame(minmax_df, columns =['x1', 'x2'])

# visualization
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols = 4, figsize =(20, 5))

ax1.set_title('Before Scaling')
ax1.set_xlabel('x')
sns.kdeplot(x['x1'], ax = ax1, color ='r', linewidth=2)
sns.kdeplot(x['x2'], ax = ax1, color ='b', linewidth=2)
ax1.legend(['x1', 'x2'])

ax2.set_title('After Robust Scaling')
ax2.set_xlabel('x') 
sns.kdeplot(robust_df['x1'], ax = ax2, color ='r', linewidth=2)
sns.kdeplot(robust_df['x2'], ax = ax2, color ='b', linewidth=2)
ax2.legend(['x1', 'x2'])

ax3.set_title('After Standard Scaling')
ax3.set_xlabel('x')
sns.kdeplot(standard_df['x1'], ax = ax3, color ='r', linewidth=2)
sns.kdeplot(standard_df['x2'], ax = ax3, color ='b' , linewidth=2)
ax3.legend(['x1', 'x2'])

ax4.set_title('After MinMax Scaling')
ax4.set_xlabel('x')
sns.kdeplot(minmax_df['x1'], ax = ax4, color ='r', linewidth=2)
sns.kdeplot(minmax_df['x2'], ax = ax4, color ='b', linewidth=2)
ax4.legend(['x1', 'x2'])

plt.show()
```

+++ {"id": "3z_qLE5iqUfC"}

**Hình 9**: Đồ thị so sánh các phương pháp chuẩn hoá khác nhau là _Robust Scaling_, _Standardization_ và _MinMax Scaling_. Biến $\mathbf{x}_1$ xuất hiện outliers ở các gía trị thấp trong khi biến $\mathbf{x}_2$ xuất hiện outliers ở các giá trị cao. Chúng ta có thể thấy phân phối của biến bị lệch hẳn sang một bên đối với phương pháp _MinMax Scaling_.  Phương pháp _Standardization_ thì do ảnh hưởng của outliers nên các phân phối sẽ bị lệch trái hoặc lệch phải chứ không hoàn toàn đối xứng qua điểm 0. Trong khi đó _Robust Scaling_ do đã loại bỏ được ảnh hưởng của outliers nên trả về kết quả phân phối đối xứng qua 0.

+++ {"id": "F1OozWpUijnX"}

# 11.4. Lựa chọn đặc trưng (_feature selection_)

Để xây dựng mô hình chúng ta sẽ rất cần đến dữ liệu lớn. Nhưng dữ liệu quá lớn cũng không thực sự tốt. Những hệ thống của các tập đoàn công nghệ lớn có thể có số lượng trường dữ liệu lên tới hàng trăm ngàn. Đây là một con số khổng lồ và sẽ gây ra những hạn chế đó là:

* Tăng chi phí tính toán.
* Quá nhiều biến giải thích có thể dẫn tới _quá khớp_ (_overfiting_). Tức hiện tượng mô hình hoạt động tốt trên _tập huấn luyện_ nhưng kém trên _tập kiểm tra_.
* Trong số các biến sẽ có những biến gây nhiễu và làm giảm chất lượng mô hình.
* Rối loạn thông tin do không thể kiểm soát và hiểu hết các biến.

Chính vì thế chúng ta cần phải có những phương pháp như giảm chiều dữ liệu hoặc lựa chọn biến quan trọng. Về phương pháp giảm chiều dữ liệu sẽ được trình bày ở một chương khác. Trong chương này này chúng ta sẽ làm quen với một số kĩ thuật lựa chọn biến thông dụng.

Bên dưới là những thuật toán quan trọng được sử dụng để lựa chọn các biến.

+++ {"id": "mN_b5BbxiwX9"}

## 11.4.1. Phương pháp thống kê

Một phương pháp quan trọng trong các phương pháp thống kê nhằm giảm số lượng biến là lựa chọn dựa trên phương sai. Dựa trên phân tích các biến không biến động thì không có tác dụng gì trong việc phân loại hoặc dự báo bởi chúng ta dường như đã biết được giá trị của chúng cho tất cả các quan sát. Do đó ý tưởng chính của phương pháp này là thông qua độ lớn phương sai của toàn bộ các _biến numeric_ để loại bỏ những biến nếu nó nhỏ hơn một ngưỡi nhất định.

Trong sklearn chúng ta có thể sử dụng _VarianceThreshold_ để lọc bỏ biến theo phương sai.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: N7PfY5cBi0om
outputId: 46f47c8a-5699-49e2-abd8-bab4124b240d
---
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import make_classification

# Khởi toạo dữ liệu example
X, y = make_classification(n_samples=500, n_features=50, random_state=123)

print('X shape:', X.shape)
print('y shape:', y.shape)

# Lọc bỏ các biến có phương sai nhỏ hơn 0.8
print('Total features with thres=0.8: {}'.format(VarianceThreshold(0.8).fit_transform(X).shape))

# Lọc bỏ các biến có phương sai nhỏ hơn 1.0
X_kvar = VarianceThreshold(0.9).fit_transform(X)
print('Total features with thres=1.0: {}'.format(X_kvar.shape))
```

+++ {"id": "7_vWaDdojCpW"}

Ngoài phương pháp phương sai, chúng ta có thể áp dụng [kiểm định thống kê đơn biến](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection). Phương pháp này sẽ đánh giá sự độc lập tuyến tính giữa hai biến ngẫu nhiên dựa trên phân phối _chi-squared_ và _Fisher_ để lựa chọn ra $k$ biến tốt nhất. Để hình dung kĩ hơn về hai phương pháp thống kê nêu trên, tiếp theo chúng ta cùng thực hành lựa chọn biến và đánh giá hiệu quả mô hình.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: OO5G-WepjOcg
outputId: d88a3d74-1d59-4110-a32a-d40479b84380
---
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Lựa chọn biến dựa trên phương pháp Fisher
X_kbest = SelectKBest(f_classif, k = 5).fit_transform(X, y)
print('X shape after applying statistical selection: ',X_kbest.shape)
```

+++ {"id": "r9DUMtqUjQpW"}

Chúng ta sẽ cùng đánh giá hiệu quả mô hình bằng cross-validation trước và sau lựa chọn biến với KFold = 5.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: HwYXqLlWjRIQ
outputId: cbb8418c-8386-48fe-b610-91b05f585887
---
# Hồi qui logistic
logit = LogisticRegression(solver='lbfgs', random_state=1)

# Cross validation cho:
# 1.dữ liệu gốc
acc_org = cross_val_score(logit, X, y, scoring = 'accuracy', cv = 5).mean()
# 2. Áp dụng phương sai
acc_var = cross_val_score(logit, X_kvar, y, scoring = 'accuracy', cv = 5).mean()
# 3. Áp dụng phương pháp thống kê
acc_stat = cross_val_score(logit, X_kbest, y, scoring = 'accuracy', cv = 5).mean()

print('Accuracy trên dữ liệu gốc:', acc_org)
print('Accuracy áp dụng phương sai:', acc_var)
print('Accuracy dụng pp thống kê:', acc_stat)
```

+++ {"id": "XQaRtczHy3fT"}

Như vậy ta thấy sau khi áp dụng feature selection đã cải thiện được độ chính xác của mô hình dự báo.


+++ {"id": "d9yQ02fvjWAa"}

## 11.4.2. Sử dụng mô hình

Đây là phương pháp rất thường xuyên được áp dụng trong các cuộc thi phân tích dữ liệu. Chúng ta sẽ dựa trên một số mô hình cơ sở để đánh giá mức độ quan trọng của các biến. Có hai lớp mô hình thường được sử dụng để đánh biến đó là _Random Forest_ và _Linear Regression_. Ưu điểm của các phương pháp này là kết quả đánh giá rất chuẩn xác, tuy nhiên nhược điểm của chúng là phải xây dựng mô hình hồi qui rồi mới xác định được biến quan trọng. Điều này dường như đi trái lại với thực tế phải lựa chọn biến trước khi huấn luyện mô hình. Để áp dụng phương pháp này chúng ta thực hiện như sau:


```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: a8QDK3nRjdVC
outputId: 05f1cded-cb35-4451-e034-9e9fea4e47de
---
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# Hồi qui theo RandomForest
rdFrt = RandomForestClassifier(n_estimators = 20, random_state = 1)
# Hồi qui theo LinearSVC
lnSVC = LinearSVC(C=0.01, penalty="l1", dual=False)
# Hồi qui theo Lasso
lassoReg = Lasso(alpha = 1.0)
# Tạo một pipeline thực hiện lựa chọn biến từ RandomForest model và hồi qui theo logit
pipe1 = make_pipeline(StandardScaler(), SelectFromModel(estimator = rdFrt), logit)
# Tạo một pipeline thực hiện lựa chọn biến từ Linear SVC model và hồi qui theo logit
pipe2 = make_pipeline(StandardScaler(), SelectFromModel(estimator = lnSVC), logit)

# Cross validate đối với 
# 1. Mô hình logit
acc_log = cross_val_score(logit, X, y, scoring = 'accuracy', cv = 5).mean()
# 2. Mô hình RandomForest
acc_rdf = cross_val_score(rdFrt, X, y, scoring = 'accuracy', cv = 5).mean()
# 3. Mô hình pipe1
acc_pip1 = cross_val_score(pipe1, X, y, scoring = 'accuracy', cv = 5).mean()
# 3. Mô hình pipe2
acc_pip2 = cross_val_score(pipe2, X, y, scoring = 'accuracy', cv = 5).mean()

print('Accuracy theo logit:', acc_log)
print('Accuracy theo random forest:', acc_rdf)
print('Accuracy theo pipeline 1:', acc_pip1)
print('Accuracy theo pipeline 2:', acc_pip2)
```

+++ {"id": "w8zLfGzEy2UP"}

Như vậy select dựa trên mô hình Random Forest và Linear SVC đã có hiệu quả trong việc cải thiện độ chính xác của mô hình. Bên cạnh việc thực hiện lựa chọn biến dựa trên model, chúng ta còn có thể lựa chọn biến theo grid search.

+++ {"id": "idYeCz6ajis9"}

## 11.4.3. Sử dụng Search

**Exhaustive Search**

Ý tưởng chính của phương pháp này là tìm ra một tập con các đặc trưng tốt nhất trong số các đặc trưng đầu vào dựa trên một thước đo mô hình cụ thể (chẳng hạn như _accuracy_). Ví dụ, khi bạn có tổng cộng $n$ đặc trưng thì bạn cần huấn luyện mô hình trên tất cả các kết hợp từ $1, 2, 3, ..., n$ đặc trưng. Tổng số lượng các kết hợp có thể sẽ là:

$$\binom{n}{1} + \binom{n}{2} + \binom{n}{3} + \dots + \binom{n}{n} = 2^{n} - 1$$

Đây là số lượng không hề nhỏ nếu bộ dữ liệu của bạn có số lượng đặc trưng lớn. Chính vì thế phương pháp này được coi là _Exhaustive_ và chỉ phù hợp với những bộ dữ liệu có số lượng đặc trưng nhỏ. Ưu điểm của phương pháp này mang lại đó là giúp tìm ra được tập con đặc trưng tốt nhất trực tiếp thông qua đánh giá _accuracy_.
+++ {"id": "ZZfRpw7Mjsp9"}

**Sequential Feature Selection**

Nếu như chúng ta tìm kiếm trên toàn bộ các bộ kết hợp đặc trưng đầu vào của mô hình sẽ rất lâu. Do đó việc đầu tiên ta cần thực hiện là giới hạn không gian tìm kiếm. Tuỳ theo hướng tìm kiếm là tăng biến hoặc giảm biến mà phương pháp này bao gồm hai hai lựa chọn là: forward hoặc backward tương ứng. Theo lựa chọn forward thì ban đầu ta xuất phát từ lựa chọn $1$ đặc trưng đầu vào mà mô hình có kết quả tốt nhất. Ở các bước tiếp theo ta sẽ tìm ra một đặc trưng phù hợp nhất để thêm vào mô hình sao cho thước đo đánh giá mô hình là lớn nhất. Quá trình này tiếp tục cho đến khi số lượng các đặc trưng được thêm vào đạt mức tối đa `k_features` hoặc tới khi hàm loss fuction mô hình không giảm nữa. Theo chiều ngược lại, bắt đầu từ toàn bộ các đặc trưng và loại dần đặc trưng thì sẽ là backward. 

So với phương pháp _Exhaustive Search_ thì _Sequential Feature Selection_ ít tốn kém hơn về chi phí nhưng không đảm bảo chắc chắn rằng tập hợp đặc trưng tìm được là tối ưu. Hướng di chuyển tìm kiếm theo forward và backward cũng hoàn toàn là lựa chọn may rủi.

Bên dưới ta sẽ tiến hành áp dụng phương pháp _Sequential Feature Selection_ để tìm kiếm đặc trưng theo backward với số biến cần lựa chọn là `k_features=3`.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: HqZcFnnij1o-
outputId: 13c7c2f1-7eae-4aa6-843d-0505963acccf
---
%%script echo skipping
!pip install mlxtend
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: iuzSaah3j1vu
outputId: b987ca78-f8f0-40c3-beb3-c165241c8d65
---
from mlxtend.feature_selection import SequentialFeatureSelector

selector = SequentialFeatureSelector(logit, scoring = 'accuracy', 
                                     verbose = 2, 
                                     k_features = 3,
                                     forward = False,
                                     n_jobs = -1)

selector.fit(X, y)
```

+++ {"id": "MyTNLolmjzBm"}

Ta có thể thấy mô hình xuất phát từ 50 biến ban đầu và sau mỗi một quá trình sẽ loại dần các biến cho đến khi số lượng biến tối thiểu đạt được là 3. Sau mỗi quá trình mức độ accuracy sẽ tăng dần.

+++ {"id": "ijgXjKN0j8p-"}

# 11.5. Tổng kết

Như vậy sau bài này các bạn đã nhận ra được _Feature Engineering_ quan trọng như thế nào trong việc tạo ra một mô hình dự báo có sức mạnh. Tổng hợp lại các phương pháp feature engineering:

1. Trích lọc đặc trưng: Ứng dụng trong deep learning như xử lý ảnh và xử lý ngôn ngữ tự nhiên, phân rã thời gian, làm việc với dữ liệu địa lý, dữ liệu người dùng tracking từ các hệ thống web, app.
2. Biến đổi đặc trưng: Minmax scaling, Unit length scaling, Standardization, Robust Scaling.
3. Lựa chọn đặc trưng: Sử dụng phương pháp thống kê, mô hình hoặc grid search.

Câu hỏi đặt ra:

> Bên cạnh những thuật toán, modeler có cần kiến thức về lĩnh vực chuyên ngành (_knowledge domain_) không?

Để xây dựng một mô hình tốt không chỉ cần có kiến thức về mô hình mà các hiểu biết về lĩnh vực chuyên ngành cũng rất quan trọng. Khi hiểu rõ về lĩnh vực, modeler sẽ nắm rõ bản chất mối quan hệ của các biến không chỉ qua các con số mà còn trên các khía cạnh business thực tiễn. Đó cũng là lý do trong một dự án phân tích dữ liệu luôn cần sự tư vấn từ BA và các chuyên gia trong ngành để giúp modeler hiểu sâu hơn các qui luật tiềm ẩn bên trong dữ liệu đang hoạt động thế nào.

> Trong mọi mô hình có nên thực hiện Feature Engineering?

Hầu hết các mô hình hiện đại đều thực hiện _Feature Engineering_ trước khi huấn luyện mô hình bởi sau khi thực hiện _Feature Engineering_ chúng ta sẽ có cơ hội tạo ra một mô hình mạnh hơn. Cần so sánh nhiều mô hình khác nhau để lựa chọn ra đâu là mô hình phù hợp nhất, trong một số trường hợp có thể sử dụng kết hợp các mô hình.

> Ý tưởng về _Feature Engineering_ rất nhiều? Làm thế nào để tìm ra một _Feature Engineering_ tối ưu?

Không có câu trả lời cụ thể cho một phương pháp _Feature Engineering_ nào là tối ưu. Chỉ có quá trình thử và sai để rút ra được phương pháp _Feature Engineering_ nào sẽ phù hợp với bài toán cụ thể nào.

+++ {"id": "wgnv51Y7kBp3"}

# 11.6. Tài liệu tham khảo

1. [Giới thiệu về feature engineering - mlcourse.ai](https://mlcourse.ai/notebooks/blob/master/jupyter_english/topic06_features_regression/topic6_feature_engineering_feature_selection.ipynb?flush_cache=true)

2. [feature engineering - blog machinelearningcoban - Vu Huu Tiep](https://machinelearningcoban.com/general/2017/02/06/featureengineering/)

3. [tfidf - Information retrieval - wiki](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

4. [package pytessaract - ứng dụng trong OCR - blog pyimagesearch](https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/)

5. [extract time in python - blog hamelg](http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-17.html)

6. [feature scaling - rpub phamdinhkhanh](http://rpubs.com/phamdinhkhanh/398690)

7. [feature scaling - arsenyinfo](https://www.kaggle.com/arsenyinfo/easy-feature-selection-pipeline-0-55-at-lb)

8. [Type of feature transformation and scaling - analyticsvidhya](https://www.analyticsvidhya.com/blog/2020/07/types-of-feature-transformation-and-scaling/)

9. [Probabilistic model selection measures](https://machinelearningmastery.com/probabilistic-model-selection-measures/)

10. [Sklearn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

# 11.7. Bài tập

1-. Trong phương pháp bag-of-word thì mỗi một đoạn văn bản sẽ được biến đổi thành véc tơ đặc trưng như thế nào?

2-. Phương pháp bag-of-ngram với bigram và trigram sẽ mã hoá một văn bản như thế nào? Số lượng các từ trong từ điển của phương pháp bag-of-ngram sẽ lớn hơn hay nhỏ hơn so với bag-of-word?

3-. Giải thích ý nghĩa của chỉ số tf-idf được sử dụng để mã hoá các từ trong bộ văn bản. Một từ có tf-idf cao thì chứng tỏ điều gì?


4-. Thực hành phân loại văn bản dựa trên phương pháp bag-of-word và tf-idf đối với bộ dữ liệu [10Topics](https://github.com/duyvuleo/VNTC/tree/master/Data/10Topics/Ver1.1).


5-. Kiến trúc chung của một mạng Deep CNN trong bài toán phân loại ảnh sẽ có dạng như thế nào? Phương pháp nào thường được sử dụng để tận dụng lại tri thức từ những mô hình đã được huấn luyện trước đó nhằm tiết kiệm tài nguyên tính toán ?


6-. Sử dụng code python để thực hành những bài tập liên quan tới biến đổi thời gian sau:
- Lấy ra current date và current time
- Current year
- Month of year
- Week number of the year
- Weekday of the week
- Day of year
- Day of the month
- Day of week

7-. Các kĩ thuật Standardization, Min-Max Scaling, Unit Length và Robust Scaling có đặc điểm như thế nào? Ưu điểm của kĩ thuật Robust Scaling so với các kĩ thuật còn lại?

8-. Nêu những phương pháp chính để thực hiện lựa chọn đặc trưng cho mô hình?

9-. Khởi tạo một mẫu dữ liệu example với 50 đặc trưng.

```
from sklearn.datasets import make_classification

# Khởi tạo dữ liệu example
X, y = make_classification(n_samples=500, n_features=50, random_state=123)
```

Thực hành phương pháp Auto-Encoder để giảm chiều dữ liệu từ 50 chiều về 10 chiều. Xây dựng mô hình phân loại trên 10 đặc trưng được giảm chiều và đánh giá độ chính xác mô hình. 

10-. Sử dụng các kĩ thuật lựa chọn đặc trưng khác nhau để lựa chọn ra 10 đặc trưng. Xây dựng mô hình phân loại trên các đặc trưng được lựa chọn và đánh giá độ chính xác.