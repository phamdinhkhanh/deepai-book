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

+++ {"id": "3xgB2k8z1Y8J"}

# 15.1. Phương pháp phân cụm dựa trên mật độ (_Density-Based Clustering_)

Khi biểu diễn các điểm dữ liệu trong không gian chúng ta sẽ thấy rằng thông thường các vùng không gian có mật độ cao sẽ xen kẽ bởi các vùng không gian có mật độ thấp. Nếu như phải dựa vào mật độ để phân chia thì khả năng rất cao những tâm cụm sẽ tập trung vào những vùng không gian có mật độ cao trong khi biên sẽ rơi vào những vùng không gian có mật độ thấp. Trong lớp các mô hình phân cụm của học không giám sát tồn tại một kĩ thuật _phân cụm dựa trên mật độ_ (_Density-Based Clustering_), kĩ thuật này này đề cập đến các phương pháp _học không giám sát_ nhằm xác định các cụm phân biệt trong phân phối của dữ liệu, dựa trên ý tưởng rằng một cụm trong không gian dữ liệu là một vùng có mật độ điểm cao được ngăn cách với các cụm khác bằng các vùng liền kề có mật độ điểm thấp .

_DBSCAN_ là một thuật toán cơ sở để phân nhóm dựa trên mật độ. Nó có thể phát hiện ra các cụm có hình dạng và kích thước khác nhau từ một lượng lớn dữ liệu chứa _nhiễu_.

+++ {"id": "UhDas99ANX04"}

## 15.1.1. Các định nghĩa trong DBSCAN

Trước khi tìm hiểu về thuật toán _DBSCAN_ chúng ta xác định một số định nghĩa mà thuật toán này sử dụng.

**Định nghĩa 1:** Lân cận epsilon (_Eps-neighborhood_) của một điểm dữ liệu $P$ được định nghĩa là tợp hợp tất cả các điểm dữ liệu nằm trong phạm vi bán kính $\epsilon$ xung quanh điểm $P$. Kí hiệu tợp hợp những điểm này là:

$$N_{eps}(P) = \{Q \in \mathcal{D}: d(P, Q) \leq \epsilon\}$$

Trong đó $\mathcal{D}$ là tập hợp tất cả các điểm dữ liệu của tập huấn luyện.

Thuật toán _DBSCAN_ yêu cầu mỗi điểm trong một cụm phải thuộc vào ít nhất một vùng lân cận epsilon của một điểm khác nằm trong cụm. Do đó lân cận epsilon nhằm tạo cơ sở để chúng ta liên kết các điểm dữ liệu lại thành cụm thông qua xem xét khả năng các điểm rơi vào vùng lân cận epsilon trong cụm lẫn nhau.

**Định nghĩa 2:** Khả năng tiếp cận trực tiếp mật độ của một điểm dữ liệu (_directly density-reachable_) đề cập tới khả năng một điểm có thể tiếp cận tới một điểm dữ liệu khác thông qua vùng lân cận epsilon. Cụ thể là một điểm $Q$ được coi là có thể tiếp cận trực tiếp bởi điểm $P$ tương ứng với tham số `epsilon` và `minPts` nếu như nó thoả mãn hai điều kiện:

1. $Q$ nằm trong vùng lân cận `epsilon` của $P$: $Q \in N_{eps}(P)$
2. Số lượng các điểm dữ liệu nằm trong vùng lân cận tối thiểu là `minPts`: $|N_{eps}(Q)| \geq \text{minPts} $

Trong một vùng lân cận epsilon của một điểm dữ liệu thì mật độ của các điểm dữ liệu sẽ phụ thuộc vào số lượng điểm dữ liệu nằm bên trong nó. Một điểm dữ liệu có thể tiếp cận được tới một điểm khác không chỉ dựa vào khoảng cách giữa chúng mà còn phụ thuộc vào mật độ các điểm dữ liệu xung quanh của điểm dữ liệu đó. Mật độ này được coi là dày đặc và cho thấy các điểm thuộc _vùng lân cận_ rơi vào miền có mật độ cao nếu số lượng điểm tối thiểu bằng `minPts`. Trong trường hợp mật độ của vùng lân cận `epsilon` quá thấp thì điểm dữ liệu ở trung tâm có thể coi là không kết nối được tới những điểm xung quanh. Khi đó điểm dữ liệu có thể rơi vào biên của cụm hoặc là một điểm dữ liệu _nhiễu_.

**Định nghĩa 3:** Khả năng tiếp cận mật độ (_density-reachable_) liên quan đến cách hình thành một chuỗi liên kết giữa các điểm dữ liệu. Cụ thể là trong một chuỗi các điểm dữ liệu $\{P_i\}_{i=1}^{n} \subset \mathcal{D}$ nếu như bất kì một điểm $P_{i}$ nào cũng đều có thể _tiếp cận trực tiếp_ theo mật độ bởi $P_{i-1}$ tương ứng với tham số `epsilon` và `minPts`. Khi đó ta nói điểm $P = P_n$ có khả năng _kết nối mật độ_ tới điểm $Q = P_1$.

Từ định nghĩa 3 ta suy ra hai điểm $P_i$ và $P_j$ bất kì thuộc chuỗi $\{P_i\}_{i=1}^{n}$ thoả mãn $i < j$ thì $P_j$ có khả năng liên kết mật độ tới $P_i$. Thuật toán _DBSCAN_ sẽ phân cụm các điểm thuộc tập $\{P_i\}_{i=1}^{n}$ về cùng một cụm. Khả năng tiếp cận mật độ đề cập tới sự mở rộng phạm vi của một cụm dữ liệu dựa trên liên kết theo chuỗi. Xuất phát từ một điểm dữ liệu ta có thể tìm được các điểm có khả năng _kết nối mật độ_ tới nó và từ đó làm cơ sở xác định cụm dữ liệu.

+++ {"id": "mF3lrJjGppkt"}

## 15.1.2. Phân loại dạng điểm trong DBSCAN

Căn cứ vào vị trí của các điểm dữ liệu so với cụm chúng ta có thể chia chúng thành ba loại: Đối với các điểm nằm sâu bên trong cụm chúng ta xem chúng là _điễm lõi_. Các _điểm biên_ nằm ở phần ngoài cùng của cụm và _điểm nhiễu_ không thuộc bất kì một cụm nào. Bên dưới là hình vẽ mô phỏng thể hiện ba loại điểm tương ứng nêu trên.

![](https://imgur.com/ohzPUif.png)

**Hình 2**: Hình minh hoạ cách xác định ba loại điểm bao gồm: _điểm lõi_ (_core_) chấm vuông màu xanh, _điểm biên_ (_border_), chấm tròn màu đen và _điểm nhiễu_ (_noise_) chấm tròn màu trắng trong thuật toán _DBSCAN_. Các hình tròn đường viền nét đứt bán kính $\epsilon$ thể hiện phạm vi khoảng cách từ các _điểm lõi_ để xác định nhãn cho từng điểm. `minPts=3` là số lượng tối thiểu để một _điểm lõi_ rơi vào vùng có mật độ cao nếu xung quanh chúng có số lượng điểm tối thiểu là 3.


Trong thuật toán _DBSCAN_ sử dụng hai tham số chính đó là:

* `minPts`: Là một ngưỡng số điểm dữ liệu tối thiểu được nhóm lại với nhau nhằm xác định một vùng lân cận `epsilon` có mật độ cao.

* `epsilon` ( kí hiệu $\epsilon$ ): Một giá trị khoảng cách được sử dụng để xác định vùng lân cận `epsilon` của bất kỳ điểm dữ liệu nào.

Hai tham số trên sẽ được sử dụng để xác định vùng lân cận epsilon và khả năng tiếp cận giữa các điểm dữ liệu lẫn nhau. Từ đó giúp kết nối chuỗi dữ liệu vào chung một cụm.

Khi phân cụm thì thuật toán DBSCAN hình tạo thành ba loại điểm chính:

* _điểm lõi_ (_core_): Đây là một điểm có ít nhất `minPts` điểm trong vùng lân cận `epsilon` của chính nó.
* _điểm biên_ (_border_): Đây là một điểm có ít nhất một _điểm lõi_ nằm ở vùng lân cận `epsilon` của nó nhưng có mật độ không đủ `minPts` điểm trong vùng lân cận.
* _điểm nhiễu_ (_noise_): Đây là điểm không phải là _điểm lõi_ hay _điểm biên_ và nó có ít hơn `minPts` điểm nằm trong vùng lân cận epsilon của nó.

Đối với một cặp điểm $(P, Q)$ bất kì sẽ có ba khả năng:

* Cả $P$ và $Q$ đều có khả năng _kết nối mật độ_ được với nhau: Cả $P$, $Q$ đều thuộc về chung một cụm.

* $P$ có khả năng _kết nối mật độ_ được với $Q$ nhưng $Q$ không _kết nối mật độ_ được với $P$. Khi đó $P$ sẽ là _điểm lõi_ của cụm còn $Q$ là một _điểm biên_.

* $P$ và $Q$ đều không _kết nối mật độ_ được với nhau: Trường hợp này $P$ và $Q$ sẽ rơi vào những cụm khác nhau hoặc một trong hai điểm là _điểm nhiễu_.

+++ {"id": "ZpYmUqxJC8z2"}

# 15.3. Các bước trong thuật toán DBSCAN

Các bước của thuật toán DBSCAN khá đơn giản. Thuật toán sẽ thực hiện lan truyền để mở rộng dần phạm vi của cụm cho tới khi chạm tới những _điểm biên_ thì thuật toán sẽ chuyển sang một cụm mới và lặp lại tiếp quá trình trên. Cụ thể bạn sẽ thấy được quá trình lan truyền này thông qua hình minh hoạ bên dưới.


![](https://imgur.com/9D6aAF2.gif)

**Hình 3**: Quá trình lan truyền để xác định các cụm của thuật toán DBSCAN. [Source - digitalvidya blog](https://www.digitalvidya.com/blog/the-top-5-clustering-algorithms-data-scientists-should-know/)

Cụ thể các bước sẽ như sau:

* **Bước 1:** Thuật toán lựa chọn một điểm dữ liệu bất kì. Sau đó tiến hành xác định các _điểm lõi_ và _điểm biên_ thông qua vùng lân cận epsilon. Nếu có ít nhất số lượng `minPts` điểm nằm trong vùng lân cận epsilon tại một điểm dữ liệu thì chúng ta coi tất cả các điểm lân cận là một phần của cùng một cụm. Nếu không thì điểm đó được xem là điểm biên.

* **Bước 2:** Các cụm sau đó được mở rộng bằng cách lặp lại đệ quy phép tính lân cận cho mỗi điểm thuộc vùng lân cận. Sau khi đi đến điểm cuối cùng của cụm thì ta lặp lại đệ qui bước 1 trên điểm dữ liệu khác trong số dữ liệu còn lại và tiếp tục quá trình xác định một cụm mới.

+++ {"id": "Lf2ctd0kJjFT"}

# 4. Xác định tham số 

Xác định tham số là một bước quan trọng và ảnh hưởng trực tiếp tới kết quả của các thuật toán. Đối với thuật _DBSCAN_ cũng không ngoại lệ. Chúng ta cần phải xác định chính xác tham số cho thuật toán _DBSCAN_ một cách phù hợp với từng bộ dữ liệu cụ thể, tuỳ theo đặc điểm và tính chất của phân phối của bộ dữ liệu. Hai tham số cần lựa chọn trong _DBSCAN_ đó chính là `minPts` và `epsilon`:

* `minPts`: Theo quy tắc chung, $\text{minPts}$ tối thiểu có thể được tính theo số chiều $D$ trong tập dữ liệu đó là $\text{minPts} \geq D + 1$. Một giá trị $\text{minPts} = 1$ không có ý nghĩa, vì khi đó mọi điểm bản thân nó đều là một cụm. Với $\text{minPts} \leq 2$, kết quả sẽ giống như _phân cụm phân cấp_ (_hierarchical clustering_) với _single linkage_ với biểu đồ _dendrogram_ được cắt ở độ cao $y=$ `epsilon`. Do đó, $\text{minPts}$ phải được chọn ít nhất là $3$. Tuy nhiên, các giá trị lớn hơn thường tốt hơn cho các tập dữ liệu có nhiễu và kết quả phân cụm thường hợp lý hơn. Theo quy tắc chung thì thường chọn $\text{minPts} = 2 \times \text{dim}$. Trong trường hợp dữ liệu có nhiễu hoặc có nhiều quan sát lặp lại thì cần lựa chọn giá trị $\text{minPts}$ lớn hơn nữa tương ứng với những bộ dữ liệu lớn.

* `epsilon`: Giá trị $\epsilon$ có thể được chọn bằng cách vẽ một biểu đồ `k-distance`. Đây là biểu đồ thể hiện giá trị khoảng cách trong thuật toán k-Means clustering đến $k = \text{minPts}-1$ điểm láng giềng gần nhất. Ứng với mỗi điểm chúng ta chỉ lựa chọn ra khoảng cách lớn nhất trong $k$ khoảng cách. Những khoảng cách này trên đồ thị được sắp xếp theo thứ tự giảm dần. Các giá trị tốt của $\epsilon$ là vị trí mà biểu đồ này cho thấy xuất hiện một điểm _khuỷ tay_ (_elbow point_): Nếu $\epsilon$ được chọn quá nhỏ, một phần lớn dữ liệu sẽ không được phân cụm và được xem là _nhiễu_; trong khi đối với giá trị $\epsilon$ quá cao, các cụm sẽ hợp nhất và phần lớn các điểm sẽ nằm trong cùng một cụm. Nói chung, các giá trị nhỏ của $\epsilon$ được ưu tiên hơn và theo quy tắc chung, chỉ một phần nhỏ các điểm nên nằm trong vùng lân cận epsilon.

* Hàm khoảng cách: Việc lựa chọn hàm khoảng cách có mối liên hệ chặt chẽ với lựa chọn $\epsilon$ và tạo ra ảnh hưởng lớn tới kết quả. Điểm quan trọng trước tiên đó là chúng ta cần xác định một thước đo hợp lý về _độ khác biệt_ (_disimilarity_) cho tập dữ liệu trước khi có thể chọn tham số $\epsilon$. Khoảng cách được sử dụng phổ biến nhất là `euclidean distance`.

Tiếp theo chúng ta sẽ cùng huấn luyện thuật toán DBSCAN trên bộ dữ liệu [shopping-data](https://raw.githubusercontent.com/phamdinhkhanh/datasets/cf391fa1a7babe490fdd10c088f0ca1b6d377f59/shopping-data.csv) để hiểu rõ cách thức lựa chọn tham số cho mô hình cũng như các bước trong quá trình huấn luyện và dự báo.

+++ {"id": "I-qoCF2heP-Z"}

# 15.4. Huấn luyện thuật toán DBSCAN

Bộ dữ liệu [shopping-data](https://raw.githubusercontent.com/phamdinhkhanh/datasets/cf391fa1a7babe490fdd10c088f0ca1b6d377f59/shopping-data.csv) bao gồm 200 quan sát về điểm chi tiêu của khách hàng. Bộ dữ liệu bao gồm các trường thông tin đầu vào như giới tính, độ tuổi, thu nhập và điểm chi tiêu. Một một quan sát được đặc trưng bởi trường CustomerID đại diện cho mã khách hàng. Nhiệm vụ của chúng ta đó là sử dụng thuật toán DBSCAN để phân cụm tập khách hàng này vào những nhóm có chung đặc tính và hành vi mua sắm để chăm sóc và phục vụ họ tốt hơn.

Để huấn luyện mô hình phân cụm sử dụng thuật toán DBSCAN thì chúng ta cần import class [sklearn.cluster.DBSCAN](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

```{code-cell}
:id: dbMPDeEoew1p

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import numpy as np
```

+++ {"id": "ha4Qw29G0-eS"}

Đọc dữ liệu đầu vào và chuẩn hoá dữ liệu.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 252
id: BE8Ghznpev8d
outputId: 2388d6ab-e276-4c9c-ed6f-6aace9ad368c
---
data = pd.read_csv("https://raw.githubusercontent.com/phamdinhkhanh/datasets/cf391fa1a7babe490fdd10c088f0ca1b6d377f59/shopping-data.csv", header=0, index_col=0)
print(data.shape)
data.head()
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: hPip7xMNe1Yp
outputId: 9aa05ba1-c500-402f-80d7-61eafca3a880
---
# Lấy ra thu nhập va điểm shopping
X = data.iloc[:, 2:4].values
print(X.shape)
```

+++ {"id": "n6xjj4xn1Cd7"}

Chúng ta nhận thấy rằng các trường dữ liệu có sự khác biệt về độ lớn đơn vị giữa các biến nên tiếp theo cần chuẩn khoá dữ liệu để đồng nhất đơn vị giữa chúng. Chúng ta chuẩn hoá `MinMaxScaler()`. Đối với thuật toán DBSCAN thì các điểm dữ liệu outliers sẽ tự động được tách khỏi cụm nên thuật toán không chịu ảnh hưởng nhiều bởi outliers như k-Means Clustering. Chúng ta có thể bỏ qua bước loại bỏ outliers cho bộ dữ liệu.

```{code-cell}
:id: 9JzEtFaSe5J8

std = MinMaxScaler()
X_std = std.fit_transform(X)
```

+++ {"id": "elc7NFoujsJT"}

## 15.4.1. Lựa chọn tham số cho mô hình DBSCAN

Tiếp theo chúng ta sẽ sử dụng biểu đồ `k-distance` như đã trình bày ở mục `xác định tham số ` để lựa chọn khoảng cách $\epsilon$ phù hợp cho mô hình _DBSCAN_. Không mất đi tính chất của khoảng cách của dữ liệu thì chúng ta giả định hàm khoảng cách được lựa chọn là `euclidean distance`. Cuối cùng chúng ta lựa chọn số lượng điểm dữ liệu tối thiểu nằm trong vùng lân cận là `minPts=11` (theo nguyên tắc chung thì $\text{minPts}$ cần tối thiểu bằng $2 \times \text{dim}$ của bộ dữ liệu). Điều này tương ứng với trong thuật toán k-Means mà chúng ta áp dụng để vẽ biểu đồ `k-distance` thì cần lựa chọn số láng giềng $k=10$.

Khi xây dựng mô hình với những tham số này sẽ tạo ra được những cụm phân chia có tính chất tổng quát nhất. Tránh được các trường hợp có quá nhiều cụm nhỏ lẻ được phân chia và nhiễu được tạo thành khi $\epsilon$ nhỏ và trường hợp khác là toàn bộ các điểm bị phân về một cụm nếu lựa chọn $\epsilon$ lớn.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 530
id: EXRXdjS0j2VP
outputId: 198ac29d-dcf9-4af6-cddb-d440ee28bca6
---
from sklearn.neighbors import NearestNeighbors


# Xây dựng mô hình k-Means với k=10
neighbors = 10
nbrs = NearestNeighbors(n_neighbors=neighbors ).fit(X_std)

# Ma trận khoảng cách distances: (N, k)
distances, indices = nbrs.kneighbors(X_std)

# Lấy ra khoảng cách xa nhất từ phạm vi láng giềng của mỗi điểm và sắp xếp theo thứ tự giảm dần.
distance_desc = sorted(distances[:, neighbors-1], reverse=True)

# Vẽ biểu đồ khoảng cách xa nhất ở trên theo thứ tự giảm dần
plt.figure(figsize=(12, 8))
plt.plot(list(range(1,len(distance_desc )+1)), distance_desc)
plt.axhline(y=0.12)
plt.text(2, 0.12, 'y = 0.12', fontsize=12)
plt.axhline(y=0.16)
plt.text(2, 0.16, 'y = 0.16', fontsize=12)
plt.ylabel('distance')
plt.xlabel('indice')
plt.title('Sorting Maximum Distance in k Nearest Neighbor of kNN')
```

+++ {"id": "SXxQg9_3npPi"}

Từ biểu đồ k-distance chúng ta có thể thấy điểm `elbow` tương ứng với $\epsilon \in [0.12, 0.16]$. Tiếp theo chúng ta sẽ tìm kiếm giá trị của tham số $\epsilon$ trong khoảng $[0.12, 0.16]$ cho mô hình _DBSCAN_. Tham số `minPts` được cố định là $11$ như lúc đầu lựa chọn và để tương ứng với biểu đồ k-Means.

+++ {"id": "GeeSjs4IjvYg"}

## 15.4.2. Xây dựng mô hình DBSCAN

+++ {"id": "E15X_pt_e__Q"}

Để xây dựng mô hình _DBSCAN_ trên sklearn chúng ta sử dụng class [sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).

```
DBSCAN(eps=0.5,
 min_samples=5, 
 metric='euclidean', 
 algorithm='auto'
)
```

Trong đó các tham số chính cần quan tâm đó là 

* `eps`: Chính là khoảng cách $\epsilon$ giúp xác định các điểm nằm trong vùng lân cận epsilon. Đây cũng là giá trị khó xác định nhất và tuỳ thuộc vào đặc trưng phân phối của mỗi bộ dữ liệu.

* `min_samples`: Số lượng tối thiểu các điểm láng giềng xung quanh một điểm để xác định một _điểm lõi_, số lượng này đã bao gồm _điểm lõi_. Tương đương với `minPts+1` đã giới thiệu ở trên.

* `metric`: Hàm khoảng cách để đo lường khoảng cách giữa hai điểm bất kì, nhận mặc định là `euclidean`. Hàm khoảng cách và giá trị $\epsilon$ là hai tham số có mối quan hệ chặt chẽ và ảnh hưởng qua lại lẫn nhau và ảnh hưởng lên kết quả phân cụm.

* `algorithm`: Phương pháp được sử dụng để xác định các điểm láng giềng. Bao gồm các phương pháp `auto, ball_tree, kd_tree, brute`. Mặc định là `auto`. Về những phương pháp này bạn có thể tìm hiểu thêm tại [sklearn.neighbors](https://scikit-learn.org/stable/modules/neighbors.html).



```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: wHATrKKRjgy2
outputId: b33205d0-eec7-4b15-af73-e4500d7515a9
---
from matplotlib.gridspec import GridSpec
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

def _plot_kmean_scatter(X, labels, gs, thres):
    '''
    X: dữ liệu đầu vào
    labels: nhãn dự báo
    '''
    # lựa chọn màu sắc
    num_classes = len(np.unique(labels))
    palette = np.array(sns.color_palette("hls", num_classes))

    # vẽ biểu đồ scatter
    ax = plt.subplot(gs)
    sc = ax.scatter(X[:,0], X[:,1], lw=0, s=40, c=palette[labels.astype(np.int)])

    # thêm nhãn cho mỗi cluster
    txts = []

    for i in range(num_classes):
        # Vẽ text tên cụm tại trung vị của mỗi cụm
        indices = (labels == i)
        xtext, ytext = np.median(X[indices, :], axis=0)
        if not (np.isnan(xtext) or np.isnan(ytext)):        
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
    plt.title('t-sne visualization for thres={:.4f}'.format(thres))

gs = GridSpec(3, 4)
plt.figure(figsize = (25, 18))
plt.subplots_adjust(wspace=0.1,hspace=0.4)

for i, thres in enumerate(np.linspace(0.11, 0.14, 12)):
    dbscan = DBSCAN(eps=thres, min_samples=11, metric='euclidean')
    labels = dbscan.fit_predict(X_std)
    _plot_kmean_scatter(X_std, labels, gs[i], thres)
```

+++ {"id": "H7GBJrElJwQ8"}

Giá trị của epsilon ảnh hưởng khá nhạy lên kết quả phân cụm. Căn cứ vào biểu đồ chúng ta có thể lựa chọn $\epsilon = 0.1209$ là giá trị mà các cụm có vẻ mang lại kết quả phân chia tổng quát nhất trên tập dữ liệu huấn luyện. Giá trị này có thể khác biệt theo phương pháp chuẩn hoá dữ liệu và cách lựa chọn trường dữ liệu đầu vào.

+++ {"id": "4uoPU-b861QQ"}

# 15.5. Tổng kết

DBSCAN là một thuật toán đơn giản và hiệu quả. Nó hoạt động dựa trên cách tiếp cận mật độ phân phối của dữ liệu. Ưu điểm của thuật toán đó là có thể tự động loại bỏ được các điểm dữ liệu nhiễu, hoạt động tốt đối với những dữ liệu có hình dạng phân phối đặc thù và có tốc độ tính toán nhanh. Tuy nhiên DBSCAN thường không hiệu quả đối với những dữ liệu có phân phối đều khắp nơi. Khi huấn luyện DBSCAN thì các tham số của mô hình như khoảng cách `epsilon`, số lượng điểm lân cận tối thiểu `minPts` và hàm khoảng cách là những tham số có ảnh hưởng rất lớn đối với kết quả phân cụm. Thực tế cho thấy thuật toán khá nhạy với tham số `epsilon` và `minPts` nên chúng ta cần phải lựa chọn tham số cho mô hình trước khi tiến hành xây dựng mô hình. 

+++ {"id": "AYhKo1Fh63HT"}

# 15.6. Bài tập

1. Phương pháp phân cụm dựa trên mật độ có nghĩa là gì?
2. Nêu cách xác định điểm lõi, điểm biên và điểm nhiễu trong thuật toán DBSCAN.
3. Quá trình huấn luyện thuật toán DBSCAN diễn ra như thế nào?
4. Thuật toán DBSCAN có ưu điểm hơn so với k-Means clustering là gì?
5. Một bộ dữ liệu có phân phối đặc biệt mà trong đó cụm này bao bọc vòng quanh cụm kia thì thuật toán nào sẽ phù hợp để phân cụm dữ liệu trong các thuật toán k-Means clustering, hierachical clustering và DBSCAN?
6. Thuật toán nào đạt được chi phí toán hiệu quả nhất trong các thuật toán k-Means clustering, hierachical clustering và DBSCAN?
7. Để lựa chọn ra được giá trị `epsilon` phù hợp cho một bộ dữ liệu trước khi huấn luyện mô hình DBSCAN thì phương pháp nào thường được sử dụng?
8. Thông thường thì nên lựa chọn `minPts` như thế nào trong thuật toán DBSCAN?
9. Lấy một bộ dữ liệu bất kì từ nguồn [UCI](https://archive.ics.uci.edu/ml/datasets.php?format=&task=clu&att=&area=&numAtt=&numIns=&type=&sort=nameUp&view=table) hãy thực hiện huấn luyện một mô hình phân cụm.
10. So sánh kết quả phân cụm với các thuật toán khác như k-Means clustering và hierachical clustering.

+++ {"id": "rzvi7xCJ64l5"}

# 15.7. Tài liệu tham khảo

+++ {"id": "BEFDn70Nk1MT"}

1. https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf

2. https://en.wikipedia.org/wiki/DBSCAN

3. https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html

4. https://towardsdatascience.com/how-dbscan-works-and-why-should-i-use-it-443b4a191c80

5. https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/

6. https://stats.stackexchange.com/questions/88872/a-routine-to-choose-eps-and-minpts-for-dbscan

7. https://www.coursera.org/lecture/predictive-analytics/dbscan-EVHfy

8. https://www.youtube.com/watch?v=6jl9KkmgDIw

9. https://www.youtube.com/watch?v=dGsxd67IFiU