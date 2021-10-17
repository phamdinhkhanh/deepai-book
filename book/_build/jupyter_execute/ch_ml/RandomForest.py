#!/usr/bin/env python
# coding: utf-8

# # 9.1. Ý tưởng của mô hình rừng cây
# 
# Mô hình _rừng cây_ được huấn luyện dựa trên sự phối hợp giữa luật _kết hợp_ (_ensembling_) và quá trình _lấy mẫu tái lặp_ (_boostrapping_). Cụ thể thuật toán này tạo ra **nhiều cây quyết định** mà mỗi cây quyết định được huấn luyện dựa trên **nhiều mẫu con** khác nhau và kết quả dự báo là _bầu cử_ (_voting_) từ toàn bộ những cây quyết định. Như vậy một kết quả dự báo được tổng hợp từ nhiều mô hình nên kết quả của chúng sẽ không bị chệch. Đồng thời kết hợp kết quả dự báo từ nhiều mô hình sẽ có phương sai nhỏ hơn so với chỉ một mô hình. Điều này giúp cho mô hình khắc phục được hiện tượng _quá khớp_. Ta sẽ bàn kỹ hơn điều này ở chương bên dưới.
# 
# Tiếp theo chúng ta sẽ tìm hiểu về luật _kết hợp_ và _lấy mẫu tái lặp_ trong mô hình rừng cây, và những điểm then chốt trong quá trình huấn luyện mô hình _rừng cây_.

# ## 9.1.1. Mô hình kết hợp (_ensemble model_)
# 
# Giả định rằng bạn đang xây dựng mô hình phân loại nhị phân ảnh chó và mèo lần lượt tương ứng với hai nhãn 0 và 1. Với một hình ảnh cụ thể, nếu chỉ sử dụng một mô hình duy nhất thì kết quả dự báo trả về có xác suất thuộc về nhãn mèo chỉ là 0.6. Đây là một xác suất không quá cao nên bạn không chắc chắn hình ảnh của của mình là mèo.
# 
# Bởi vì không chắc chắn, bạn muốn tham vấn kết quả từ nhiều mô hình hơn. Chính vì vậy bạn quyết định xây dựng 9 mô hình khác nhau và tiến hành _bầu cử_ kết quả trả về giữa chúng. Do đây là một trường hợp khó phát hiện, chẳng hạn ảnh bị nhoè và con vật đang núp dưới một gốc cây nên các mô hình đều dự báo xác suất không quá gần 1. Nhưng bất ngờ đó là trong kết quả trả về từ 9 mô hình thì có **8 mô hình** dự báo nhãn 1 và **1 mô hình** dự báo nhãn 0. Như vậy căn cứ vào kết quả _bầu cử_ bạn có thể tin cậy rằng nhãn dự báo cho bức ảnh là mèo là đúng.
# 
# ![](https://imgur.com/TGEOokD.jpeg)
# 
# Ở trên là ý tưởng của mô hình kết hợp cho tác vụ phân loại ảnh chó và mèo. Thông thường những kết quả từ mô hình kết hợp sẽ tốt hơn so với chỉ sử dụng một mô hình bởi chúng ta đang vận dụng trí thông minh đám đông (_wisdom of the crowd_). Điều này đã được kiểm chứng ở nhiều lớp mô hình khác nhau trong thực nghiệm.
# 
# Trên sklearn chúng ta có thể xây dựng một mô hình bầu cử thông qua class `sklearn.ensemble.VotingClassifier`.

# In[1]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
import numpy as np

# Load the dataset 
iris = load_iris()
X = iris.data
y = iris.target == 1

# Three model in ensemble learning
log_clf = LogisticRegression()
svm_clf = SVC()
tree_clf = DecisionTreeClassifier(max_depth=3)

voting_clf = VotingClassifier(
  estimators=[('lr', log_clf), ('svc', svm_clf), ('tree_clf', tree_clf)],
  voting='hard'
)

cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=1)


# In[2]:


# Đánh giá mô hình trên từng mô hình đơn lẻ
scores = cross_val_score(log_clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Logistic Regression Mean Accuracy: {:.03f}'.format(np.mean(scores)))
scores = cross_val_score(svm_clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('SVM Mean Mean Accuracy: {:.03f}'.format(np.mean(scores)))
scores = cross_val_score(tree_clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('DecisionTree Classifier Mean Accuracy: {:.03f}'.format(np.mean(scores)))


# In[3]:


# Đánh giá mô hình trên mô hình kết hợp
scores = cross_val_score(voting_clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Voting Classifier Mean Accuracy: {:.03f}'.format(np.mean(scores)))


# Chúng ta nhận thấy rằng trong ví dụ trên thì mô hình kết hợp có độ chính xác cao nhất so với toàn bộ các mô hình đơn lẻ khác (lưu ý kết quả có thể khác nhau ở mỗi lần chạy nhưng hầu hết các trường hợp đều cho mô hình kết hợp là chính xác nhất). Thậm chí khác biệt của nó so với mô hình kém nhất là `LogisticRegression` lên tới gần 0.24. Đây là một mức độ cải thiện rất cao.

# ## 9.1.2. Lấý mẫu tái lập (_boostrapping_)
# 
# Giả định dữ liệu huấn luyện mô hình là một tập $\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\}$ bao gồm $N$ quan sát. Thuật toán _rừng cây_ sẽ sử dụng phương pháp _lấy mẫu tái lập_ để tạo thành $B$ tập dữ liệu con. Quá trình _lấy mẫu tái lập_ này còn gọi là bỏ túi (_bagging_). Tức là chúng ta sẽ thực hiện $M$ lượt nhặt các mẫu từ tổng thể và bỏ vào túi để tạo thành tập $\mathcal{B}_i=\{(x_1^{(i)}, y_1^{(i)}), (x_2^{(i)}, y_2^{(i)}), \dots, (x_M^{(i)}, y_M^{(i)})\}$. Tập $\mathcal{B}_i$ cho phép các phần tử được lặp lại. Như vậy sẽ tồn tại những quan sát thuộc $\mathcal{D}$ nhưng không thuộc $\mathcal{B}_i$. Đây là những quan sát chưa được bỏ vào túi và chúng ta gọi chúng là _nằm ngoài túi_ (_out of bag_).
# 
# ![](https://imgur.com/hpOFHo3.jpeg)
# 
# Với mỗi tập dữ liệu $\mathcal{B}_i$ chúng ta xây dựng một mô hình _cây quyết định_ và trả về kết quả dự báo là $\hat{y}_j^{(i)} = f_i(\mathbf{x}_j)$. Trong đó $\hat{y}_j^{(i)}$ là dự báo của quan sát thứ $j$ từ mô hình thứ $(i)$, $\mathbf{x}_j$ là giá trị véc tơ đầu vào, $f_i(.)$ là hàm dự báo của mô hình thứ $i$. Mô hình dự báo từ cây quyết định là giá trị trung bình hoặc bầu cử của $B$ cây quyết định.
# 
# * Đối với mô hình dự báo: Chúng ta tính giá trị trung bình của các dự báo từ mô hình con.
# 
# $$\hat{y}_j = \frac{1}{B} \sum_{i=1}^{B} \hat{y}_j^{(i)}$$
# 
# * Đối với mô hình phân loại: Chúng ta thực hiện _bầu cử_ từ các mô hình con để chọn ra nhãn dự báo có tần suất lớn nhất.
# 
# $$\hat{y}_j = \arg \max_{c} \sum_{i=1}^{B} p(\hat{y_j}^{(i)} = c)$$
# 
# Như vậy phương sai của mô hình trong trường hợp đối với bài toán dự báo:
# 
# $$\begin{eqnarray}\sigma_{\hat{y}}^2 & = & \text{Var}(\frac{1}{B}\sum_{i=1}^{B} \hat{y}^{(i)})\\
# & = & \frac{1}{B^2} [\sum_{i=1}^{B} \text{Var}(\hat{y}^{(i)}) + 2 \sum_{1 \leq m < n \leq B}\text{cov}(y^{(m)}, y^{(n)}) ]
# \end{eqnarray}$$
# 
# Do kết quả của mô hình con $A$ không chịu ảnh hưởng hoặc phụ thuộc vào mô hình con $B$ nên ta có thể giả định kết quả dự báo từ các mô hình là hoàn toàn độc lập nhau. Tức là ta có $\text{cov}(y^{(m)}, y^{(n)}) = 0, ~~\forall{1 \leq \ m < n \leq B}$. Đồng thời giả định chất lượng các mô hình là đồng đều, được thể hiện qua phương sai dự báo là đồng nhất  $\text{Var}(\hat{y}^{(i)}) = \sigma^2, ~~\forall i=\overline{1, B}$. Từ đó suy ra:
# 
# $$\begin{eqnarray}\sigma_{\hat{y}}^2 & = & \frac{1}{B^2} [\sum_{i=1}^{B} \text{Var}(\hat{y}^{(i)})] \\
# & = & \frac{1}{B^2} B \sigma^2 = \frac{1}{B} \sigma^2
# \end{eqnarray}$$
# 
# Như vậy nếu sử dụng dự báo là trung bình kết hợp từ nhiều mô hình cây quyết định thì phương sai có thể giảm $B$ lần so với chỉ sử dụng một mô hình duy nhất. Trong một mô hình _rừng cây_, số lượng các _cây quyết định_ là rất lớn. Do đó phương sai dự báo từ mô hình có thể giảm gấp nhiều lần và tạo ra một dự báo ổn định hơn. 
# 
# Trên sklearn chúng ta sử dụng module `sklearn.ensemble.BaggingClassifier` để áp dụng thuật toán bỏ túi. Bên dưới là ví dụ về quá trình bỏ túi sử dụng 200 mô hình cây quyết định khác nhau. Mỗi mô hình được xây dựng dựa trên 100 mẫu dữ liệu đầu vào được lựa chọn một cách ngẫu nhiên. Khi chúng ta lựa chọn `boostrap=True` thì mô hình sẽ sử dụng phương pháp chọn mẫu có lặp lại trái lại thì quá trình lấy mẫu không cho phép các quan sát được lặp lại (hay còn gọi là phương pháp lấy mẫu _pasting_).

# In[4]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
  DecisionTreeClassifier(), 
  n_estimators=200,
  max_samples=100, 
  bootstrap=True, 
  n_jobs=-1
)

bag_clf.fit(X, y)

scores = cross_val_score(bag_clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Logistic Regression Mean Accuracy: {:.03f}'.format(np.mean(scores)))


# Ta có thể thấy khi sử dụng _bỏ túi_ thì độ chính xác của mô hình đã cải thiện được 0.002 điểm so với chỉ sử dụng mô hình kết hợp trên một bộ dữ liệu duy nhất.

# ## 9.1.3. Đánh giá mô hình dựa trên _mẫu nằm ngoài túi_ (_out of bag_)
# 
# Phương pháp bỏ túi cho phép lấy mẫu lặp lại các quan sát trên tập huấn luyện nên sẽ có một lượng lớn những quan sát chưa được đưa vào các tập huấn luyện con. Tập hợp những mẫu này gọi là _mẫu nằm ngoài túi_ (_out of bag_), được viết tắt là _oob_. Những dữ liệu này được lựa chọn ngẫu nhiên, độc lập và hoàn toàn không được học từ mô hình nên có thể được sử dụng để đánh giá mô hình tương đương với một _tập kiểm tra_. Trong `sklearn.ensemble.BaggingClassifier` chúng ta sử dụng thêm tuỳ chọn `oob_score=True` để đánh giá mô hình dựa trên các mẫu oob.

# In[5]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

idx = np.arange(X.shape[0])
np.random.seed(0)
np.random.shuffle(idx)

idx_train = idx[:100]
idx_test = idx[100:]

X_train, y_train = X[idx_train, :], y[idx_train]
X_test, y_test = X[idx_test, :], y[idx_test]

bag_clf = BaggingClassifier(
  DecisionTreeClassifier(), 
  n_estimators=200,
  max_samples=100, 
  bootstrap=True,
  oob_score=True, 
  n_jobs=-1
)

bag_clf.fit(X_train, y_train)
print('Out of bag accuracy: ', bag_clf.oob_score_)


# Đánh giá từ các quan sát oob cho thấy mô hình đạt được độ chính xác là `96.0%`. Còn trên tập kiểm tra độ chính xác đạt được là:

# In[6]:


y_pred = bag_clf.predict(X_test)
print('Out of bag accuracy: ', accuracy_score(y_pred, y_test))


# Như vậy kết quả đánh giá từ các mẫu _oob_ và tập kiểm tra là gần như bằng nhau. Bởi bản chất đây là những quan sát độc lập mà mô hình chưa được học. Về cơ bản thì nó cũng tương tự như những quan sát trên tập kiểm tra. Khi đánh giá mô hình dựa trên tập _oob_ sẽ tạo ra một kết quả khá khách quan giúp nhận biết hiện tượng _quá khớp_.

# ## 9.1.4. Mô hình rừng cây
# 
# Mô hình rừng cây sẽ áp dụng cả hai phương pháp _học kết hợp_ (_ensemble learning_) và _lấy mẫu tái lập_ (_boostrapping_). Thứ tự của quá trình tạo thành một mô hình rừng cây như sau:
# 
# ![](https://imgur.com/XdXbFfs.jpeg)
# 
# 
# 1. Lấy mẫu tái lập một cách ngẫu nhiên từ tập huấn luyện để tạo thành một tập dữ liệu con.
# 
# 2. Lựa chọn ra ngẫu nhiên $d$ biến và xây dựng mô hình cây quyết định dựa trên những biến này và tập dữ liệu con ở bước 1. Chúng ta sẽ xây dựng nhiều cây quyết định nên bước 1 và 2 sẽ lặp lại nhiều lần.
# 
# 3. Thực hiện _bầu cử_ hoặc lấy _trung bình_ giữa các cây quyết định để đưa ra dự báo.
# 
# Kết quả dự báo từ mô hình _rừng cây_ là sự kết hợp của nhiều cây quyết định nên chúng tận dụng được trí thông minh đám đông và giúp cải thiện độ chính xác so với chỉ sử dụng một mô hình _cây quyết định_. 
# 
# Nếu như mô hình cây quyết định thường bị nhạy cảm với dữ liệu _ngoại lai_ (_outlier_) thì mô hình _rừng cây_ được huấn luyện trên nhiều tập dữ liệu con khác nhau, trong đó có những tập được loại bỏ dữ liệu _ngoại lai_, điều này giúp cho mô hình ít bị nhạy cảm với dữ liệu _ngoại lai_ hơn. 
# 
# Sự kết hợp giữa các cây quyết định giúp cho kết quả ít bị chệch và phương sai giảm. Như vậy chúng ta giảm thiểu được hiện tượng _quá khớp_ ở mô hình rừng cây, một điều mà mô hình _cây quyết định_ thường xuyên gặp phải. 
# 
# Cuối cùng các bộ dữ liệu được sử dụng từ những cây quyết định đều xuất phát từ dữ liệu huấn luyện nên quy luật học được giữa các cây quyết định sẽ gần tương tự như nhau và tổng hợp kết quả giữa chúng không có xu hướng _bị chệch_.

# # 9.2. Huấn luyện mô hình rừng cây
# 
# Để huẩn luyện mô hình rừng cây cho bài toán phân loại chúng ta sử dụng class [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) và cho bài toán dự báo là class [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html). Hai class trên có các tham số có ý nghĩa tương tự nhau nên chúng ta sẽ đề cập đến `RandomForestClassifier`.
# 
# ```
# RandomForestClassifier(*,
# n_estimators=100,
# criterion='gini', 
# max_depth=None, 
# min_samples_split=2, 
# min_samples_leaf=1, 
# max_features=None,
# max_leaf_nodes=None, 
# min_impurity_decrease=0.0, 
# min_impurity_split=None,
# bootstrap=True, 
# oob_score=False,
# max_samples=None
# )
# ```
# 
# Bản chất của mô hình rừng cây là sự kết hợp giữa nhiều cây quyết định được huấn luyện theo phương pháp _lấy mẫu tái lập_. Do đó các tham số của nó sẽ lấy các tham số thiết lập cây quyết định từ class `DecisionTreeClassifier` và tham số tạo mẫu dữ liệu từ `BaggingClassifier`. Đối với các tham số của mô hình _cây quyết định_, các bạn có thể xem giải thích ý nghĩa tại [tuning siêu tham số cho mô hình cây quyết định](https://phamdinhkhanh.github.io/deepai-book/ch_ml/DecisionTree.html#8.10). Ba tham số cho thiết lập mẫu dữ liệu bao gồm `n_estimators`, `bootstrap`, `oob_score` và `max_samples`. Trong đó:
# 
# 
# * `n_estimators` là số lượng các cây quyết định được sử dụng trong mô hình rừng cây.
# * `bootstrap=True` tương ứng với sử dụng phương pháp _lấy mẫu tái lập_ khi xây dựng các cây quyết định. Trái lại thì chúng ta sử dụng toàn bộ dữ liệu. Lưu ý rằng ở đây do chúng ta đã lựa chọn ngẫu nhiên ra $d$ biến khi xây dựng cây quyết định nên dù sử dụng chung một tập dữ liệu thì các cây quyết định vẫn khác nhau. 
# 
# * `oob_score` chỉ có hiệu lực khi sử dụng khi sử dụng `boostrap`. Nếu `oob_score=True` thì sẽ tính toán thêm điểm số trên các _mẫu nằm ngoài túi_.
# 
# * `max_samples` là số lượng mẫu được sử dụng để huấn luyện mô hình cây quyết định. Mặc định `max_samples=None` thì chúng ta lấy ra các mẫu con có kích thước bằng với tập huấn luyện.
# 
# Bên dưới là một ví dụ huấn luyện mô hình _rừng cây_.

# In[7]:


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Huấn luyện mô hình trên tập train
rdf_clf = RandomForestClassifier(
    max_depth = 3,
    max_leaf_nodes = 16,
    min_samples_split = 10,
    min_samples_leaf = 10
)

rdf_clf.fit(X_train, y_train)

# Dự báo trên tập test
y_pred = rdf_clf.predict(X_test)
scores = accuracy_score(y_pred, y_test)
print('RandomForest Accuracy: {:.03f}'.format(scores))


# # 9.3. Đánh giá mức độ quan trọng của biến
# 
# Khi đánh giá mức độ quan trọng của một biến, mô hình _rừng cây_ là một lựa chọn nhanh gọn và tiện ích.
# 
# Chúng ta để ý thấy rằng những biến càng quan trọng thì càng gần _node gốc_ trong mô hình cây quyết định. Như vậy tính trung bình giá trị vị trí độ sâu của các biến xuất hiện trong toàn bộ các _cây quyết định_ của mô hình _rừng cây_ chúng ta hoàn toàn có thể xếp hạng được khá chuẩn xác biến nào có mức độ quan trọng hơn.

# In[8]:


import pandas as pd

pd.DataFrame({'features': iris.feature_names, 'importance':rdf_clf.feature_importances_}).sort_values('importance', ascending=False)


# Như vậy trong bộ dữ liệu iris thì `petal width` và `petal length` là những biến quan trọng nhất trong việc phân loại loài hoa `versicolor` với những loài hoa khác. Các biến `sepal length` và `sepal width` thì mức độ quan trọng là không đáng kể.

# # 9.4. Tổng kết
# 
# Qua bài viết này các bạn đã hiểu về bản chất của mô hình _rừng cây_ là sự kết hợp giữa nhiều mô hình _cây quyết định_ được huấn luyện trên các tập dữ liệu khác nhau được rút ra từ tập huấn luyện. Mô hình rừng cây có ưu điểm đó là giảm thiểu được hiện tượng _quá khớp_ do có phương sai thấp và ít bị ảnh hưởng bởi nhiễu như mô hình _cây quyết định_. Khi huấn luyện mô hình, mô hình _rừng cây_ cũng giúp chúng ta đánh giá nhanh tầm quan trọng của các biến đối với việc phân loại. Điều này cực kì hữu ích đối với những bộ dữ liệu có số chiều lớn.

# # 9.5. Bài tập
# 
# Tập dữ liệu đầu vào là [breast cancer](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29).
# 
# 1. Hãy xây dựng một mô hình kết hợp giữa nhiều mô hình phân loại khác nhau cho bài toán này. So sánh với từng mô hình riêng lẻ thì mô hình kết hợp có độ chính xác như thế nào?
# 
# 2. Sử dụng phương pháp _lấy mẫu tái lập_ (_boostrapping_) để huấn luyện mô hình cây quyết định trên những mẫu này. Đánh giá độ chính xác của mô hình trên các quan sát _nằm ngoài túi_ (viết tắt là _oob_). Những đánh giá trên _oob_ có tác dụng gì?
# 
# 3. Mô hình rừng cây hoạt động dựa trên sự kết hợp của những ý tưởng nào?
# 
# 4. Chúng ta làm thế nào để đánh giá mức độ quan trọng của biến đầu vào sử dụng mô hình _rừng cây_?
# 
# 5. Ưu và nhược điểm của mô hình rừng cây là gì?

# # 9.6. Tài liệu tham khảo
# 
# 1. [Random forest ensemble in python - machine learning mastery](https://machinelearningmastery.com/random-forest-ensemble-in-python/)
# 2. [Random forest datacamp](https://www.datacamp.com/community/tutorials/random-forests-classifier-python)
# 3. [Random forest wikipedia](https://en.wikipedia.org/wiki/Random_forest)
# 4. [Random forest freecodecamp](https://www.freecodecamp.org/news/how-to-use-the-tree-based-algorithm-for-machine-learning/)
