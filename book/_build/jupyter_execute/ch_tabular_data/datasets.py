# Các bộ dữ liệu sử dụng trong sách

Các bộ dữ liệu dạng bảng trong cuốn sách này sẽ được chủ yếu lấy từ [các cuộc thi trên Kaggle](https://www.kaggle.com/competitions).

![](imgs/kaggle_competitions.png)

Đặc điểm nhận ra các bộ dữ liệu dạng bảng là các file csv ở phần "Data Explorer" trong tab "Data" như hình dưới đây với cuộc thi [Titanic](https://www.kaggle.com/c/titanic/overview).

```{margin}
Titanic được coi như "Hello world" với dữ liệu dạng bảng. Đây là bộ dữ liệu được dùng để minh họa các cách làm sạch dữ liệu và xây dựng đặc trưng điển hình.
```

![](imgs/titanic_data.png)

Trước khi download các bộ dữ liệu này, bạn cần tạo tài khoản và chấp nhận điều lệ của từng cuộc thi.

## Kaggle API

Để có thể download và nộp kết quả thông qua cửa sổ dòng lệnh (_terminal_), bạn cần cài đặt [kaggle-api](https://github.com/Kaggle/kaggle-api).
Chú ý đọc kỹ phần [API Credentials](https://github.com/Kaggle/kaggle-api#api-credentials) để cài đặt API Token cho tài khoản của bạn.

Mỗi cuộc thi sẽ có mã riêng để giúp các bạn thao thác thông qua API này. Câu lệnh để download dữ liệu cho mỗi cuộc thi cũng được cho trong tab "Data" của cuộc thi đó.
Ví dụ, để download dữ liệu Titanic, bạn có thể chạy lệnh sau đây:

```
kaggle competitions download -c titanic
```

Đoạn lệnh ví dụ dưới đây thực hiện thao tác download bộ dữ liệu Titanic về thư mục `data/tianic` và giải nén.

%%capture
!rm -rf ../data/titanic; mkdir -p ../data/titanic
!kaggle competitions download -c titanic -p ../data/titanic;
!cd ../data/titanic; unzip titanic.zip; rm titanic.zip;

```{note}
Site này được viết trong một jupyter notebook.
Các dấu `!` để báo với chương trình rằng đây không phải là một đoạn code python mà là các câu lệnh nên được thực hiện ở cửa sổ dòng lệnh.
```

!ls ../data/titanic

Sau khi giải nén, thư mục `data/titanic` chưa ba file: file `train.csv` chứa thông tin về tập huấn luyện, file `test.csv` chứa thông tin về tập kiểm tra, và file `gender_submission.csv` chứa ví dụ mẫu về cấu trúc của file nộp bài để Kaggle tính điểm. 

Ngoài Titanic, cuốn sách sẽ sử dụng các bộ dữ liệu sau đây làm ví dụ minh họa.

(sec_california_housing)=
## Giá nhà California

Đây là một bộ dữ liệu vừa tầm (khoảng 20k dòng và 9 cột) được sử dụng nhiều trong các bài hướng dẫn về xử lý dữ liệu trước khi xây dựng mô hình.
Bộ dữ liệu này được sử dụng trong [Machine Learning Crash Course của Google](https://developers.google.com/machine-learning/crash-course/california-housing-data-description), [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) và có thể tải về từ [Kaggle](https://www.kaggle.com/camnugent/california-housing-prices?select=housing.csv).

Lưu ý: một phiên bản khác của bộ dữ liệu này có thể được tìm thấy tại [sklearn.datasets](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). Tuy nhiên phiên bản này thiếu một trường dữ liệu hạng mục thú vị là "ocean_proximity".

Phiên bản sử dụng trong cuốn sách này được download từ [Kaggle](https://www.kaggle.com/camnugent/california-housing-prices?select=housing.csv) vào thư mục `../data/california_housing`.

Bài toán đặt ra là dự đoán trung vị của giá nhà tại các hạt trong bang California vào những năm 1990 dựa vào các thông tin như tuổi đời trung bình của nhà, thu nhập trung vị của mỗi hộ gia đình, số lượng phòng, dân số vùng và tọa độ của mỗi hạt.
Mặc dù bộ dữ liệu này đã lỗi thời, nó vẫn mang rất nhiều giá trị trong việc giảng dạy kỹ thuật xử lý dữ liệu dạng bảng.

## Dự đoán lượng mua

Có một loại bài toán phổ biến với dữ liệu dạng bảng là dự đoán lượng mua của mỗi sản phẩm trong các cửa hàng trong một khoảng thời gian dựa vào những thông tin bán hàng trong quá khứ.
Đây là một dạng bài toán hữu ích cho các nhà bán lẻ vì nó giúp họ chuẩn bị lượng sản phẩm cần thiết trong kho để tránh hiện tượng thiếu hàng hay tồn kho, đặc biệt trong các dịp lễ tết.

Trong cuốn sách này, một số bộ dữ liệu sau đây sẽ được sử dụng:

* [Rossmann store sales](https://www.kaggle.com/c/rossmann-store-sales): Trong cuộc thi này, các đội chơi được yêu cầu dự đoán lượng mua của các sản phẩm trong hơn 1000 cửa hàng của hãng Rossmann tại Đức.
Các thông tin về ngày lễ, khuyến mại cũng được sử dụng.

%%capture
!rm -rf ../data/rossmann; mkdir -p ../data/rossmann
!kaggle competitions download -c rossmann-store-sales -p ../data/rossmann;
!cd ../data/rossmann; unzip rossmann-store-sales.zip; rm rossmann-store-sales.zip;

* [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales): Cuộc thi này yêu cầu các đội chơi dự đoán số sản phẩm bán được trong một tháng tại một chuỗi các cửa hàng khác nhau của Nga dựa trên thông tin về giá cả, tên và lượng bán của mỗi sản phẩm mỗi ngày trong gần ba năm trước đó.

%%capture
!rm -rf ../data/sales; mkdir -p ../data/sales
!kaggle competitions download -c competitive-data-science-predict-future-sales -p ../data/sales;
!cd ../data/sales; unzip competitive-data-science-predict-future-sales.zip; rm competitive-data-science-predict-future-sales.zip

Các loại dữ liệu dựa vào lịch sử kèm thời gian thường được xếp vào dạng dữ liệu chuỗi thời gian.
Với dữ liệu dạng này, chúng ta sẽ được giới thiệu các kỹ thuật tạo đặc trưng, đặc biệt là đăng trưng dạng mùa vụ.

## Hệ thống gợi ý

Một nhóm bài toán thú vị khác là các bài hệ thống gợi ý.
Trong các bài toán này, nhiệm vụ của các kỹ sư machine learning và nhà khoa học dữ liệu là đưa ra sản phẩm gợi ý cho mỗi người dùng tại một thời điểm nhất định dựa trên lịch sử thể hiện sự ưa thích của người dùng với sản phẩm đã có trước đó. Tôi sẽ sử dụng một trong các bộ dữ liệu tại [Kaggle Recommendation System](https://www.kaggle.com/tags/recommender-systems).

Bạn đọc có thể xem thêm phần [Hệ thống gợi ý](https://machinelearningcoban.com/2017/05/17/contentbasedrecommendersys/) trong blog "Machine Learning cơ bản" về các bài toán lại này. Tuy nhiên, xin lưu ý rằng nội dung trên blog chưa hề đề cập đến dữ liệu dạng bảng và cách xử lý chúng. Cuốn sách này sẽ bàn kỹ hơn về cách tận dụng các thông tin liên quan đến người dùng và sản phẩm để có kết quả tốt hơn.

## Cấu trúc của mỗi bộ dữ liệu

Trong thực tế, các công ty hiếm khi lưu dữ liệu ở dạng csv vì tốc độ truy xuất và lưu trữ chậm.
Chúng thường được lưu ở các định dạng phù hợp với dữ liệu lớn hơn như hdfs và được trích xuất ra các dạng phù hợp hơn khi làm việc với các bài toán ML.
Trong phạm vi cuốn sách, việc lưu trữ và trích xuất dữ liệu như thế nào sẽ không được đề cập.
Dữ liệu được giả sử là đã ở dạng csv và chúng ta sẽ làm việc trực tiếp trên các file csv này.

Dữ liệu cho một bài toán có thể được cho trong một bảng lưu trong một file csv như file `train.csv` trong bộ dữ liệu Titanic.
Mỗi hàng trong file csv thường ứng với một mẫu dữ liệu với các trường thông tin được phân tách bởi dấu phẩy (`,`).
Dưới đây là 55 dòng đầu tiên của file `train.csv` khi nó được mở trong hầu hết các text editor.

!cat ../data/titanic/train.csv | head -5

Để cho dễ nhìn, bạn có thể sử dụng [`csvlook`](https://csvkit.readthedocs.io/en/1.0.2/scripts/csvlook.html) để quan sát dữ liệu dưới dạng bảng:

!csvlook ../data/titanic/train.csv | head -6 # one additional line as header/content spliter

Thông thường, các mô hình ML được yêu cầu dự đoán một cột sử dụng thông tin trong các cột còn lại.
Trong bài toán Titanic, cột dữ liệu đó là `"Survived"` thể hiện một hành khách có sống sót sau thảm họa Titanic hay không.
File `test.csv` có cấu trúc tương tự ngoại trừ việc nó không có cột `"Survived"`, cột này các đội tham gia cần dự đoán.

Đôi khi, cột cần dự đoán không có sẵn trong bảng dữ liệu mà được tính dựa trên các cột khác.
Chẳng hạn, với bảng dữ liệu trên, ta cũng có thể xây dựng bài toán "Dự đoán một hành khách có trên 30 tuổi hay không" dựa trên các cột còn lại.
Tất nhiên, khi đó cột `"Age"` chỉ có trong tập huấn luyện mà không có trong tập kiểm tra.
Khi xử lý dữ liệu, người kỹ sư ML cần xây dựng thêm một trường dữ liệu nữa có tên, chẳng hạn, `"Age_greater_30"` dựa vào cột `"Age"`

Trong hầu hết các trường hợp khác, dữ liệu thường được lưu ở nhiều bảng khác nhau.
Ví dụ với cuộc thi [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data), dữ liệu được lưu ở nhiều bảng khác nhau:

![](imgs/sales_data.png)

Dữ liệu chính được lưu trong file `sales_train.csv`, các thông tin liên quan về cửa hàng và sản phẩm lần lượt được lưu ở `shops.csv` và `items.csv`.
Ngoài ra, thông tin về mỗi hạng mục của sản phẩm được lưu ở `item_categories.csv`.
Các file `test.csv` và `sample_submision.csv` có mục đích chỉ ra những thông tin mà các kỹ sư ML cần dự đoán.

Nhìn chung, _mỗi bài toán ML với dự liệu dạng bảng có một cấu trúc dữ liệu khác nhau_.
Trong mỗi bài toán có thể có nhiều bảng dữ liệu khác nhau.
Trong ML với dữ liệu dạng bảng, phần lớn thời gian của các kỹ sư được dành cho việc xử lý các bảng dữ liệu này để tạo ra các đặc trưng dưới dạng số trước khi đưa chúng vào các mô hình ML.