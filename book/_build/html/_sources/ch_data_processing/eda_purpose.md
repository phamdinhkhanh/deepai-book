# Mục đích của EDA

Bước EDA này giúp chúng ta có cái nhìn đầu tiên về dữ liệu.
Bạn cần có một cảm giác nhất định về những gì mình có trong tay trước khi có những chiến lược xây dựng mô hình.
EDA giúp bạn mường tượng được độ phức tạp của bài toán và vạch ra những bước đầu tiên cần làm.

Việc khám phá dữ liệu không chỉ dừng lại ở lần đầu tiên trước khi xây dựng đặc trưng mà còn cần được thực hiện trong suốt quá trình phát triển hệ thống.
Sau khi xây dựng xong các đặc trưng, bạn cũng cần làm lại EDA một lần nữa để xem dữ liệu đã qua xử lý đó đã thực sự _sạch_ chưa.
Ngoài ra, sau khi xây dựng và phân tích mô hình, ta cũng thường xuyên cần quay lại EDA để tiếp tục khám phá những điều còn ẩn giấu trong dữ liệu bài toán. Càng hiểu sâu về dữ liệu, bạn sẽ càng sớm giải thích được những hành vi của mô hình và đưa ra những thay đổi phù hợp.


```{note}
Để phân biệt dữ liệu trước và sau bước tiền xử lý, các cột trong bảng dữ liệu ban đầu được gọi là *trường thông tin*, *trường dữ liệu* hoặc đôi khi ngắn gọn là *trường*.
Các cột đã được xử lý và sẵn sàng cho việc huấn luyện mô hình được gọi là *đặc trưng*.
```

Vậy, làm EDA là làm những gì?

## Kích thước dữ liệu

Trước tiên, bạn cần hình dung được dữ liệu có khoảng bao nhiêu mẫu và có bao nhiêu trường dữ liệu.
Nếu dữ liệu quá ít, khả năng cao là bạn không thể dùng Deep Learning để giải quyết mà cần dùng các phương pháp khác.
Biết về kích thước dữ liệu cũng giúp bạn xác định kích thước tập huấn luyện (_training data_) và tập kiểm định (_validation data_) cũng như chuẩn bị bộ nhớ phù hợp.

## Ý nghĩa của từng trường dữ liệu

Làm việc với dữ liệu dạng bảng, bạn cần chuẩn bị tâm lý là sẽ làm việc với dữ liệu nhiều hơn là với mô hình.
Biết ý nghĩa của từng trường dữ liệu giúp bạn có những cách xử lý và tạo đặc trưng phù hợp.
Ý nghĩa của mỗi trường dữ liệu thường được đi kèm theo bộ dữ liệu hoặc đôi khi bạn phải tự suy luận ra dựa trên các giá trị trong cột.
Chẳng hạn, nếu các giá trị trong một cột là "Thái Bình", "Nghệ An", "Bến Tre", ... thì khả năng cao cột này là tên tỉnh thành.

Tuy nhiên, có những trường hợp vì lý do bảo mật thông tin mà các giá trị trong mỗi trường đã được mã hóa thành những giá trị vô nghĩa.
Với những trường hợp này, việc biết ý nghĩa của trường thông tin là cực kỳ quan trọng.
Nó ảnh hưởng trực tiếp tới cách xử lý đặc trưng và chất lượng của mô hình.
Nếu bạn không biết ý nghĩa và đặc biệt là dữ liệu được mã hóa dưới dạng số, chúc may mắn vì khả năng cao sẽ đến lúc mô hình có chất lượng không như ý muốn mà bạn không giải thích được.

## Kiểu dữ liệu của mỗi trường

Các mô hình Machine Learning cho dữ liệu dạng bảng khá nhạy cảm với kiểu dữ liệu.
Nhìn chung, các mô hình Machine Learning đều nhận dữ liệu đã qua xử lý ở dạng số.
Một trong những yêu cầu của một mô hình tốt là tính ổn định với những thay đổi nhỏ ở đầu vào.
Điều này tức là nếu đầu vào mô hình là những giá trị gần nhau thì đầu ra cũng được kỳ vọng là có giá trị gần nhau.
Nếu một dữ liệu dạng hạng mục được mã hóa về số, chẳng hạn mã số người dùng, mà bạn nghĩ nó là dạng số thì mô hình sẽ học được rằng những
người dùng có mã số gần nhau sẽ có những đặc trưng gần giống nhau.
Điều này khả năng rất cao không đúng, nhất là khi mã số được đánh một cách ngẫu nhiên.

## Phân phối xác suất của từng trường

Ta cần nắm được phân phối xác suất của từng trường dữ liệu để lên kế hoạch làm sạch dữ liệu và tạo các đặc trưng liên quan tới dữ liệu đó.
Với mỗi cột dữ liệu, những trường hợp sau đây ta cần lưu tâm:

1. **Mọi giá trị trong cột bằng nhau:** Ví dụ, trong dữ liệu có một cột là "Năm" và mọi giá trị đều bằng 2020.
Như vậy cột này không mang lại ý nghĩa dự đoán. Ta có thể xóa cột này khi làm sạch dữ liệu.

2. **Có quá nhiều giá trị bị khuyết:** Nếu thấy ý nghĩa cột này không quan trọng, ta có thể xóa.
Nếu nó quan trọng, bạn cần có những chiến lược phù hợp (Xem {ref}`sec_missing_data`).

3. **Xuất hiện giá trị không hợp lệ:**
Nếu trường dữ liệu "Tuổi" mang những giá trị là số âm hoặc lớn hơn 200, khả năng cao chúng là những giá trị không hợp lệ.
Với những giá trị không hợp lệ, ta có thể gán lại nó về giá trị hợp lệ gần nhất hoặc coi như dữ liệu bị khuyết.


4. **Xuất hiện giá trị ngoại lệ:**
Giả sử trường thông tin thu nhập hàng tháng chứa hầu hết giá trị trong khoảng từ 1-100 triệu đồng nhưng có một vài trường hợp ngoại lệ kiếm được tới 10 tỉ.
Nếu giữ nguyên giá trị 10 tỉ đó, mô hình dường như được huấn luyện gượng ép, nó phải căng sức dự đoán những giá trị ngoại lệ đó khiến chất lượng đối với những giá trị phổ biến bị ảnh hưởng. Ngoài ra, nếu phải làm bước chuẩn hóa dữ liệu về đoạn $[0, 1]$ (một kỹ thuật rất phố biến khi xử lý dữ liệu) thì hầu hết các giá trị nằm trong khoảng $[0, 0.01]$. Những giá trị rất nhỏ và gần nhau này có thể dẫn đến việc mô hình không phân biệt được sự khác nhau giữa các mức thu nhập khác nhau.
Đó là ví dụ với dữ liệu dạng số. Với dữ liệu dạng hạng mục có nhiều hạng mục khác nhau, nếu một số hạng mục chiếm tới 99% tổng số mẫu trong khi tổng số mẫu của nhiều hạng mục khác lại chỉ có 1%. Ta cần có những cách xử lý đặc biệt với dữ liệu loại này.


## Mối tương quan giữa các trường dữ liệu

Khi làm EDA, ta cũng cần tính toán độ tương quan giữa các trường dữ liệu, đặc biệt là giữa nhãn dự đoán và các trường còn lại.
Nếu hệ số tương quan giữa một cột và cột nhãn bằng không, khả năng cao cột đó không mang lại nhiều giá trị dự đoán.
Bạn có thể giành sự ưu tiên cho những cột có độ tương quan hơn.
Ngược lại, nếu hệ số tương quan giữa một cột và cột nhãn quá cao, có hai khả năng xảy ra:

* Cột đó có khả năng mang lại kết quả dự đoán tốt. Khi đó ta cần tập trung làm sạch và xây dựng các đặc trưng liên quan đến cột này trước.

* Dữ liệu có thể bị rò rỉ (_data leakage_). Giả sử bạn cần dự đoán tuổi của người dùng và nhận ra một cột có hệ số tương quan bằng 1.
Nếu đó là cột "Năm sinh" thì rõ ràng bạn chẳng cần làm mô hình Machine Learning mà chỉ cần một phép trừ.
Tuy nhiên, bài toán có thể là dự đoán tuổi khi trường "Năm sinh" bị khuyết nhưng biết nhiều thông tin khác.
Rõ ràng ta không thể dùng trường "Năm sinh" trong trường hợp này mà cần loại bỏ nó đi.

Ngoài ra, nếu hai cột không phải cột nhãn mà có độ tương quan cao, ta cũng nên kiểm tra ý nghĩa của chúng xem có thể bỏ qua một cột hay không.

------

Trả lời được những câu hỏi này sẽ giúp ích rất nhiều cho việc làm sạch dữ liệu và xây dựng đặc trưng sau này.

Tùy vào lượng thời gian bạn có và kiến thức của bạn về dữ liệu, bạn có thể phân tích sâu hơn về dữ liệu. Càng hiểu dữ liệu để xây dựng những đặc trưng phù hợp, mô hình của bạn càng có chất lượng tốt hơn.

```{tip}
Bạn không nên cố gắng dành thật nhiều thời gian vào EDA khi bắt đầu dự án mà chỉ nên dừng lại ở những đánh giá cơ bản trên đây để tìm ra những trường dữ liệu có khả năng cao mang lại kết quả tốt và xây dựng đặc trưng dựa trên những trường đó.
```

Bạn sẽ còn phải quay lại EDA nhiều lần nữa sau khi xây dựng được mô hình đầu tiên.
Chúng ta cần nhanh chóng xây dựng một pipeline hoàn thiện cho bài toán bao gồm xử lý dữ liệu, huấn luyện mô hình và đánh giá chất lượng mô hình. Bạn không cần quá chú trọng vào việc xây dựng một mô hình tốt ngay từ đầu, bạn nên quan tâm nhiều hơn tới một hệ thống hoàn chỉnh để đánh giá chất lượng mô hình và tìm ra những điểm cần cải thiện. Dựa trên những đánh giá đó, bạn có thể đưa ra những suy luận và kiểm chứng chúng bằng dữ liệu. Từ đó đưa ra những điều chỉnh phù hợp.

EDA là công việc tương đối nhàm chán nếu bạn chỉ thích huấn luyện mô hình. Rất may, hiện có rất nhiều thư viện hỗ trợ những chức năng cơ bản được đề cập trên đây. {ref}`sec_pandas_profiling` là một ví dụ.

Trước khi cùng tìm hiểu Pandas profiling, hãy làm một ví dụ nhỏ với dữ liệu Titanic về ý nghĩa của các trường dữ liệu.
