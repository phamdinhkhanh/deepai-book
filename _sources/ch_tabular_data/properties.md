# Đặc điểm của dữ liệu dạng bảng

## Sự khan hiếm của dữ liệu
 
Một trong những đặc điểm của dữ liệu dạng bảng là khó khăn trong việc thu thập dữ liệu.
Dữ liệu ảnh hay văn bản có thể được tìm kiếm dễ dàng qua các bộ dữ liệu được công khai trên mạng.
Với dữ liệu bảng, mỗi công ty thường có dữ liệu và cách thu thập riêng;
và quan trọng hơn, những dữ liệu này là tài sản quan trọng của họ và rất ít khi được công bố rộng rãi.
Các công ty lớn có thể công bố thuật toán, mã nguồn của nghiều mô hình ML, nhưng dữ liệu mới là tài sản quý hơn cả.
Việc khan hiếm của dữ liệu dạng bảng một phần dẫn đến sự thiếu hụt về các tài liệu cho dữ liệu loại này và cũng gián tiếp dẫn đến việc các thuật toán Deep Learning (DL), vốn cần rất nhiều dữ liệu để huấn luyện, thường không mang lại kết quả tốt nhất.

(sec_mising_data_intro)=
## Dữ liệu bị nhiễu hoặc khuyết

Nhiều đặc trưng trong dữ liệu dạng bảng thường được thu thập bằng các phiếu khảo sát
(điện tử hoặc thủ công). Chẳng hạn, khi người dùng tạo tài khoản ở một trang mạng, họ
được yêu cầu nhập tên, tuổi, quê quán, vị trí địa lý, v.v; chuyện người dùng cố tình
khai báo sai thông tin chắc chắn không phải là chuyện hiếm. Thậm chí, một người dùng có
thể có nhiều tài khoản ảo với những thông tin trái ngược. Hoặc họ có thể đã từ chối cung
cấp một loại thông tin nào đó, chẳng hạn tắt GPS, khiến trường thông tin đó bị khuyết.

## Nhiều đặc trưng hạng mục

Các mô hình ML, đặc biệt là các mô hình DL, thường hoạt động tốt khi dữ liệu đầu vào ở dạng số và liên tục. Dữ liệu ảnh, mặc dù
nhận các giá trị số nguyên nhưng cũng có thể coi là liên tục với màu sắc thay đổi từ từ theo giá trị
các điểm ảnh. Đầu vào của các mô hình NLP cũng thường là các embedding vector của các từ/câu/văn bản, các
vector này là vector của các số thực liên tục. Các embedding gần nhau trong không gian cũng thường mang
ý nghĩa gần nhau. Thật không may, dữ liệu dạng bảng thường ít khi ở dưới dạng liên tục.

Đặc trưng trong dữ liệu bảng có thể là một trong nhiều hạng mục khác nhau (_categorical data_).
Chẳng hạn, nơi sinh của người dùng, tên của một loại sản phẩm hay mã của một phần quảng cáo là các loại đặc trưng ở dạng danh mục.
Mặc dù vẫn có thể có các hạng mục mang ý nghĩa gần với nhau (ví dụ về mặt địa lý hoặc về mặt chủng loại), rất khó để đo đếm sự gần nhau đó.
Hà Nội có thể rất xa Tp HCM và gần Hà Giang hơn, nhưng Hà Nội lại giống Tp HCM hơn theo nghĩa đều là các thành phố lớn.

## Đặc trưng hạng mục có nhiều phần tử phân biệt

Một khó khăn khác khi làm việc với dữ liệu dạng bảng là các đặc trưng hạng mục thường có nhiều giá trị khác nhau.
Một cửa hàng có thể có tới hàng ngàn sản phẩm khác nhau, một hệ thống gợi ý có thể phải phục vụ hàng triệu người dùng với id khác nhau.

Cách truyền thống để biến các đặc trưng hạng mục về dạng số là sử dụng phép biến đổi one-hot (Xem {ref}`sec_one_hot`).
Ở phép biến đổi này, mỗi giá trị của một đặc trưng hạng mục được biến đổi thành một vector có chiều dài bằng số giá trị khác nhau trong đặc trưng đó và chỉ có một phần tử bằng một trong khi khác phần tử còn lại bằng không.
Đây là một cách đơn giản để biến đổi đặc trưng dạng này về số.
Tuy nhiên, phương pháp này có những hạn chế rõ rệt khi số lượng giá trị phân biệt của một hạng mục là cực lớn:

* Vector đặc trưng ở dạng one-hot này cũng sẽ rất lớn. Với các tập dữ liệu có số mẫu nhỏ, số chiều của vector đặc trưng có thể còn lớn hơn số mẫu nhiều lần.
Việc này rất dễ khiến mô hình rơi vào tình trạng quá khớp.

* Vì chỉ có một phần tử bằng một và còn lại bằng không trong mỗi vector one-hot, các vector đặc trưng nhiều khả năng sẽ ở dạng rất thưa trong khi lượng thông tin mang lại không nhiều.
Việc này sẽ có tác động tiêu cực tới chất lượng của mô hình.

* Ở dạng one-hot, khoảng cách (Euclid) giữa hai vector khác nhau bất kỳ luôn bằng $\sqrt{2}$ vì có đúng hai vị trí mà hai vector đó có giá trị khác nhau (0 và 1).
Việc này không mang lại những thông tin quan trọng về sự giống nhau giữa hai giá trị hạng mục khác nhau.

Một cách giải quyết vấn đề này là xây dựng các _embedding vector_ có số chiều nhỏ hơn và "dày đặc" (_dense_) hơn so với các vector one-hot. Kỹ thuật này sẽ được thảo luận kỹ hơn trong {ref}`sec_embedding`.

## Khó áp dụng Transfer Learning

Với dữ liệu ảnh hay văn bản, kể cả khi không có lượng dữ liệu đủ lớn, các kỹ sư ML vẫn có thể tạo ra các mô hình với chất lượng cao dựa trên kỹ thuật _Transfer Learning_ (Học Chuyển Tiếp).
Bạn có thể lấy các bộ phân loại đã được huấn luyện sẵn trên bộ dữ liệu ImageNet như ResNet, DenseNet về làm bộ phân loại chó mèo như một bài tập lớn.
Các bộ phân loại này có thể được sử dụng trực tiếp hoặc _tinh chỉnh_ (fine tuning) để có kết quả tốt hơn.
Với một tác vụ phân loại sắc thái bình luận tiếng Việt, bạn có thể tinh chỉnh [PhoBERT](https://github.com/VinAIResearch/PhoBERT) một chút là đã có kết quả tốt.

Tuy nhiên, dữ liệu dạng bảng không đơn giản như vậy. Hai tập dữ liệu dạng bảng hiếm khi có các trường thông tin giống nhau.
Ngay cả trong tưởng tượng khi Google hoặc Facebook cung cấp bộ dữ liệu cho gợi ý quảng cáo và thậm chí cả thuật toán của họ, việc Cốc Cốc lấy các mô hình này áp dụng trực tiếp vào dữ liệu của họ gần như là không thể.
Chưa kể tới những khác biệt về cơ sở hạ tầng cho việc huấn luyện mô hình, việc Cốc Cốc có một bộ dữ liệu cho các trường thông tin tương tự khó xảy ra.
Học chuyển tiếp trong trường hợp này có thể được áp dụng cho các _kỹ sư_ xây dựng mô hình đó.
Sẽ có rất nhiều kỹ thuật xây dựng đặc trưng mà họ có thể học được trong trường hợp này.
