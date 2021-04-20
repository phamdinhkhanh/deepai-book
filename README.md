# Machine Learning Algorithms to Practice

## Về dự án 

Đây là dự án viết sách cộng đồng thử nghiệm lần đầu tiên tại Việt Nam. Việc viết sách thì đi ngược lại quá trình của dịch sách. Tức là chúng ta phải nghĩ ra nội dung và gắn kết nội dung. Xuất phát là một người viết blog về AI, với mong muốn đóng góp **vì một cộng đồng AI vững mạnh hơn**. Mình quyết định tổng hợp những bài viết trên blog của mình thành một cuốn sách với tựa đề **Machine Learning Algorithms to Practice**.

Mục tiêu của cuốn sách **Machine Learning Algorithms to Practice** đó là hướng dẫn một người mới bắt đầu tiếp cận AI có thể tự đọc hiểu thuật toán và áp dụng được vào công việc. Như vậy cuốn sách này sẽ cân bằng giữa lý thuyết và thực hành.

## Đóng góp vào dự án

Sẽ có hai mảng chính mà một người có thể đóng góp vào dự án.

- Nghiên cứu: Mục tiêu của nghiên cứu là viết về lý thuyết. Bạn sẽ đăng ký một mục cụ thể ở check list (trừ các chương 1, 2 vì quá đơn giản). Có thể đã trùng với một người khác đã đăng ký cũng được vì chúng ta chấp nhận nhiều bản thảo, cách trình bày và bố cục. Hoặc bạn có thể tự đề xuất viết về một đề tài cụ thể và tự viết. Hình thức này sẽ phù hợp với các bạn đam mê nghiên cứu và có kiến thức nền tảng tốt về machine learning và deep learning. Việc tham gia viết lách sẽ giúp bạn rèn luyện được tư duy nghiên cứu và khả năng diễn giải.

- Ứng dụng: Bạn sẽ tham gia phát triển các case studies dựa trên một thuật toán cụ thể liên quan tới nội dung cuốn sách hoặc tự đề xuất những case studies mà bạn nghĩ đó là quan trọng. Các bạn cũng có thể nhận ý kiến tham vấn từ tác giả, dựa trên sự chia sẻ kinh nghiệm từ tác giả để hoàn thiện kỹ năng xây dựng mô hình của mình.

Sau khi hoàn thiện bài viết, bạn sẽ gửi đến dự án thông qua một trong hai cách bên dưới. Phần đóng góp của bạn sẽ được publish ở phần mở rộng của cuốn sách (mục BÀI VIẾT ĐÓNG GÓP) nhưng không nhất thiết nó sẽ được đưa vào cuốn sách. Điều đó phụ thuộc vào chất lượng bài viết và mức độ phù hợp của nó với nội dung cuốn sách. Nhưng bạn yên tâm là nó sẽ nằm ở phần BÀI VIẾT ĐÓNG GÓP nhé, và cộng đồng sẽ vẫn biết đến nội dung bạn viết.

## Cách gửi bài viết

Bạn có thể đóng góp theo hai cách:

* **Cách 1**: Tạo một pull request cho cuốn sách. Tiêu đề sẽ có dạng: `Tên mục - hovaten`. Ví dụ bạn viết về  `Bayes inference` và có nickname là `nguyenva` thì có thể để tiêu đề là `4.4 Bayes Inference - nguyenva`.

Để có thể tạo pull request thì bạn cần thực hiện theo các bước:

1. Clone repository của cuốn sách về. 
2. Tạo một branch với tên `chuong3/bayes_inference_nguyenva` và thực hiện thay đổi trên branch. 
3. git push branch của bạn.
4. Vào mục pull request của repository của tác và create pull request với tiêu đề `4.4 Bayes Inference - nguyenva`.

Bạn có thể theo dõi hướng dẫn bên dưới về [pull request](https://www.youtube.com/watch?v=MVGgNteyflw).

* **Cách 2**: Bạn gửi trực tiếp note book của bạn cho tác giả theo facebook: [phamdinhkhanh](https://www.facebook.com/langnhin.anhtrang)


## Cách lựa chọn đề tài phù hợp

Các mục trong cuốn sách sẽ phân cấp về nội dung, từ dễ, trung bình tới khó. Do đó hướng dẫn này sẽ là gợi ý tốt để bạn lựa chọn một đề tài phù hợp với khả năng theo các cấp độ:

* Dễ: Là các hướng dẫn về cách sử dụng module, package và cấu trúc cơ sở dữ liệu như list, tuple, dictionary, set,.... Phù hợp với những bạn mới bắt đầu.
* Trung bình: Các case studies xây dựng mô hình. Các chương không liên quan tới lý thuyết về thuật toán. Phần này sẽ phù hợp với những bạn theo hướng thực hành.
* Khó: Các lý thuyết về mô hình, thuật toán. Sẽ yêu cầu các bạn có background mạnh về toán, thống kê và kinh nghiệm nghiên cứu.

Bạn đọc có thể ước lượng khả năng của mình và tìm ra mục phù hợp để viết.

## Ngôn ngữ và frameworks chính

Trong dự án này, chúng ta sẽ ưu tiên sử dụng python3 là ngôn ngữ chính. Các mô hình deep learning được khuyến khích huấn luyện trên framework pytorch. Trong trường hợp bạn sử dụng ngôn ngữ R hoặc các frameworks deep learning khác như tensorflow hoặc mxnet thì cũng có thể được chấp nhận. Nhưng sẽ tốt hơn là chỉ sử dụng thống nhất một ngôn ngữ và frameworks để đỡ tốn thời gian tích hợp.

## Qui ước chung khi viết bài

Để thống nhất chung trên toàn bộ các chương. Khi viết bài bạn cần tuân theo chuẩn [latex](https://github.com/phamdinhkhanh/deepai-book/blob/main/book/latex.md) và cập nhật các thuật ngữ vào [bảng thuật ngữ](https://github.com/phamdinhkhanh/deepai-book/blob/main/book/grossary.md).

## Cách viết 

Đầu tiên bạn chọn đề tài của mình sẽ viết trong [check list - Machine Learning Algorithms to Practice](https://docs.google.com/spreadsheets/d/1cv1NmjZayeA7nlhKR8PVRsyAmfU3gEy8yIWLPg1H78Y/edit?usp=sharing). Sau khi lựa chọn xong bạn nhớ điền thông tin của mình nhé. Tiếp theo bạn sẽ khảo cứu tài liệu và tiến hành viết. Một bài viết chất lượng là một bài viết:

- Không mắc lỗi chính tả, không lỗi latex.
- Có các hình minh hoạ diễn giải nội dung.
- Các mục được sắp đặt hợp lý.
- Được tham khảo cẩn thận từ nhiều nguồn tài liệu tin cậy.
- Có code thực hành.
- Bài tập để hiểu nội dung hơn.
- Tài liệu đã tham khảo.

## Lời cảm ơn

Tác giả xin chân thành cảm ơn sự đóng góp từ cộng đồng để hoàn thiện nội dung cuốn sách. Mỗi một đóng góp nhỏ bé sẽ là một mảnh ghép để góp phần **vì một cộng đồng AI vững mạnh hơn**.