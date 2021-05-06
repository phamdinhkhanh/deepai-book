---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.8.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++ {"id": "dxuQMYAKo2c1"}

# 2. Pandas

Pandas là một package rất hiệu quả khi làm việc với dữ liệu dạng bảng. Nó cho phép chúng ta thực hiện các phép biến đổi và thống kê trên dữ liệu dạng bảng với tốc độ rất nhanh. Nhờ những hàm tiện ích trong hệ sinh thái của pandas mà chúng ta có thể liên kết các bảng có quan hệ một cách dễ dàng. Việc biểu đồ hoá trên pandas cũng được triển khai hiệu quả nhờ tích hợp được đa dạng những biểu đồ cơ bản trong matplotlib. Với pandas, bạn có thể đọc dữ liệu từ đa dạng các định dạng từ phổ biến đến hiếm gặp như: csv, txt, xlsx, hdf5, json, dat, SQL table.Việc truy vấn dữ liệu của pandas cũng gần gũi như trên numpy nên rất dễ học và dễ nhớ. Không những thế những hàm xử lý missing data và sắp xếp dữ liệu của pandas giúp quá trình tiền xử lý dữ liệu nhanh chóng hơn. Pandas cũng là một package được tích hợp với các hàm về chuỗi thời gian nên nó được sử dụng rộng rãi trong quantitative finance.

Nhờ những tiện ích đó mà pandas đã trở thành một trong những package được sử dụng phổ biến nhất đối với data science. Hay nói theo một khía cạnh khác, nếu chúng ta muốn trở thành một data scientist giàu kinh nghiệm thì việc thành thạo pandas dường như là một điều bắt buộc.

Để nói về pandas là một đề tài rất rộng nên bài viết này không nhằm bao quát hết mọi thứ về pandas. Mục tiêu của bài viết là đưa ra hướng dẫn cho bạn đọc để giúp nắm được những chức năng chính và hiểu cách áp dụng nó như thế nào trên những ví dụ cụ thể. Bài viết đồng thời là một nguồn tài liệu tham khảo và tra cứu khi cần cho data scientist khi xử lý dữ liệu.

+++ {"id": "w3D4Ov-WgBtm"}

