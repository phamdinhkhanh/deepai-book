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

# Latex

Trong cuốn sách này, tác giả sẽ trình bày các ký hiệu toán học thống nhất giữa các chương như sau:

| Ký hiệu      | latex | Định dạng | Ý nghĩa |
| ----------- | ----------- | ----------- | ----------- |
| $x, y, N, k$      | `$x, y, N, k$` | chữ thường hoặc chữ hoa viết thường      | số vô hướng |
| $\mathbf{x}, \mathbf{y}$   | `$\mathbf{x}, \mathbf{y}$` | chữ thường in đậm        | véc tơ |
| $\mathbf{A}, \mathbf{B}$   | `$\mathbf{A}, \mathbf{B}$`| chữ hoa in đậm        | ma trận |
| $\mathbb{R}, \mathbb{N}$ |  `$\mathbb{R}, \mathbb{N}$`  | chữ hoa, nét đôi        | tập số thực, số nguyên,... |
| $\mathbb{R}^{m \times n}$ | `$\mathbb{R}^{m \times n}$`  | chữ hoa, nét đôi        | không gian ma trận số thực $m \times n$ |
| $\mathcal{L}()$ | `$\mathcal{L}()$`  | chữ hoa, nét thanh in đậm        | hàm loss function |
| $\mathbf{P}()$ | `$\mathbf{P}()$`  | chữ hoa, nét đậm        | xác suất |
| $\mathbf{E}(\mathbf{x})$ |  `$\mathbf{E}(\mathbf{x})$` | chữ hoa, nét đậm | kỳ vọng của véc tơ hoặc ma trận |
| $\mu, \sigma, \lambda$ | `$\mu, \sigma, \lambda$`  | chữ cái latin thường | tham số phân phối xác suất |
| $\alpha$ | `$\alpha$`  | chữ cái `alpha` thường | learning rate của mô hình |
| $\in$  | `$\in$` | phần tử thuộc tập hợp |
| $\subseteq$   | `$\subseteq$`       | tập con thuộc tập hợp |
| $\nsubseteq$   | `$\nsubseteq$` | không phải là tập con thuộc tập hợp |
| $\forall$ | `$\forall$` | với mọi phần tử , thường được dùng sau một khẳng định |
| $\exists$ | `$\exists$`| tồn tại |
| $\triangleq$ | `$\triangleq$` | đặt | $f(x) \triangleq 2x+1 $, tức là đặt $f(x)$ bằng $2x+1$, thường sử dụng lần đầu tiên định nghĩa $f(x)$ |
| $x_i$ | `$x_i$` | phần tử thứ $i$ của véc tơ $\mathbf{x}$ |
| $\exp(x)$ | `$\exp(x)$` | số mũ cơ số tự nhiên $e$ | $e^{x}$ |
| $\log(x)$ | `$\log(x)$` | logarith cơ số tự nhiên $e$, $e = \lim_{n \rightarrow +\infty}{(1+\frac{1}{n})}^{n}$ | logarith cơ số tự nhiên $e$ của $x$|
| $a_{ij}$ | `$a_{ij}$` | phần tử thuộc dòng $i$ cột $j$ của ma trận $\mathbf{A}$ |
| $\mathbf{X}^{\intercal}$ | `$\mathbf{X}^{\intercal}$` | ma trận chuyển vị của $\mathbf{X}$ |
| $\mathbf{X}^{-1}$ | `$\mathbf{X}^{-1}$` | ma trận nghịch đảo $\mathbf{X}$ |
| $\det(\mathbf{X})$ |`$\det(\mathbf{X})$` | định thức ma trận $\mathbf{X}$ |
| $\text{rank}(\mathbf{X})$ | `$\text{rank}(\mathbf{X})$` | rank ma trận $\mathbf{X}$ |
| $\Vert\mathbf{x}\Vert_{p}$ | `$\Vert\mathbf{x}\Vert$` | norm chuẩn bậc $p$ của véc tơ $\mathbf{x}$ |
| $\mathbf{I}_n$ | `$\mathbf{I}_n$` | ma trận đơn vị kích thước $n \times n$ |
|$\frac{d(f(x))}{dx}$ | `$\frac{d(f(x))}{dx}$` | đạo hàm với hàm một biến, chẳng hạn $f(x) = 2x+1$ |
|$\frac{\delta(f(x))}{\delta{x}}$| `$\frac{\delta(f(x))}{\delta{x}}$` | Đạo hàm của hàm nhiều biến, chẳng hạn $f(x_1, x_2) = x_1+2x_2+1$ |
|$\nabla_{\mathbf{x}} f(\mathbf{x})$| `$\nabla_{\mathbf{x}} f(\mathbf{x})$` | gradient descent của hàm $f$ theo véc tơ $\mathbf{x}$ |
|$\nabla_{\mathbf{x}}^{2} f(\mathbf{x})$| `\nabla_{\mathbf{x}}^{2} f(\mathbf{x})` | đạo hàm bậc hai của hàm $f$ theo véc tơ hoặc ma trận $\mathbf{x}$ |
| $\propto$ | `$\propto$` | ký hiệu đồng dạng giữa hai phân phối, ví dụ $\mathbf{P}(y\|D)) \propto \mathbf{P}(D\|y)\mathbf{P}(y)$ |
| $\odot$ |`$\odot$`| tích hardamard hoặc element-wise giữa hai ma trận hoặc véc tơ có cùng kích thước |
|$\langle \mathbf{x}, \mathbf{y} \rangle = \sum_{i=1}^{d} x_i y_i$| `\langle \mathbf{x}, \mathbf{y} \rangle = \sum_{i=1}^{d} x_i y_i` | tích vô hướng của hai véc tơ |

# Tham khảo latex

Bạn có thể tham khảo cách gõ latex tại [wikibooks - latex](https://en.wikibooks.org/wiki/LaTeX/Mathematics) (chứa các bảng ký hiệu `latex`), tại [rpub - phamdinhkhanh - latex basical](https://rpubs.com/phamdinhkhanh/408217) (chứa các công thức `latex` thông dụng) và [các ký hiệu latex](https://www.caam.rice.edu/~heinken/latex/symbols.pdf).

Khi bạn quên một ký hiệu `latex`, bạn có thể vào link [wikibooks - latex](https://en.wikibooks.org/wiki/LaTeX/Mathematics) và sử dụng `ctr+F` để tìm kiếm.

* Nếu đó là các ký hiệu so sánh ($>, <, \neq, \succeq, \propto, \sim, \dots$), search `Relation Symbols`.

* Nếu đó là các toán tử ($\odot, \oplus, \ominus
$), search `Binary Operations
`.

* Nếu đó là ký hiệu logic ($\implies, \rightarrow, \mapsto, \exists, \forall, \dots $), search `Logic Notation`.

* Chữ cái Hi Lạp, search `Greek Letters`.

* Hàm lượng giác, search `Trigonometric Functions`.