# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

<!--
HƯỚNG DẪN - đọc rồi XÓA TOÀN BỘ các khối chú thích này sau khi điền xong:

  - Giới hạn: KHÔNG QUÁ 1 TRANG A4, tương đương khoảng 450 - 550 từ nội dung.
  - Chỉ điền vào các chỗ ___ và các ô trong bảng. Không thêm mục mới.
  - Viết bằng câu hoàn chỉnh, không gạch đầu dòng cụt lủn.
  - Kiểm tra độ dài sau khi đã xóa hết chú thích:
        wc -w nop-bai/bao-cao.md
    và xem trước bản in bằng cách mở file trên GitHub rồi Ctrl+P / Cmd+P.
-->

| | |
|---|---|
| Họ và tên | ___ |
| MSSV | ___ |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/___/___ |
| Ngày nộp | ___ |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

<!-- Khoảng 120 - 150 từ. Điền kết quả thật từ MLflow UI ở Bước 1, tối thiểu 3 lần chạy. -->

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 2 | 100 | 0.10 | 3 | 0.7109 | 0.8780 |
| 3 | 100 | 0.20 | 3 | 0.7290 | 0.8840 |

**Bộ siêu tham số đã chọn:** `n_estimators=100`, `learning_rate=0.2`, `max_depth=3`.

**Lý do:** Đây là cấu hình có F1 lớp dương cao nhất trong năm lần chạy (0.7290), đồng
thời vượt ngưỡng triển khai 0.65. Trong các thử nghiệm này, cấu hình có accuracy cao nhất
cũng có F1 cao nhất, nhưng quyết định vẫn dựa trên F1 vì đây là chỉ số phản ánh trực tiếp
khả năng nhận diện lớp thu nhập cao. Cấu hình 200 cây, độ sâu 5 chỉ đạt F1 0.7149, cho
thấy tăng độ phức tạp không đảm bảo kết quả tốt hơn. Khi quét ngưỡng quyết định, ngưỡng
0.30 nâng F1 của cấu hình đã chọn từ 0.7290 lên 0.7519.

<!--
Trả lời trong phần Lý do:
  - Vì sao bộ này tốt hơn các bộ còn lại (dựa trên f1_score, không phải accuracy)?
  - Lần chạy có accuracy cao nhất có trùng với lần có f1_score cao nhất không?
    Nếu không, điều đó nói lên điều gì?
  - Bạn quan sát thấy đánh đổi nào giữa n_estimators và learning_rate?
-->

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

<!-- Khoảng 120 - 150 từ. -->

Dữ liệu chỉ có khoảng 24,8% mẫu thuộc lớp thu nhập trên 50K. Một mô hình luôn dự đoán
"thu nhập thấp" vẫn đạt accuracy xấp xỉ 75,2%, dù không phát hiện được trường hợp thu
nhập cao nào và có F1 lớp dương bằng 0. F1 là trung bình điều hòa của precision và recall,
nên chỉ cao khi mô hình vừa hạn chế gán nhầm vừa không bỏ sót quá nhiều mẫu của lớp
dương. Vì mục tiêu của quality gate là bảo vệ chất lượng nhận diện lớp thu nhập cao, lab
dùng `f1_score(y_eval, preds)` với lớp dương mặc định. Không dùng `average="weighted"`
vì lớp đa số sẽ chi phối kết quả; cũng không dùng `average="macro"` vì ngưỡng 0.65 của
lab được định nghĩa riêng cho lớp dương.

<!--
Cần nêu được:
  - Phân bố lớp của tập dữ liệu (tỷ lệ lớp thu nhập > 50K) và hệ quả của nó.
  - Accuracy của một mô hình luôn trả lời "thu nhập thấp" là bao nhiêu, vì sao con số
    đó gây hiểu nhầm.
  - F1 của lớp dương đo điều gì mà accuracy không đo được.
  - Vì sao KHÔNG dùng average="weighted" hay average="macro" khi gọi f1_score.
-->

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

<!-- Nêu 2 - 3 khó khăn thật, mỗi ô một câu ngắn. -->

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow lỗi thiếu `pkg_resources` | Setuptools mới đã bỏ API cũ mà MLflow 2.13 vẫn dùng | Pin `setuptools==70.0.0` trong requirements |
| DVC không ghi được `/Library/Caches/dvc` | DVC 3.50 chọn system site cache trên macOS | Đặt `core.site_cache_dir=.dvc/tmp/site-cache` |
| GCP cấm tạo service-account key JSON | Project bật policy `iam.disableServiceAccountKeyCreation` | Dùng ADC local, service account gắn VM và Workload Identity Federation cho CI |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

<!-- Lấy số liệu từ bảng ở mục 3.6 của tasks/buoc-3.md. -->

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | ___ | ___ |
| Bước 3 (thêm `train_batch2`) | ___ | ___ |

**Nhận xét:** ___

<!--
Một câu trả lời trung thực kiểu "f1 giảm 0,01 vì dữ liệu mới cùng phân phối, không mang
thêm thông tin mới" được đánh giá cao hơn kết luận sai rằng thêm dữ liệu luôn tốt hơn.
-->

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

<!-- Xóa cả mục 5 nếu không làm bonus. Mỗi bonus tối đa 1 dòng. -->

- [x] Bonus 1 - Tracking MLflow từ xa với DagsHub: run CI đã được ghi lên DagsHub.
- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: ngưỡng 0.30 nâng F1 từ 0.7290 lên 0.7519.
- [x] Bonus 3 - Báo cáo precision / recall tự động: đã tạo `outputs/detail.txt`.
- [x] Bonus 4 - Hoàn trả về phiên bản trước: pipeline chặn candidate có F1 thấp hơn production.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: tỷ lệ lớp dương 24,77%, không phát cảnh báo.
