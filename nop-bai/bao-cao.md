# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Đặng Hữu Khanh |
| MSSV | 2A202601104 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/HuuKhanh19/K4-Track2-Day21-DangHuuKhanh-01104 |
| Ngày nộp | 21/08/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---:|---:|---:|---:|---:|
| 1 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 2 | 100 | 0.10 | 3 | 0.7109 | 0.8780 |
| 3 | 100 | 0.20 | 3 | 0.7290 | 0.8840 |

**Bộ siêu tham số đã chọn:** `n_estimators=100`, `learning_rate=0.2`, `max_depth=3`.

**Lý do:** Cấu hình có F1 lớp dương cao nhất (0.7290) và vượt ngưỡng 0.65. Lần chạy
có accuracy cao nhất cũng có F1 cao nhất, nhưng quyết định vẫn dựa trên khả năng
nhận diện lớp thu nhập cao. Cấu hình 200 cây, độ sâu 5 chỉ đạt F1 0.7149, cho thấy tăng
độ phức tạp không bảo đảm tốt hơn. Learning rate thấp thường cần nhiều cây để bù lại.
Ngưỡng quyết định 0.30 tiếp tục nâng F1 từ 0.7290 lên 0.7519.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Chỉ 24,8% mẫu thuộc lớp thu nhập trên 50K. Mô hình luôn dự đoán "thu nhập thấp" vẫn
đạt accuracy 75,2% nhưng không phát hiện mẫu dương nào, nên F1 lớp dương bằng 0. F1
kết hợp precision và recall, vì vậy phản ánh cả gán nhầm lẫn bỏ sót. Quality gate dùng
`f1_score(y_eval, preds)` cho lớp dương. Không dùng `average="weighted"` vì lớp đa số
chi phối kết quả, cũng không dùng `average="macro"` vì ngưỡng 0.65 được định nghĩa
riêng cho lớp dương.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow thiếu `pkg_resources` | Setuptools mới bỏ API cũ | Pin `setuptools==70.0.0` |
| DVC lỗi cache hệ thống | DVC chọn system cache trên macOS | Đặt cache tại `.dvc/tmp/site-cache` |
| GCP cấm tạo key JSON | Policy bảo mật của project | Dùng ADC, service account gắn VM và WIF |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---:|---:|
| Bước 2 (chỉ `train_batch1`) | 0.7290 | 0.8840 |
| Bước 3 (thêm `train_batch2`) | 0.7330 | 0.8820 |

**Nhận xét:** Khi dữ liệu tăng từ 22.361 lên 44.722 mẫu, F1 tăng 0.0041 còn accuracy
giảm 0.0020. Dao động nhỏ là hợp lý vì batch mới có cùng nguồn và phân phối. Thành quả
chính là DVC, huấn luyện, quality gate và triển khai đều tự chạy từ một commit dữ liệu.

---

## 5. Phần Bonus Đã Thực Hiện

- [x] Bonus 1 - Các run CI được ghi lên DagsHub.
- [x] Bonus 2 - Điều chỉnh ngưỡng: ngưỡng 0.30 nâng F1 từ 0.7290 lên 0.7519.
- [x] Bonus 3 - Pipeline tạo báo cáo `outputs/detail.txt`.
- [x] Bonus 4 - Bảo vệ production: candidate F1 0.6051 bị chặn, Release không chạy.
- [x] Bonus 5 - Bước 3 có 24,78% dương, không drift.
