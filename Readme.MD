# Dự án Role Recognition trong hội thoại tiếng Việt
- Môn học: NLP và ứng dụng
- Khóa: K35
- Khoa: Khoa Học Tích Hợp
- Học Viên:
    > - Đinh Trịnh Thanh Vy: 25C02035
    > - Trần Như Linh: 25C02020
    > - Lê Ngọc Minh Thư: 25C02006
    > - Đỗ Mạnh Hoàng: 25C02032

## Cài đặt môi trường
Các thư viện cần thiết để chạy script được mô tả trong file: python_require_libraries.txt. Chạy dòng lệnh sau để cài đặt các thư viện

```bash
pip install -r python_require_libraries.txt
```

## Cách chạy script

```bash
python NLPRoleRecognition.py <option> <option type>
```

### Bảng mô tả option
| Option                         |Mô Tả                                                                                                                                                                 |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| train                          | Train model với data train trong data folder																															|
| predict                        | Predict role trong file hội thoại với model đã được train																											|

### Bảng mô tả option type (train)
| Option                         |Mô Tả                                                                                                                                                                 |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TF-IDF                         | Train model với TF-IDF và Logistic Regression																														|
| PhoBert                        | Train model với PhoBert (finetune PhoBert cho bài toán Role Recognition)																								|
| Context                        | Train model với BiLSTM + Linear. Dùng PhoBert để trích vector																										|

### Bảng mô tả option type (predict)
Mẫu câu lệnh sẽ như sau với hoi_thoai.txt là tập tin chứa các đoạn hội thoại

```bash
python NLPRoleRecognition.py predict hoi_thoai.txt
```