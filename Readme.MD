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
python NLPRoleRecognition.py --data DATA --option OPTION [--optionType OPTIONTYPE]
```

### Bảng mô tả OPTION
| Option                         |Mô Tả                                                                                                                                                                 |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Train                          | Train model với data train trong data folder																															|
| Normalize                      | Load data from Parquet files and store into central data for train model later																						|
| Predict                        | Predict role trong file hội thoại với model đã được train																											|

### Bảng mô tả OPTIONTYPE (cho OPTION: Train)
| Option Type                    |Mô Tả                                                                                                                                                                 |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TF-IDF                         | Train model với TF-IDF và Logistic Regression																														|
| PhoBert                        | Train model với PhoBert (finetune PhoBert cho bài toán Role Recognition)																								|
| Context                        | Train model với BiLSTM + Linear. Dùng PhoBert để trích vector																										|

### Bảng mô tả OPTIONTYPE (cho OPTION: Normalize)
| Option                         |Mô Tả                                                                                                                                                                 |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data folder                    | Thư mục chứa toàn bộ cái tập tin Parquet mà NLPRoleRecognition sẽ dùng để chuẩn hóa data vào trong database trung tâm												|

### Bảng mô tả OPTIONTYPE (cho OPTION: Predict)
| Option                         |Mô Tả                                                                                                                                                                 |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data file                      | Tập tin dạng .csv hoặc .txt chứa đoạn hội thoại mà NLPRoleRecognition cần dự đoán role																				|

```bash
python NLPRoleRecognition.py --option "Normalize" --optionType "./data"

HOẶC

python NLPRoleRecognition.py --option "Train" --optionType "TF-IDF"

HOẶC

python NLPRoleRecognition.py --option "Predict" --optionType "./data/predict.csv"
```