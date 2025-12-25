# Unit Tests

Thư mục này chứa các unit tests cho dự án DeepFake Detection.

## Cấu trúc

- `test_utils.py`: Tests cho `src/utils/utils.py`
- `test_dataset.py`: Tests cho `src/training/dataset.py`
- `test_app.py`: Tests cho `src/app/main_app.py`

## Chạy tests

### Chạy tất cả tests:
```bash
python -m pytest tests/
```

### Chạy một file test cụ thể:
```bash
python -m pytest tests/test_utils.py
```

### Chạy với coverage:
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Yêu cầu

Cài đặt pytest (nếu chưa có):
```bash
pip install pytest pytest-cov
```

## Ghi chú

- Tests sử dụng temporary directories để không ảnh hưởng đến dữ liệu thực
- Một số tests có thể cần GPU để chạy đầy đủ (có thể skip nếu không có GPU)

