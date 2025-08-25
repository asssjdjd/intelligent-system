# 1. Chọn Python base image
FROM python:3.11-slim

# 2. Set working directory trong container
WORKDIR /app

# 3. Copy file requirements.txt và cài đặt dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy toàn bộ source code vào container
COPY . .

# 5. Expose port (Flask mặc định là 5000)
EXPOSE 5000

# 6. Chạy Flask app
CMD ["python", "app.py"]