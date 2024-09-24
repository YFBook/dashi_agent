FROM python:3.11.4

WORKDIR /ai_server

COPY requirements.txt .
RUN pip install -r requirements.txt 


# 拷贝代码到容器
COPY . .

# # 启动命令
# CMD ["python","server.py"]