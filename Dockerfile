FROM public.ecr.aws/lambda/python:3.10

# Install minimal system dependencies
RUN yum -y install gcc gcc-c++

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# Streamlit port
EXPOSE 8080

# Streamlit config
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Lambda runtime emulator
ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie /usr/local/bin/aws-lambda-rie
RUN chmod +x /usr/local/bin/aws-lambda-rie

ENTRYPOINT ["/usr/local/bin/aws-lambda-rie", "streamlit", "run", "app.py"]
