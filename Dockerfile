# Use a base image with CUDA and cuDNN support
FROM tensorflow/tensorflow:2.19.0-gpu

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . /app

# Set environment variables
ENV TF_ENABLE_ONEDNN_OPTS=0
