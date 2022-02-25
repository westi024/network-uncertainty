FROM tensorflow/tensorflow:2.3.0-gpu

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends graphviz

# Install python requirements
WORKDIR /
# copy the requirements.txt into the container
COPY requirements.txt .
# install the pip packages in requirement.txt with pip
RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir -r requirements.txt
RUN useradd -u 1000 user && echo "user:user" | chpasswd && adduser user sudo