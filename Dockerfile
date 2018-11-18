FROM jupyter/scipy-notebook:latest

COPY requirements.txt /home/jovyan/requirements.txt

RUN pip install -r /home/jovyan/requirements.txt && \
    pip install --no-cache-dir vdom==0.5
