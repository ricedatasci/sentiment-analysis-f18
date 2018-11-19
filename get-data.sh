#!/usr/bin/env bash

mkdir -p data && \
    pushd data && \
    wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz && \
    wget https://github.com/abhmul/DataScienceTrack/raw/master/NLP/imdb_dataset.zip && \
    echo "Extracting data..." && \
    tar zxf aclImdb_v1.tar.gz && \
    echo "Done!" && \
    popd
