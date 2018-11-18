#!/usr/bin/env bash

mkdir -p data && \
    pushd data && \
    curl -O http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz && \
    echo "Extracting data..." && \
    tar zxf aclImdb_v1.tar.gz && \
    echo "Done!" && \
    popd
