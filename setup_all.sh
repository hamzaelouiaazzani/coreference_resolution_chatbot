#!/bin/bash

# Download pretrained embeddings.
curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz
gunzip -d cc.fr.300.vec.gz

# Build custom kernels.
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# Linux (pip)
g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0
