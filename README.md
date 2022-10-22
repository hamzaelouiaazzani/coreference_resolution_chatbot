# **Presentation**

This notebook presents an application of the pre-trained [COFR system](https://hal.archives-ouvertes.fr/hal-02476902/document) of [Mr. Bruno Oberle](https://boberle.com/projects/coreference-resolution-with-cofr/) with many modifications to make it more suitable for deployment in a chatbot. Most of the files in the repository are modified to reduce memory consumption and the system's runtime. To run this repository, open the notebook coreference_resolution.ipynb in the repository.

## **1.   Preparing environnement and dependencies**

**If you want to use the RASA framework to deploy your coreference resolution model in a chatbot, you should create a new virtual environment for this project to avoid dependency conflicts (the model in this project is built with Tensorflow v1, whereas the RASA framework for chatbots only supports Tensorflow v2). The [automatic immigration from Tensorflow V1 to V2](https://www.tensorflow.org/guide/migrate) doesn't work for this code, so the only solution is to create two separate virtual environments; one for your coreference resolution model Tensorflow 1 and another for your RASA framework Tensorflow 2}.**


```python
%cd /content/drive/MyDrive/coreference_resolution_chatbot 
#your working folder
```

    /content/drive/MyDrive/coreference_resolution_chatbot
    



*   Check this link if you want to learn how to [create virtual environnements with conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) 



```python
!pip3 install -r requirements.txt
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting tensorflow==1.15.2
      Downloading tensorflow-1.15.2-cp37-cp37m-manylinux2010_x86_64.whl (110.5 MB)
    [K     |████████████████████████████████| 110.5 MB 51 kB/s 
    [?25hCollecting numpy==1.19.1
      Downloading numpy-1.19.1-cp37-cp37m-manylinux2010_x86_64.whl (14.5 MB)
    [K     |████████████████████████████████| 14.5 MB 48.2 MB/s 
    [?25hRequirement already satisfied: tensorflow-hub>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 3)) (0.12.0)
    Requirement already satisfied: h5py in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 4)) (3.1.0)
    Requirement already satisfied: nltk in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 5)) (3.7)
    Collecting pyhocon
      Downloading pyhocon-0.3.59.tar.gz (116 kB)
    [K     |████████████████████████████████| 116 kB 82.4 MB/s 
    [?25hRequirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 7)) (1.7.3)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 8)) (4.64.1)
    Collecting colorama
      Downloading colorama-0.4.5-py2.py3-none-any.whl (16 kB)
    Collecting stanza
      Downloading stanza-1.4.2-py3-none-any.whl (691 kB)
    [K     |████████████████████████████████| 691 kB 71.6 MB/s 
    [?25hCollecting stanfordnlp
      Downloading stanfordnlp-0.2.0-py3-none-any.whl (158 kB)
    [K     |████████████████████████████████| 158 kB 72.6 MB/s 
    [?25hRequirement already satisfied: astor>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.2->-r requirements.txt (line 1)) (0.8.1)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.2->-r requirements.txt (line 1)) (1.15.0)
    Requirement already satisfied: absl-py>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.2->-r requirements.txt (line 1)) (1.3.0)
    Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.2->-r requirements.txt (line 1)) (3.3.0)
    Requirement already satisfied: keras-preprocessing>=1.0.5 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.2->-r requirements.txt (line 1)) (1.1.2)
    Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.2->-r requirements.txt (line 1)) (2.0.1)
    Collecting tensorboard<1.16.0,>=1.15.0
      Downloading tensorboard-1.15.0-py3-none-any.whl (3.8 MB)
    [K     |████████████████████████████████| 3.8 MB 69.0 MB/s 
    [?25hRequirement already satisfied: protobuf>=3.6.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.2->-r requirements.txt (line 1)) (3.17.3)
    Collecting keras-applications>=1.0.8
      Downloading Keras_Applications-1.0.8-py3-none-any.whl (50 kB)
    [K     |████████████████████████████████| 50 kB 7.0 MB/s 
    [?25hCollecting gast==0.2.2
      Downloading gast-0.2.2.tar.gz (10 kB)
    Requirement already satisfied: wrapt>=1.11.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.2->-r requirements.txt (line 1)) (1.14.1)
    Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.2->-r requirements.txt (line 1)) (0.37.1)
    Requirement already satisfied: google-pasta>=0.1.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.2->-r requirements.txt (line 1)) (0.2.0)
    Collecting tensorflow-estimator==1.15.1
      Downloading tensorflow_estimator-1.15.1-py2.py3-none-any.whl (503 kB)
    [K     |████████████████████████████████| 503 kB 75.9 MB/s 
    [?25hRequirement already satisfied: grpcio>=1.8.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.2->-r requirements.txt (line 1)) (1.49.1)
    Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<1.16.0,>=1.15.0->tensorflow==1.15.2->-r requirements.txt (line 1)) (57.4.0)
    Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard<1.16.0,>=1.15.0->tensorflow==1.15.2->-r requirements.txt (line 1)) (1.0.1)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard<1.16.0,>=1.15.0->tensorflow==1.15.2->-r requirements.txt (line 1)) (3.4.1)
    Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard<1.16.0,>=1.15.0->tensorflow==1.15.2->-r requirements.txt (line 1)) (4.13.0)
    Requirement already satisfied: typing-extensions>=3.6.4 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<1.16.0,>=1.15.0->tensorflow==1.15.2->-r requirements.txt (line 1)) (4.1.1)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<1.16.0,>=1.15.0->tensorflow==1.15.2->-r requirements.txt (line 1)) (3.9.0)
    Requirement already satisfied: cached-property in /usr/local/lib/python3.7/dist-packages (from h5py->-r requirements.txt (line 4)) (1.5.2)
    Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.7/dist-packages (from nltk->-r requirements.txt (line 5)) (2022.6.2)
    Requirement already satisfied: click in /usr/local/lib/python3.7/dist-packages (from nltk->-r requirements.txt (line 5)) (7.1.2)
    Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (from nltk->-r requirements.txt (line 5)) (1.2.0)
    Collecting pyparsing~=2.0
      Downloading pyparsing-2.4.7-py2.py3-none-any.whl (67 kB)
    [K     |████████████████████████████████| 67 kB 6.9 MB/s 
    [?25hRequirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from stanza->-r requirements.txt (line 10)) (2.23.0)
    Requirement already satisfied: torch>=1.3.0 in /usr/local/lib/python3.7/dist-packages (from stanza->-r requirements.txt (line 10)) (1.12.1+cu113)
    Collecting emoji
      Downloading emoji-2.1.0.tar.gz (216 kB)
    [K     |████████████████████████████████| 216 kB 82.7 MB/s 
    [?25hRequirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->stanza->-r requirements.txt (line 10)) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->stanza->-r requirements.txt (line 10)) (2022.9.24)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->stanza->-r requirements.txt (line 10)) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->stanza->-r requirements.txt (line 10)) (1.24.3)
    Building wheels for collected packages: gast, pyhocon, emoji
      Building wheel for gast (setup.py) ... [?25l[?25hdone
      Created wheel for gast: filename=gast-0.2.2-py3-none-any.whl size=7554 sha256=3f2ada27f5c857fafcbc046feddb213ac0bbb346eeccdcee8500da325f83098d
      Stored in directory: /root/.cache/pip/wheels/21/7f/02/420f32a803f7d0967b48dd823da3f558c5166991bfd204eef3
      Building wheel for pyhocon (setup.py) ... [?25l[?25hdone
      Created wheel for pyhocon: filename=pyhocon-0.3.59-py3-none-any.whl size=19968 sha256=949c86e0a069f86a996683cee999bc3e53ea72f4e9da99e19e3a1a0e3739498b
      Stored in directory: /root/.cache/pip/wheels/69/f4/73/2afd609de4a040ee997bb9026d145e626124bc3ce8e5c23f79
      Building wheel for emoji (setup.py) ... [?25l[?25hdone
      Created wheel for emoji: filename=emoji-2.1.0-py3-none-any.whl size=212392 sha256=b498b6a69e7b02966d531e77b8ec33babdd52f714671dba01597d2b217c18bd9
      Stored in directory: /root/.cache/pip/wheels/77/75/99/51c2a119f4cfd3af7b49cc57e4f737bed7e40b348a85d82804
    Successfully built gast pyhocon emoji
    Installing collected packages: numpy, tensorflow-estimator, tensorboard, pyparsing, keras-applications, gast, emoji, tensorflow, stanza, stanfordnlp, pyhocon, colorama
      Attempting uninstall: numpy
        Found existing installation: numpy 1.21.6
        Uninstalling numpy-1.21.6:
          Successfully uninstalled numpy-1.21.6
      Attempting uninstall: tensorflow-estimator
        Found existing installation: tensorflow-estimator 2.9.0
        Uninstalling tensorflow-estimator-2.9.0:
          Successfully uninstalled tensorflow-estimator-2.9.0
      Attempting uninstall: tensorboard
        Found existing installation: tensorboard 2.9.1
        Uninstalling tensorboard-2.9.1:
          Successfully uninstalled tensorboard-2.9.1
      Attempting uninstall: pyparsing
        Found existing installation: pyparsing 3.0.9
        Uninstalling pyparsing-3.0.9:
          Successfully uninstalled pyparsing-3.0.9
      Attempting uninstall: gast
        Found existing installation: gast 0.4.0
        Uninstalling gast-0.4.0:
          Successfully uninstalled gast-0.4.0
      Attempting uninstall: tensorflow
        Found existing installation: tensorflow 2.9.2
        Uninstalling tensorflow-2.9.2:
          Successfully uninstalled tensorflow-2.9.2
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    xarray-einstats 0.2.2 requires numpy>=1.21, but you have numpy 1.19.1 which is incompatible.
    tensorflow-probability 0.16.0 requires gast>=0.3.2, but you have gast 0.2.2 which is incompatible.
    kapre 0.3.7 requires tensorflow>=2.0.0, but you have tensorflow 1.15.2 which is incompatible.
    jaxlib 0.3.22+cuda11.cudnn805 requires numpy>=1.20, but you have numpy 1.19.1 which is incompatible.
    jax 0.3.23 requires numpy>=1.20, but you have numpy 1.19.1 which is incompatible.
    cupy-cuda11x 11.0.0 requires numpy<1.26,>=1.20, but you have numpy 1.19.1 which is incompatible.
    cmdstanpy 1.0.7 requires numpy>=1.21, but you have numpy 1.19.1 which is incompatible.[0m
    Successfully installed colorama-0.4.5 emoji-2.1.0 gast-0.2.2 keras-applications-1.0.8 numpy-1.19.1 pyhocon-0.3.59 pyparsing-2.4.7 stanfordnlp-0.2.0 stanza-1.4.2 tensorboard-1.15.0 tensorflow-1.15.2 tensorflow-estimator-1.15.1
    




```python
#import stanfordNLP (for NLP tasks) for French language.
!python3 -c "import stanfordnlp; stanfordnlp.download('fr')"
```

    Using the default treebank "fr_gsd" for language "fr".
    Would you like to download the models for: fr_gsd now? (Y/n)
    Y
    
    Default download directory: /root/stanfordnlp_resources
    Hit enter to continue or type an alternate directory.
    
    
    Downloading models for: fr_gsd
    Download location: /root/stanfordnlp_resources/fr_gsd_models.zip
    100% 235M/235M [00:39<00:00, 5.88MB/s]
    
    Download complete.  Models saved to: /root/stanfordnlp_resources/fr_gsd_models.zip
    Extracting models file for: fr_gsd
    Cleaning up...Done.
    

Keep sure your notebook is using Numpy version: 1.19.1 and tensorflow version 1.15.2. If not restart your runtime.


```python
import numpy as np
import tensorflow as tf


np. __version__ , tf.__version__
```




    ('1.19.1', '1.15.2')



If you are working on your machine or with Google Colab Pro, the following three bash instructions need to be executed only once in your environment. The resources (memory and computation) provided by the free version of Google Colab are not enough to run this project. The project needs at least 16GB of RAM for prediction.


```python
#!bash -x -e setup_all.sh
#!bash -x -e setup_models_dem1921.sh
#!bash -x -e setup_corpus_dem1921.sh
```

With the instruction !bash -x -e setup_all.sh you will:
*   Install the pre-trained GloVe embedding for French (this will generate the file cc.fr.300.vec.
*   Create a new tensorflow operation {coref_ops.extract_spans()} based on the C++ file **coref_kernels.cc** by generating a **coref_kernels.so** file, as long as this file exists, the created tensorflow op is availaible. Check the following link if you'd like to learn how to  [create new tensorflow operations](https://www.tensorflow.org/guide/create_op).
Once you create the operation and download the GloVe embeddings, you won't need to execute this bash instruction again.


```python
!bash -x -e setup_all.sh
```

    + curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 1228M  100 1228M    0     0  39.1M      0  0:00:31  0:00:31 --:--:-- 38.5M
    + gunzip -d cc.fr.300.vec.gz
    + TF_CFLAGS=($(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
    ++ python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'
    + TF_LFLAGS=($(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))
    ++ python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'
    + g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -fPIC -I/usr/local/lib/python3.7/dist-packages/tensorflow_core/include -D_GLIBCXX_USE_CXX11_ABI=0 -L/usr/local/lib/python3.7/dist-packages/tensorflow_core -l:libtensorflow_framework.so.1 -O2 -D_GLIBCXX_USE_CXX11_ABI=0
    

The instruction !bash -x -e setup_corpus_dem1921.sh will enable you: 
*   To download the Enriched version of the DEMOCRAT corpus used for training and evaluating the [COFR system](https://hal.archives-ouvertes.fr/hal-02476902/document) by [Mr. Bruno Oberle](https://boberle.com/projects/coreference-resolution-with-cofr/)
*   To generate the vocabulary of the all including charaters in the corpus.

After this instruction, the files dev.french.jsonlines, test.french.jsonlines, train.french.jsonlines and char_vocab.french.txt will be generated. 


```python
!bash -x -e setup_corpus_dem1921.sh
```

    + curl -Lo dev.french.jsonlines.bz2 http://boberle.com/files/corpora/dem1921/dem1921_sg_cut2000.dev.jsonlines.bz2
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   194  100   194    0     0    638      0 --:--:-- --:--:-- --:--:--   638
    100 70448  100 70448    0     0  69543      0  0:00:01  0:00:01 --:--:--  308k
    + curl -Lo train.french.jsonlines.bz2 http://boberle.com/files/corpora/dem1921/dem1921_sg_cut2000.train.jsonlines.bz2
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   194  100   194    0     0    822      0 --:--:-- --:--:-- --:--:--   822
    100  520k  100  520k    0     0   365k      0  0:00:01  0:00:01 --:--:--  735k
    + curl -Lo test.french.jsonlines.bz2 http://boberle.com/files/corpora/dem1921/dem1921_sg_cut2000.test.jsonlines.bz2
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   194  100   194    0     0    743      0 --:--:-- --:--:-- --:--:--   740
    100 73641  100 73641    0     0  73494      0  0:00:01  0:00:01 --:--:-- 73494
    + bzip2 -d dev.french.jsonlines.bz2
    + bzip2 -d train.french.jsonlines.bz2
    + bzip2 -d test.french.jsonlines.bz2
    + python3 get_char_vocab.py
    Wrote 112 characters to char_vocab.french.txt
    

By executing the bash file setup_corpus_dem1921.sh, you will download the pre-trained checkpoints of [COFR system](https://hal.archives-ouvertes.fr/hal-02476902/document) by [Mr. Bruno Oberle](https://boberle.com/projects/coreference-resolution-with-cofr/). This bash instruction generates the folder **/logs** where the checkpoints are stored. In this project we will only be interested in the pre-trained checkponits of the [Baseline model](https://aclanthology.org/P19-1066/); it means the checkpoints **logs/fr_mentcoref**.


```python
!bash -x -e setup_models_dem1921.sh
```

    + curl -LO http://boberle.com/files/models/dem1921_models.tar
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   194  100   194    0     0    683      0 --:--:-- --:--:-- --:--:--   683
    100  624M  100  624M    0     0  17.6M      0  0:00:35  0:00:35 --:--:-- 18.5M
    + tar xf dem1921_models.tar
    + rm dem1921_models.tar
    

Downloading Bert Model for contextualized embeddings.


```python
!wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
!unzip multi_cased_L-12_H-768_A-12.zip
!rm multi_cased_L-12_H-768_A-12.zip
```

    --2022-10-21 15:06:05--  https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
    Resolving storage.googleapis.com (storage.googleapis.com)... 108.177.121.128, 142.250.159.128, 142.251.120.128, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|108.177.121.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 662903077 (632M) [application/zip]
    Saving to: ‘multi_cased_L-12_H-768_A-12.zip’
    
    multi_cased_L-12_H- 100%[===================>] 632.19M   107MB/s    in 6.0s    
    
    2022-10-21 15:06:11 (105 MB/s) - ‘multi_cased_L-12_H-768_A-12.zip’ saved [662903077/662903077]
    
    Archive:  multi_cased_L-12_H-768_A-12.zip
       creating: multi_cased_L-12_H-768_A-12/
      inflating: multi_cased_L-12_H-768_A-12/bert_model.ckpt.meta  
      inflating: multi_cased_L-12_H-768_A-12/bert_model.ckpt.data-00000-of-00001  
      inflating: multi_cased_L-12_H-768_A-12/vocab.txt  
      inflating: multi_cased_L-12_H-768_A-12/bert_model.ckpt.index  
      inflating: multi_cased_L-12_H-768_A-12/bert_config.json  
    


## **2.   Trying the coreference resolution model with notebook**

The following cell instanciates the model architecture based on the configuration of the model.


```python
import util
from coref_model import CorefModel as cm

coref_model = "fr_mentcoref"
config = util.initialize_from_env(coref_model)
model = cm(config)
```

    WARNING:tensorflow:
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
      * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.
    
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_ops.py:11: The name tf.NotDifferentiable is deprecated. Please use tf.no_gradient instead.
    
    

    Setting CUDA_VISIBLE_DEVICES to: 
    Running experiment: fr_mentcoref
    max_top_antecedents = 50
    max_training_sentences = 50
    top_span_ratio = 0.4
    filter_widths = [
      3
      4
      5
    ]
    filter_size = 50
    char_embedding_size = 8
    contextualization_size = 200
    contextualization_layers = 3
    ffnn_size = 150
    ffnn_depth = 2
    feature_size = 20
    max_span_width = 30
    use_metadata = true
    use_features = true
    model_heads = true
    coref_depth = 2
    lm_layers = 4
    lm_size = 768
    coarse_to_fine = true
    refinement_sharing = false
    max_gradient_norm = 5.0
    lstm_dropout_rate = 0.4
    lexical_dropout_rate = 0.5
    dropout_rate = 0.2
    optimizer = "adam"
    learning_rate = 0.001
    decay_rate = 1.0
    decay_frequency = 100
    ema_decay = 0.9999
    eval_frequency = 6
    report_frequency = 2
    log_root = "logs"
    cluster {
      addresses {
        ps = [
          "130.79.164.53:2230"
        ]
        worker = [
          "130.79.164.53:2228"
          "130.79.164.33:2229"
          "130.79.164.52:2235"
        ]
      }
      gpus = [
        0
      ]
    }
    multi_gpu = false
    gold_loss = false
    b3_loss = false
    mention_loss = false
    antecedent_loss = true
    entity_equalization = true
    antecedent_averaging = false
    use_cluster_size = true
    entity_average = false
    use_gold_mentions = false
    save_frequency = 100
    include_singletons = true
    eval_for_mentions = false
    char_vocab_path = "char_vocab.french.txt"
    head_embeddings {
      path = "cc.fr.300.vec"
      size = 300
    }
    context_embeddings {
      path = "cc.fr.300.vec"
      size = 300
    }
    train_path = "train.french.jsonlines"
    eval_path = "dev.french.jsonlines"
    genres = [
      "ge"
    ]
    bert_model_path = "multi_cased_L-12_H-768_A-12"
    lm_path = "bert_features_evaluate.hdf5"
    log_dir = "logs/fr_mentcoref"
    Loading word embeddings from cc.fr.300.vec...
    

    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:49: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:51: The name tf.PaddingFIFOQueue is deprecated. Please use tf.queue.PaddingFIFOQueue instead.
    
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:281: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:378: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.
    
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/util.py:169: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
    
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:415: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    

    Done loading word embeddings.
    

    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:842: bidirectional_dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `keras.layers.Bidirectional(keras.layers.RNN(cell))`, which is equivalent to this API
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow_core/python/ops/rnn.py:464: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `keras.layers.RNN(cell)`, which is equivalent to this API
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/util.py:156: The name tf.nn.xw_plus_b is deprecated. Please use tf.compat.v1.nn.xw_plus_b instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow_core/python/ops/rnn.py:244: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:274: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:643: The name tf.log is deprecated. Please use tf.math.log instead.
    
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:500: The name tf.unsorted_segment_min is deprecated. Please use tf.math.unsorted_segment_min instead.
    
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:586: The name tf.losses.sigmoid_cross_entropy is deprecated. Please use tf.compat.v1.losses.sigmoid_cross_entropy instead.
    
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:57: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.
    
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:58: The name tf.train.exponential_decay is deprecated. Please use tf.compat.v1.train.exponential_decay instead.
    
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:61: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.
    
    /usr/local/lib/python3.7/dist-packages/tensorflow_core/python/framework/indexed_slices.py:424: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
      "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
    /usr/local/lib/python3.7/dist-packages/tensorflow_core/python/framework/indexed_slices.py:424: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
      "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
    /usr/local/lib/python3.7/dist-packages/tensorflow_core/python/framework/indexed_slices.py:424: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
      "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
    /usr/local/lib/python3.7/dist-packages/tensorflow_core/python/framework/indexed_slices.py:424: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
      "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:65: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.
    
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:66: The name tf.train.GradientDescentOptimizer is deprecated. Please use tf.compat.v1.train.GradientDescentOptimizer instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow_core/python/training/moving_averages.py:433: Variable.initialized_value (from tensorflow.python.ops.variables) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use Variable.read_value. Variables in 2.X are initialized automatically both in eager and graph (inside tf.defun) contexts.
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:77: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
    
    WARNING:tensorflow:From /content/drive/MyDrive/cofr/coref_model.py:80: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
    
    

The following function tokenizes the input conversation


```python
#tokenize the text
import json
import stanfordnlp
import re

def tokenize_text(list_text, lang , nlp1  , doc):
    res_sents = []
    res_pars = []
    res_pos = []
    start_par = 0
    for par in list_text:
        par = par.strip()
        if not par:
            continue
        doc = stanfordnlp.Document(par)
        
        doc = nlp1(doc)
        #print(doc.conll_file.conll_as_string())
        #print(doc.conll_file.sents)
        sents = [
            [ token[1] for token in sent if '-' not in token[0] ]
            for sent in doc.conll_file.sents
        ]
        pos = [
            [ token[3] for token in sent if '-' not in token[0] ]
            for sent in doc.conll_file.sents
        ]
        res_sents.extend(sents)
        res_pos.extend(pos)
        length = sum((len(s) for s in sents))
        res_pars.append([start_par, start_par+length-1])
        start_par = start_par+length
    return res_sents, res_pos, res_pars

nlp1 = stanfordnlp.Pipeline(lang="fr", processors="tokenize,pos,mwt")
```

    Use device: cpu
    ---
    Loading: tokenize
    With settings: 
    {'model_path': '/root/stanfordnlp_resources/fr_gsd_models/fr_gsd_tokenizer.pt', 'lang': 'fr', 'shorthand': 'fr_gsd', 'mode': 'predict'}
    ---
    Loading: pos
    With settings: 
    {'model_path': '/root/stanfordnlp_resources/fr_gsd_models/fr_gsd_tagger.pt', 'pretrain_path': '/root/stanfordnlp_resources/fr_gsd_models/fr_gsd.pretrain.pt', 'lang': 'fr', 'shorthand': 'fr_gsd', 'mode': 'predict'}
    ---
    Loading: mwt
    With settings: 
    {'model_path': '/root/stanfordnlp_resources/fr_gsd_models/fr_gsd_mwt_expander.pt', 'lang': 'fr', 'shorthand': 'fr_gsd', 'mode': 'predict'}
    Building an attentional Seq2Seq model...
    Using a Bi-LSTM encoder
    Using soft attention for LSTM.
    Finetune all embeddings.
    Done loading processors!
    ---
    

Predicting the coreference clusters within a given text (conversation).


```python
import time

from deployment import make_json
from predict import predict


def predicting(string , model , config  , nlp):
  paragraphs = re.split(r'\n+', string)
  doc = stanfordnlp.Document(string) 
  sents, pos, pars = tokenize_text(paragraphs , "fr" , nlp  , doc)
  conver_2_json_object = make_json(sents, pos, pars, fpath = "file", genre = "ge")
  coreferenced_json_object = predict(conver_2_json_object , model , config)
  return coreferenced_json_object

string = '''Quand l'université Sorbonne a été fondée ? Sur quels principes elle est fondé ? Est-elle la plus ancienne Université de France ?'''
#string = '''Quand Marie Curie est née ? Quel vaccin elle a fait ? Combien de prix Nobel elle a gagné ?'''


start = time.time()

coreferenced_json_object = predicting(string , model , config  , nlp1)

end = time.time()
print("the necessary time for prediction is : " , end-start , "seconds")
```

    WARNING:tensorflow:Estimator's model_fn (<function model_fn_builder.<locals>.model_fn at 0x7f17f3f1e3b0>) includes params argument, but params are not passed to Estimator.
    WARNING:tensorflow:Using temporary folder as model directory: /tmp/tmpl486ky53
    INFO:tensorflow:Using config: {'_model_dir': '/tmp/tmpl486ky53', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': None, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f17f2758790>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1, '_tpu_config': TPUConfig(iterations_per_loop=2, num_shards=8, num_cores_per_replica=None, per_host_input_for_training=3, tpu_job_name=None, initial_infeed_sleep_secs=None, input_partition_dims=None, eval_training_input_configuration=2, experimental_host_call_every_n_steps=1), '_cluster': None}
    INFO:tensorflow:_TPUContext: eval_on_tpu True
    WARNING:tensorflow:eval_on_tpu ignored because use_tpu is False.
      0%|          | 0/24 [00:00<?, ?it/s]INFO:tensorflow:Could not find trained model in model_dir: /tmp/tmpl486ky53, running initialization to predict.
    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Running infer on CPU
    INFO:tensorflow:**** Trainable Variables ****
    INFO:tensorflow:  name = bert/embeddings/word_embeddings:0, shape = (119547, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/embeddings/token_type_embeddings:0, shape = (2, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/embeddings/position_embeddings:0, shape = (512, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/embeddings/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/embeddings/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/value/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/value/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/query/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/query/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/key/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/key/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/dense/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_0/intermediate/dense/kernel:0, shape = (768, 3072), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_0/intermediate/dense/bias:0, shape = (3072,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_0/output/dense/kernel:0, shape = (3072, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_0/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_0/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_0/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/value/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/value/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/query/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/query/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/key/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/key/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_1/attention/output/dense/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_1/attention/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_1/attention/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_1/attention/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_1/intermediate/dense/kernel:0, shape = (768, 3072), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_1/intermediate/dense/bias:0, shape = (3072,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_1/output/dense/kernel:0, shape = (3072, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_1/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_1/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_1/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/value/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/value/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/query/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/query/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/key/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/key/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_2/attention/output/dense/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_2/attention/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_2/attention/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_2/attention/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_2/intermediate/dense/kernel:0, shape = (768, 3072), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_2/intermediate/dense/bias:0, shape = (3072,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_2/output/dense/kernel:0, shape = (3072, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_2/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_2/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_2/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/value/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/value/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/query/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/query/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/key/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/key/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_3/attention/output/dense/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_3/attention/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_3/attention/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_3/attention/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_3/intermediate/dense/kernel:0, shape = (768, 3072), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_3/intermediate/dense/bias:0, shape = (3072,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_3/output/dense/kernel:0, shape = (3072, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_3/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_3/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_3/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/value/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/value/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/query/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/query/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/key/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/key/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_4/attention/output/dense/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_4/attention/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_4/attention/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_4/attention/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_4/intermediate/dense/kernel:0, shape = (768, 3072), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_4/intermediate/dense/bias:0, shape = (3072,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_4/output/dense/kernel:0, shape = (3072, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_4/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_4/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_4/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/value/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/value/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/query/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/query/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/key/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/key/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_5/attention/output/dense/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_5/attention/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_5/attention/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_5/attention/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_5/intermediate/dense/kernel:0, shape = (768, 3072), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_5/intermediate/dense/bias:0, shape = (3072,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_5/output/dense/kernel:0, shape = (3072, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_5/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_5/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_5/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_6/attention/self/value/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_6/attention/self/value/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_6/attention/self/query/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_6/attention/self/query/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_6/attention/self/key/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_6/attention/self/key/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_6/attention/output/dense/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_6/attention/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_6/attention/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_6/attention/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_6/intermediate/dense/kernel:0, shape = (768, 3072), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_6/intermediate/dense/bias:0, shape = (3072,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_6/output/dense/kernel:0, shape = (3072, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_6/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_6/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_6/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_7/attention/self/value/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_7/attention/self/value/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_7/attention/self/query/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_7/attention/self/query/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_7/attention/self/key/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_7/attention/self/key/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_7/attention/output/dense/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_7/attention/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_7/attention/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_7/attention/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_7/intermediate/dense/kernel:0, shape = (768, 3072), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_7/intermediate/dense/bias:0, shape = (3072,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_7/output/dense/kernel:0, shape = (3072, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_7/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_7/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_7/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_8/attention/self/value/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_8/attention/self/value/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_8/attention/self/query/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_8/attention/self/query/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_8/attention/self/key/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_8/attention/self/key/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_8/attention/output/dense/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_8/attention/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_8/attention/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_8/attention/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_8/intermediate/dense/kernel:0, shape = (768, 3072), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_8/intermediate/dense/bias:0, shape = (3072,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_8/output/dense/kernel:0, shape = (3072, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_8/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_8/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_8/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_9/attention/self/value/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_9/attention/self/value/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_9/attention/self/query/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_9/attention/self/query/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_9/attention/self/key/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_9/attention/self/key/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_9/attention/output/dense/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_9/attention/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_9/attention/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_9/attention/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_9/intermediate/dense/kernel:0, shape = (768, 3072), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_9/intermediate/dense/bias:0, shape = (3072,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_9/output/dense/kernel:0, shape = (3072, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_9/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_9/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_9/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_10/attention/self/value/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_10/attention/self/value/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_10/attention/self/query/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_10/attention/self/query/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_10/attention/self/key/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_10/attention/self/key/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_10/attention/output/dense/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_10/attention/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_10/attention/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_10/attention/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_10/intermediate/dense/kernel:0, shape = (768, 3072), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_10/intermediate/dense/bias:0, shape = (3072,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_10/output/dense/kernel:0, shape = (3072, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_10/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_10/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_10/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_11/attention/self/value/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_11/attention/self/value/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_11/attention/self/query/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_11/attention/self/query/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_11/attention/self/key/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_11/attention/self/key/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_11/attention/output/dense/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_11/attention/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_11/attention/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_11/attention/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_11/intermediate/dense/kernel:0, shape = (768, 3072), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_11/intermediate/dense/bias:0, shape = (3072,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_11/output/dense/kernel:0, shape = (3072, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_11/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_11/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/encoder/layer_11/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/pooler/dense/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*
    INFO:tensorflow:  name = bert/pooler/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    100%|██████████| 24/24 [00:06<00:00,  3.70it/s]INFO:tensorflow:prediction_loop marked as finished
    INFO:tensorflow:prediction_loop marked as finished
    100%|██████████| 24/24 [00:06<00:00,  3.67it/s]
    

    Restoring from logs/fr_mentcoref/model.max.ckpt
    

    INFO:tensorflow:Restoring parameters from logs/fr_mentcoref/model.max.ckpt
    

    the necessary time for prediction is :  10.258581399917603 seconds
    

Now that the coreference clusters within the input user questions are predicted, we need to replace every pronoun by its representative entity. 


```python
from deployment import return_coreferenced_sentence
paragraphs = re.split(r'\n+', string)

last_question_coreferenced = return_coreferenced_sentence(coreferenced_json_object , paragraphs) # last_question_coreferenced is the string of the last question the user asks where all pronouns are replaced by the entities they refer to. This string in real-world project is sent to the RASA chatbot servers. So that the chatbot could recognize the user's question and then answer it.
```


```python
last_question_coreferenced
```




    "Quand l' université Sorbonne a été fondée ? Sur quels principes l' université Sorbonne est fondé ? Est l' université Sorbonne la plus ancienne Université de France ?"




# 3.   **Evaluation**


```python
from eval_metrics import calculate_recall_precision , muc , b_cubed , lea , ceafe
```

Split your corpus


```python
import re             # start and end are number of lines we want extract from the input jsonlines file.
def extractLines(input_file , output_file , start, end):
  file=open(input_file,'r')
  file_content=file.read()

  objects=re.findall('(.*)\n', file_content)
  target=objects[start-1:end]# format the target 
  target_string='\n'.join([line for line in target])
  fp=open(output_file,'w')# file-like object
  fp.write(target_string)
  fp.close()
```


```python
oringinal_file = 'test.french.jsonlines'
partial_filename = 'test.french.jsonlines_part'
start , end = 1 , 5


extractLines(oringinal_file , partial_filename , start, end)   #this command create a new file containing the first 5 texts of the input file (test set of the Enriched DEMOCRAT corpus).
```

Create a Json file for the prediction resluts. So we can evaluate some coreference resolution metrics for this system or for another. for more information about these metrics, check the following link [Which Coreference Evaluation Metric Do You Trust?A Proposal for a Link-based Entity Aware Metric](https://aclanthology.org/P16-1060/)


```python
import json
import time

start = time.time()

actual_clusters_filename = partial_filename
predicted_clusters_filename = 'predictions.jsonlines'



with open(predicted_clusters_filename , "w") as fout:

  with open(actual_clusters_filename) as fin :

    i = 0  
    for line in fin.readlines():
        example = json.loads(line)
        example = predict(example , model , config)
        fout.write(json.dumps(example))
        fout.write("\n")
        print("text : " ,  i+1)
        i = i+1


end = time.time()
print("the necessary time for generating this prediction file is : " , end-start , "seconds")
```

    the necessary time for generating this prediction file is :  817.2504370212555 seconds
    

The function calculate_recall_precision expects to receive two filenames as inputs. The first filename is the one of the actual coreference clusters; the filename of the predicted clusters by the system (model) is the other.


```python
muc_ = calculate_recall_precision(predicted_clusters_filename , actual_clusters_filename , muc)
b_cubed_ = calculate_recall_precision(predicted_clusters_filename , actual_clusters_filename , b_cubed)
lea_ = calculate_recall_precision(predicted_clusters_filename , actual_clusters_filename , lea)
ceafe_ = calculate_recall_precision(predicted_clusters_filename , actual_clusters_filename , ceafe)
```

We evaluate the system with differents coreference resolution metrics on the the first 5 documents of the test part of the Enriched version of DEMOCRAT. You can evaluate on whatever corpus you want (your can generate your own corpus, annotate it and evaluate it with these metrics).


```python
print("The Recall, Precision and F1-score of MUC metric in this portion of the data are : " ,muc_[:3] , " respectively")
print("The Recall, Precision and F1-score of B_CUBBED metric in this portion of the data are : " ,b_cubed_[:3] , " respectively")
print("The Recall, Precision and F1-score of LEA metric in this portion of the data are : " ,lea_[:3] , " respectively")
print("The Recall, Precision and F1-score of CEAFe in this portion of the data metric are : " ,ceafe_[:3] , " respectively")
```

    The Recall, Precision and F1-score of MUC metric in this portion of the data are :  (68.92948190604024, 78.71497717893598, 73.3989911382387)  respectively
    The Recall, Precision and F1-score of B_CUBBED metric in this portion of the data are :  (52.10735581244137, 62.401987561168724, 56.1943280787216)  respectively
    The Recall, Precision and F1-score of LEA metric in this portion of the data are :  (48.2986455852399, 58.90011235963118, 52.46201242653933)  respectively
    The Recall, Precision and F1-score of CEAFe in this portion of the data metric are :  (12.459064153771276, 72.71935184855849, 21.0843651340603)  respectively
    

For better analysis, you can visualize te results of the prediction with an HTML file.


```python
!python3 visualization/jsonlines2text.py predictions.jsonlines -i -o visualize_results.html --sing-color "" --cm ""
```



## 4.  **Deployment**

To deploy this coreference resolution system in a chatbot, you have to create two virtaul environnements:


**1.   The first for the coreference resolution model by following these steps:**

*   open a terminal
*   conda create -n coref_env python=3.7
*   conda activate coref_env
*   (coref_env) pip3 install -r requirements.txt
*   (coref_env) cd <working_folder>
*   (coref_env) python3 app.py

These instructions create a Flask server that will be responsible for coreference resolution task.


**2.   Thes second environnement where your RASA framework for chatbots occurs (Tensorflow 2):**


*   open another terminal
*   conda create -n rasa_chatbot python=3.9
*   conda activate rasa_chatbot
*   (rasa_chatbot) pip3 install rasa    #and some other dependencies.
*   #run the needed rasa servers and connect your chatbot to the coreference resolution system).
*   For an example open a python file in this environnement and send the variable paragraphs = re.split(r'\n+', string) with an HTTP post request to (coref_env). The app.py server will provide you with the coreferenced string where coreference resolution is resolved.












