# SOTA
 repo for shared work and communication on Ydata NLP course




+++++++++++++++++++++++++++++++++++++++++++++++++++++

DOCKER

for embedding shared folder, on windows use the "/e/WORK/ML/data/" format and git bash

from the root folder:

$ docker-compose build <SERVICE_NAME>            ==> sotanlp_gpu / sotanlp_cpu

$ docker run -it -p 22:22 -p 8080:8080 --name sota-nlp-container --user root -v //path/to/embedding/:/root/opt -v /${PWD}:/root/main sota-nlp
(for windows might need to do 'winpty docker run ...')

on AWS to run with GPU, use:
--gpus all (https://towardsdatascience.com/how-to-set-up-docker-for-deep-learning-on-aws-bce751eaf662) => this worked for me
--runtime=nvidia  (https://wadehuang36.github.io/2019/05/29/a-easy-way-to-use-nvidia-docker-on-aws.html)

use run_notebook to open jupyter, it will open on 127.0.0.1:8080

todo:
 try: BuildKit to save pip cache and reduce build time when changing requirements.ini


to use with vscode follow:
https://www.analyticsvidhya.com/blog/2020/08/docker-based-python-development-with-cuda-support-on-pycharm-and-or-visual-studio-code/
"Setting up Visual Studio Code" section


How to avoid downloading the pre-trained models every time you restart the container:

*GLOVE
cosine similarities: torchtext.vocab , twitter.27B, 100
POS: torchtext.vocab , twitter.27B, 100
WMD: chakin , GloVe.840B.300d

*FASTTEXT
cosine similarities: with gensim

+++++++++++++++++++++++++++++++++++++++++++++++++++++