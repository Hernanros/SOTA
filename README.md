# SOTA
 repo for shared work and communication on Ydata NLP course

+++++++++++++++++++++++++++++++++++++++++++++++++++++

DOCKER

$ cd docker 
$ docker build -t sota-nlp .
$ winpty docker run -it -p 8080:8080 --name sota-nlp-container --user root -v /${PWD}:/root sota-nlp

+++++++++++++++++++++++++++++++++++++++++++++++++++++