# SOTA
 repo for shared work and communication on Ydata NLP course

+++++++++++++++++++++++++++++++++++++++++++++++++++++

DOCKER

for embedding shared folder, on windows use the "/e/WORK/ML/data/" format and git bash

$ docker build -t sota-nlp docker
$ winpty docker run -it -p 8080:8080 --name sota-nlp-container --user root -v //path/to/embedding/:/root/opt -v /${PWD}:/root/main sota-nlp

use run_notebook to open jupyter, it will open on 127.0.0.1:8080

+++++++++++++++++++++++++++++++++++++++++++++++++++++