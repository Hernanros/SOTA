FROM nvidia/cuda

ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update 
RUN apt-get install -y software-properties-common 
RUN add-apt-repository ppa:deadsnakes/ppa 
#RUN add-apt-repository -y "deb https://dl.winehq.org/wine-builds/ubuntu/ bionic main" 
RUN apt-get install -yq python3.6 python3-pip htop nano git wget 
RUN apt-get install -yq libglib2.0-0 autoconf automake 
RUN apt-get install -yq libtool build-essential unzip 
RUN apt-get install -yq libarchive-dev vim


# Install Python dependencies.
ADD requirements.txt /
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN pip install "widgetsnbextension==3.5.1"
RUN pip install jupyter-contrib-core jupyter-contrib-nbextensions jupyter-nbextensions-configurator

# Install Jupyter.
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
RUN jupyter contrib nbextension install
RUN jupyter nbextension enable codefolding/main
RUN echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.port = 8080" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "jupyter notebook --no-browser --allow-root" >> /usr/local/bin/run_notebook && chmod +x /usr/local/bin/run_notebook

########
# ROUGE
########
#RUN pip uninstall -yq easy-rouge
RUN cd /tmp && mkdir rouge && git clone https://github.com/neural-dialogue-metrics/rouge.git && cd rouge && python3 ./setup.py install
RUN cd -


# Welcome message.
ADD docker/welcome_message.txt /
RUN echo '[ ! -z "$TERM" -a -r /etc/motd ] && cat /etc/motd' \
        >> /etc/bash.bashrc \
        ; cat welcome_message.txt > /etc/motd

RUN ln -s /root/opt/gensim-data /root/gensim-data
RUN ln -s /root/opt/nltk_data /root/nltk_data

WORKDIR /root
EXPOSE 8080

#expose port 22 for remote vscode
EXPOSE 22 
