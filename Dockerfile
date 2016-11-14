FROM andrewosh/binder-base
MAINTAINER Alexis Boukouvalas <alexis.boukouvalas@gmail.com>

USER main

# Install requirements for Python 2
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Install requirements for Python 3
RUN /home/main/anaconda/envs/python3/bin/pip install -r requirements.txt

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp35-cp35m-linux_x86_64.whl

RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade $TF_BINARY_URL

# get underworld, compile, delete some unnecessary files, run tests.
RUN git clone https://github.com/GPflow/GPflow.git && \
    cd GPflow && \
    python setup.py build        && \
    python setup.py develop     

