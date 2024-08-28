FROM nvcr.io/nvidia/pytorch:23.11-py3

ENV DEBIAN_FRONTEND noninteractive
ENV CONDA_HOME /opt/conda

WORKDIR /workspace

RUN apt-get update && apt-get install -y wget nano 
# Download and install Miniconda
#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
#    bash /tmp/miniconda.sh -b -p $CONDA_HOME && \
#    rm /tmp/miniconda.sh

# Add Conda to the system PATH
#ENV PATH $CONDA_HOME/bin:$PATH

RUN apt-get install -y build-essential libgtk-3-dev libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev openssh-server
RUN mkdir /usr/local/lib/cmake/opencv3
RUN cd /usr/local/lib/cmake/opencv3 && git clone https://github.com/opencv/opencv.git
RUN cd /usr/local/lib/cmake/opencv3 && git clone https://github.com/opencv/opencv_contrib.git && cd opencv_contrib && git checkout 3.4
RUN cd /usr/local/lib/cmake/opencv3/opencv && git checkout 3.4 && mkdir build && cd build && cmake -DOPENCV_EXTRA_MODULES_PATH=/usr/local/lib/cmake/opencv3/opencv_contrib/modules -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. && make -j7 # runs 7 jobs in parallel && make install

RUN mv /usr/local/lib/cmake/opencv3/opencv/build/*.cmake /usr/local/lib/cmake/opencv3/

#RUN conda create --name my_env python=3.6.9

RUN pip install pybind11
RUN git clone https://github.com/hfutcgncas/normalSpeed.git \
    && cd normalSpeed/normalSpeed \
    && python3 setup.py install --user

RUN pip install plyfile
RUN pip install Cython==0.29.15
RUN pip install matplotlib
RUN pip install open3d
RUN pip install scikit-learn

ADD ./docker/. /workspace/dd6d/

RUN cd /workspace/dd6d/models/RandLA/utils/nearest_neighbors && python3 setup.py install --home="."
RUN cd /workspace/dd6d/models/RandLA/utils/cpp_wrappers/cpp_subsampling && python3 setup.py build_ext --inplace

RUN pip install protobuf==3.20.3

RUN ln -s /data_private/YCB_Video_Dataset/ /workspace/dd6d/datasets/ycb/

# Install useful tools
RUN apt-get install -y screen

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]

