FROM nvcr.io/nvidia/tensorrt:21.09-py3

ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=user
ARG WORKDIR=/workspace/ByteTrack

RUN apt-get update && apt-get install -y \
        automake autoconf libpng-dev nano python3-pip \
        curl zip unzip libtool swig zlib1g-dev pkg-config \
        python3-mock libpython3-dev libpython3-all-dev \
        g++ gcc cmake make pciutils cpio gosu wget \
        libgtk-3-dev libxtst-dev sudo apt-transport-https \
        build-essential gnupg git xz-utils vim \
        libva-drm2 libva-x11-2 vainfo libva-wayland2 libva-glx2 \
        libva-dev libdrm-dev xorg xorg-dev protobuf-compiler \
        openbox libx11-dev libgl1-mesa-glx libgl1-mesa-dev \
        libtbb2 libtbb-dev libopenblas-dev libopenmpi-dev \
        mesa-utils libgl1-mesa-glx ffmpeg tmux x11-apps \
    && sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \ 
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ifzhang/ByteTrack \
    && cd ByteTrack \
    && git checkout 8df560f2c85518f2512399cae0d493933d32f548 \
    && mkdir -p YOLOX_outputs/yolox_x_mix_det/track_vis \
    && sed -i 's/torch>=1.7/torch==1.9.1+cu111/g' requirements.txt \
    && sed -i 's/torchvision==0.10.0/torchvision==0.10.1+cu111/g' requirements.txt \
    && sed -i "s/'cuda:6'/0/g" tools/demo_track.py \
    && pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html \
    && python3 setup.py develop \
    && pip3 install cython \
    && pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' \
    && pip3 install cython_bbox gdown \
    && pip3 install jupyter jupyterlab ffmpeg-python ray protobuf termcolor \
    && ldconfig \
    && pip cache purge

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt \
    && cd torch2trt \
    && git checkout 0400b38123d01cc845364870bdf0a0044ea2b3b2 \
    # https://github.com/NVIDIA-AI-IOT/torch2trt/issues/619
    && wget https://github.com/NVIDIA-AI-IOT/torch2trt/commit/8b9fb46ddbe99c2ddf3f1ed148c97435cbeb8fd3.patch \
    && git apply 8b9fb46ddbe99c2ddf3f1ed148c97435cbeb8fd3.patch \
    && python3 setup.py install

# GluonCV development build
RUN git clone https://github.com/dmlc/gluon-cv \
    && echo $(ls) \
    && sudo chmod 777 -R gluon-cv && cd gluon-cv && python3 setup.py develop \
    && python3 -c "import gluoncv; print(gluoncv.__version__)" \
    && pip install indexed \
    && ldconfig \
    && pip cache purge
# Patch GluonCV 
COPY patch_files/dataset.py /workspace/gluon-cv/gluoncv/torch/data/gluoncv_motion_dataset/
COPY patch_files/multigrid_helper.py /workspace/gluon-cv/gluoncv/torch/data/video_cls/
COPY patch_files/video_io.py /workspace/gluon-cv/gluoncv/torch/data/gluoncv_motion_dataset/io/

RUN echo "root:root" | chpasswd \
    && adduser --disabled-password --gecos "" "${USERNAME}" \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && echo "%${USERNAME}    ALL=(ALL)   NOPASSWD:    ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && touch /home/user/.Xauthority 

USER ${USERNAME}
RUN sudo chown -R ${USERNAME}:${USERNAME} ${WORKDIR}
WORKDIR ${WORKDIR}