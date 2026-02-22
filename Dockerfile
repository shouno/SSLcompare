# 1. ベースイメージの選択
# サーバーのNVIDIAドライバと互換性のあるCUDAバージョンを選択してください。
# 例: CUDA 12.5.1, cuDNN 8, Ubuntu 22.04 のruntimeイメージ
ARG CUDA_VERSION=12.5.1
#ARG CUDNN_VERSION=8
ARG OS_VERSION=ubuntu22.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-${OS_VERSION}

# 対話的なプロンプトを無効化
ENV DEBIAN_FRONTEND=noninteractive

# 2. 必要なパッケージのインストール
# 開発に必要な基本的なツールやPythonをインストールします。
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    sudo \
    python3 \
    python3-pip \
    python3-venv \
    tmux \
    rsync \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. 一般ユーザーの作成
# セキュリティのため、rootではない一般ユーザーを作成して作業します。
# VSCodeのdevcontainer.jsonからユーザー名やIDを受け取れるようにARGを設定します。
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# bash as default shell
RUN chsh -s /bin/bash ${USERNAME}
COPY --chown=${USERNAME}:${USERNAME} .bashrc /home/${USERNAME}/.bashrc
COPY --chown=${USERNAME}:${USERNAME} .tmux.conf /home/${USERNAME}/.tmux.conf

# 4. ワークスペースの設定
# コンテナ内の作業ディレクトリを指定します。
USER $USERNAME
WORKDIR /workspace

