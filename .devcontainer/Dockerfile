# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.155.1/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version: 3, 3.9, 3.8, 3.7, 3.6
ARG VARIANT="3"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# COPY requirements.txt /tmp/pip-tmp/
# RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
RUN wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | apt-key add -
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get -y install --no-install-recommends libtool automake build-essential libboost-all-dev libxml2-dev flex gfortran bison adoptopenjdk-8-hotspot texinfo
RUN pipx install poetry
# RUN cd /workspaces/tenet && export JAVA_HOME=/usr/lib/jvm/adoptopenjdk-8-hotspot-amd64 && export CPATH=/usr/lib/x86_64-linux-gnu/openmpi/include && export C_INCLUDE_PATH=/usr/lib/x86_64-linux-gnu/openmpi/include && poetry install
