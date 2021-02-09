ARG VARIANT="3"
FROM python:${VARIANT}

RUN apt-get update && apt-get install -y software-properties-common

RUN wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | apt-key add - && add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get -y install --no-install-recommends libtool automake build-essential libboost-all-dev libxml2-dev flex gfortran bison adoptopenjdk-8-hotspot texinfo

# Install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python - && source $HOME/.poetry/env