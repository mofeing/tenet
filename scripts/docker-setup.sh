#!/bin/bash

export JAVA_HOME=/usr/lib/jvm/adoptopenjdk-8-hotspot-amd64
export CPATH=/usr/lib/x86_64-linux-gnu/openmpi/include

source $HOME/.poetry/env
poetry config virtualenvs.create false
poetry install --verbose