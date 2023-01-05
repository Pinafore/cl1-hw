#!/usr/bin/env bash

apt-get install -y python3 python3-pip python3-dev

pip3 install -r /autograder/source/requirements.txt

python3 -m nltk.downloader cmudict

python3 -m nltk.downloader punkt

python3 -m nltk.downloader words
