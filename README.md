# Installation
1.	Clone the git repository from [here](https://github.com/arijit-intel/VectorDBBench/tree/changes_for_vdms).
2.	Follow the installation instructions from [here](https://github.com/zilliztech/VectorDBBench).

# Start VDMS docker
1. docker pull intellabs/vdms:latest
2. docker run $DOCKER_PROXY_RUN_ARGS --rm -a stdout -a stderr -p 55555:55555 --name vdms_new intellabs/vdms:latest
