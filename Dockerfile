# homunculus with its own Python. The base image has no Python and none is needed: uv downloads Python 3.15
# and installs the locked dependencies, all of it under /opt/homunculus.
#
#     docker build -t homunculus .
#     docker run --rm -it homunculus          # a shell with the `homunculus` command
#
# To add homunculus to an image of your own, whatever Python that image has, copy that one folder:
#
#     COPY --from=homunculus /opt/homunculus /opt/homunculus
#     ENV PATH=/opt/homunculus/bin:$PATH MUJOCO_GL=egl
#
# The image needs glibc (Debian, Ubuntu and the like) and, for camera images, the libraries installed below.
FROM debian:trixie-slim

# Certificates, to download a rig's robot. Mesa's EGL, so MuJoCo renders cameras with no display and no GPU.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libegl1 libgl1 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/homunculus/python \
    UV_PROJECT_ENVIRONMENT=/opt/homunculus/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
COPY pyproject.toml uv.lock .python-version README.md /src/
COPY src /src/src
# /opt/homunculus/bin holds the two commands and nothing else, so it can go on PATH beside another Python.
RUN uv sync --project /src --locked --no-dev --no-editable --extra mujoco \
    && mkdir /opt/homunculus/bin \
    && ln -s ../venv/bin/homunculus ../venv/bin/hom /opt/homunculus/bin/

ENV PATH=/opt/homunculus/bin:$PATH \
    MUJOCO_GL=egl
CMD ["bash"]
