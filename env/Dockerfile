# syntax=docker/dockerfile:1
# DRAFT - not built or tested (revision 1; Reviewer 3, M8 request 2). Rebuilds the three Python environments and the
# document toolchain of the selectivity_ceiling study from the pins recorded in this directory on 2026-10-15 by
# scripts/63_repro_inventory.py.
#
#   docker build -f env/Dockerfile -t selectivity-ceiling .          # build context: the selectivity_ceiling/ directory
#   docker run --rm -v "$PWD":/work -w /work/scripts selectivity-ceiling \
#       /opt/envs/analysis/bin/python 61_regenerate_frozen.py        # writes only results/revision_repro/regenerated/
#
# Interpreters (details in python-versions.txt):
#   /opt/envs/analysis  CPython 3.8.13  + requirements-venv.rebuild.txt  -- replaces ../.venv. The original is a venv
#                       over miniconda's base Python with system AND user (~/.local) site-packages enabled, so its
#                       packages come from three layers (inventory-venv.csv); its full `pip freeze`
#                       (requirements-venv.lock.txt, 279 entries, 9 conda-built `@ file://` entries) is a record, not an
#                       installable spec. The rebuild file pins the packages the .venv scripts import, plus their
#                       runtime dependencies, at the versions the .venv actually imports.
#   /opt/envs/dreval    CPython 3.11.17 (conda-venv-dreval.explicit.txt) + requirements-venv-dreval.lock.txt
#                       -- replaces ../.venv-dreval (scripts 27, 28, 34-40)
#   /opt/envs/gpu       CPython 3.10.12 + requirements-venv-gpu.lock.txt -- replaces ../.venv-gpu (script 09b; CUDA
#                       12.4 wheels, needs the NVIDIA container runtime to train)
#   /opt/envs/ots       opentimestamps-client 0.7.2 (the study's copy lived in a discarded scratch venv)
#   tectonic 0.15.0     its built-in default bundle is default_bundle_v33.tar (tlextras-2022.0r0; toolchain.txt)
#
# Bit-for-bit notes. The frozen predictions regenerate byte-for-byte on the original host (Intel Xeon Gold 6226R,
# glibc 2.35) with one BLAS thread (results/revision_repro/regeneration_report.md). OpenBLAS selects CPU-specific
# kernels at run time; OPENBLAS_CORETYPE pins the kernel family used on the original host when the container runs on
# another x86-64 CPU (an AVX-512 CPU is still required for that family). Scripts that use Python sets for tie-breaks
# depend on the string-hash seed (script 01; see the report), hence PYTHONHASHSEED. Unset the host's PYTHONPATH.

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl bzip2 git strace python3.10 python3.10-venv \
        libfontconfig1 libgraphite2-3 libharfbuzz0b libicu70 libssl3 \
    && rm -rf /var/lib/apt/lists/*

# conda (the study used conda 22.9.0 from miniconda3)
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

COPY env/ /opt/env/

# 1. analysis environment (replaces ../.venv)
RUN conda create -y -p /opt/envs/analysis python=3.8.13 pip \
    && /opt/envs/analysis/bin/pip install --no-cache-dir -r /opt/env/requirements-venv.rebuild.txt

# 2. DrEval environment (replaces ../.venv-dreval: a conda environment plus pip packages)
RUN conda create -y -p /opt/envs/dreval --file /opt/env/conda-venv-dreval.explicit.txt \
    && /opt/envs/dreval/bin/pip install --no-cache-dir \
         --extra-index-url https://download.pytorch.org/whl/cu126 -r /opt/env/requirements-venv-dreval.lock.txt

# 3. GPU environment (replaces ../.venv-gpu: a venv over the system Python 3.10.12, isolated)
RUN python3.10 -m venv /opt/envs/gpu \
    && /opt/envs/gpu/bin/pip install --no-cache-dir \
         --extra-index-url https://download.pytorch.org/whl/cu124 -r /opt/env/requirements-venv-gpu.lock.txt

# 4. OpenTimestamps client, in its own environment as in the study
RUN python3.10 -m venv /opt/envs/ots && /opt/envs/ots/bin/pip install --no-cache-dir opentimestamps-client==0.7.2
ENV PATH=$PATH:/opt/envs/ots/bin

# 5. tectonic 0.15.0 (scripts/16_build_documents.py calls `tectonic --keep-logs <name>.tex`)
RUN curl -fsSL "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz" \
    | tar -xz -C /usr/local/bin

ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_CORETYPE=SkylakeX \
    PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=
WORKDIR /work
