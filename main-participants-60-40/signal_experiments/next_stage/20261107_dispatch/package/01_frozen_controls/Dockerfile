FROM python:3.13@sha256:ecbe213fb818d557fd19db4078244339e2e2e52874f9c81052d3094ea114d797
RUN python -m pip --disable-pip-version-check install --no-cache-dir numpy==2.5.4 scipy==1.18.1 lightgbm==4.7.0 narwhals==2.28.0
RUN python -I -B -c 'import sys, numpy, scipy, lightgbm, narwhals; print(sys.version, numpy.__version__, scipy.__version__, lightgbm.__version__, narwhals.__version__)'
