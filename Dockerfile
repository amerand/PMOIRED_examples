FROM python:3.11.8-alpine3.19
RUN apk add gcc python3-dev musl-dev linux-headers 
RUN pip3 install scipy numpy matplotlib astropy astroquery jupyterlab ipympl
# -- github.com/amerand/PMOIRED
#ADD ../PMOIRED /PMOIRED
#RUN cd /PMOIRED; pip3 install .
RUN pip install -i https://test.pypi.org/simple/ pmoired==1.2
# -- github.com/amerand/PMOIRED_examples
ADD . /PMOIRED_examples
WORKDIR /PMOIRED_examples
EXPOSE 8888
# --ip 0.0.0.0 to allow external connection
ENTRYPOINT ["jupyter-lab", "--allow-root", "--ip", "0.0.0.0", "--NotebookApp.token=''"]