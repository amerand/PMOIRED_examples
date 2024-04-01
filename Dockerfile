FROM python:3.11.8-alpine3.19
RUN apk add gcc python3-dev musl-dev linux-headers 
RUN pip3 install scipy numpy matplotlib astropy astroquery jupyterlab ipympl
# -- github.com/amerand/PMOIRED
ADD PMOIRED /PMOIRED
# -- github.com/amerand/PMOIRED_examples
ADD PMOIRED_examples /PMOIRED_examples
RUN cd /PMOIRED; pip3 install .
WORKDIR /PMOIRED_examples
EXPOSE 8888
# --ip 0.0.0.0 to allow external connection
ENTRYPOINT ["jupyter-lab", "--allow-root", "--ip", "0.0.0.0", "--NotebookApp.token=''"]
# docker build -t pmoired:latest .
# docker run -p 8888:8888 pmoired 