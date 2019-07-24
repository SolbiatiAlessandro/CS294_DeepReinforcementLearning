# Specifies base image and tag
FROM rbahumi/cs294_roboschool_image
WORKDIR /root

# Installs additional packages
# RUN pip install pkg1 pkg2 pkg3

# Downloads training data
# RUN curl https://example-url/path-to-data/data-filename --output /root/data-filename

# Copies the trainer code to the docker image.
COPY ./trainer/GCSproxy.py /root/GCSproxy.py 
COPY ./trainer/__init__.py /root/__init__.py 
COPY ./trainer/model.py /root/model.py 
COPY ./trainer/run_expert.py /root/run_expert.py 
COPY ./trainer/task.py /root/task.py 
COPY ./trainer/tf_util.py /root/tf_util.py 
COPY ./trainer/utils.py /root/utils.py 

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "task.py"]
