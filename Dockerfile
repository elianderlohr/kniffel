FROM tensorflow/tensorflow
ADD / /
RUN pip install -r requirements.txt
CMD [ "python", "./src/ai/ai.py" ]