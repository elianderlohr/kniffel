FROM tensorflow/tensorflow
ADD / /
RUN pip install -r requirements.txt
VOLUME [ "/kniffel-volume" ]
CMD [ "python", "./src/ai/ai.py" ]