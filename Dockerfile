FROM python:3
ADD / /
RUN pip install -r requirements.txt
ARG gridsearch
VOLUME [ "/kniffel" ]
RUN echo "<h1>Hello World (app1)</h1>" > index.html

CMD [ "python", "./src/ai/ai.py" ]