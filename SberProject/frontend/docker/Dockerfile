FROM nginx:1.23.0

COPY data /usr/share/nginx/data
COPY nginx/nginx.conf /etc/nginx/nginx.conf

ENV FRONTEND_HOST "130.193.54.34"
ENV SEARCH_HOST "51.250.20.120"
ENV METRICS_HOST "51.250.102.156"

RUN sed -i "s/FRONTEND_HOST/${FRONTEND_HOST}/g" /usr/share/nginx/data/www/index.html
RUN sed -i "s/SEARCH_HOST/${SEARCH_HOST}/g" /usr/share/nginx/data/js/script.js
RUN sed -i "s/METRICS_HOST/${METRICS_HOST}/g" /usr/share/nginx/data/js/script.js
