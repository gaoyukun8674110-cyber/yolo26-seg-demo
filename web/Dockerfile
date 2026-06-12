# ---- YOLO26 web showcase (multi-stage: build static assets, serve via nginx) ----

# Stage 1: build the React/Vite static bundle.
FROM node:20-alpine AS build
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# Stage 2: serve the built dist/ with nginx, proxying API calls to the backend.
FROM nginx:1.27-alpine
COPY web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 80
