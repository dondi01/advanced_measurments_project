@echo off
REM docker system prune -af
docker build -t test4 .
docker buildx prune -f
