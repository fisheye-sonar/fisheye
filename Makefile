DOCKER_IMAGE_JETSON=fisheye-jetson
DOCKER_IMAGE_WINDOWS=fisheye-windows

# Build the Docker image
build-jetson:
	docker build -f docker/Dockerfile.mac -t $(DOCKER_IMAGE_JETSON) .

build-windows:
	docker build -f docker/Dockerfile.windows -t $(DOCKER_IMAGE_WINDOWS) .

# Start a tmp interactive shell in the container
shell-jetson:
	docker run --rm -it -v $(PWD):/app $(DOCKER_IMAGE_JETSON)

shell-windows:
	docker run --rm -it -v $(PWD):/app $(DOCKER_IMAGE_WINDOWS)

# Persistent containers
run-jetson:
	docker run -it --name fisheye-prod -v $(PWD):/app $(DOCKER_IMAGE_JETSON)

run-windows:
	docker run --gpus all -it --name fisheye-prod -v $(PWD):/app $(DOCKER_IMAGE_WINDOWS)