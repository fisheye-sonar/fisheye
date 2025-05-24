DOCKER_IMAGE=fisheye-mac
DOCKER_IMAGE_WINDOWS=fisheye-windows

# Build the Docker image
build-mac:
	docker build -f docker/Dockerfile.mac -t $(DOCKER_IMAGE) .

build-windows:
	docker build -f docker/Dockerfile.windows -t $(DOCKER_IMAGE_WINDOWS) .

# Start a tmp interactive shell in the container
shell-mac:
	docker run --rm -it -v $(PWD):/app $(DOCKER_IMAGE)

shell-windows:
	docker run --rm -it -v $(PWD):/app $(DOCKER_IMAGE_WINDOWS)

# Persistent containers
run-mac:
	docker run -it --name fisheye-prod -v $(PWD):/app $(DOCKER_IMAGE)

run-windows:
	docker run -it --name fisheye-prod -v $(PWD):/app $(DOCKER_IMAGE_WINDOWS)