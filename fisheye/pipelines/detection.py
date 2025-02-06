import torch
from tqdm import tqdm


class DetectionPipeline:
    """Object Detection Pipeline"""
    def __init__(self, model, device, preprocess_fn=None, postprocess_fn=None, batch_size=BATCH_SIZE):
        """
        Initializes the DetectionPipeline with given model, device, and optional processing functions.

        Args:
            model: PyTorch detection model.
            device: The device (CPU/GPU) for inference.
            preprocess_fn: Optional preprocessing function to apply to the images.
            postprocess_fn: Optional postprocessing function to process the model's output.
            batch_size: Batch size for detection.
        """
        self.model = model
        self.device = device
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.batch_size = batch_size

    def run(self, dataloader, gp=None, verbose=True):
        """Runs detection.

        Args:
            dataloader: PyTorch DataLoader containing input images.
            gp: Optional callback function for progress updates.
            verbose: Whether to display progress bar.

        Returns:
            inference: The model's raw output.
            image_shapes: List of image shapes for resizing later.
            width: Width of the input images.
            height: Height of the input images.
        """
        if gp:
            gp(0, "Running detection pipeline...")

        inference = []
        image_shapes = []

        with tqdm(total=len(dataloader) * self.batch_size, desc="Running detection", ncols=0,
                  disable=not verbose) as pbar:
            for batch_i, (img, *extra_data) in enumerate(dataloader):
                if gp:
                    gp(batch_i / len(dataloader), pbar.__str__())

                img = img.to(self.device, non_blocking=True)
                img = img.half() if self.device.type != 'cpu' else img.float()
                img /= 255.0  # Normalize to [0, 1]

                if self.preprocess_fn:
                    img = self.preprocess_fn(img)

                with torch.no_grad():
                    output = self.model(img)

                if self.postprocess_fn:
                    output = self.postprocess_fn(output)

                image_shapes.append(img.shape[2:])
                inference.append(output)
                pbar.update(1 * self.batch_size)

            # outputs, shapes, width, and height
            return inference, image_shapes, img.shape[3], img.shape[2]