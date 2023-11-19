import os
import time

from random import randint, seed

import numpy as np
import cv2


class MaskGenerator():
    def __init__(self, height, width, channels=3, rand_seed=None, filepath=None):
        """Convenience functions for generating masks to be used for inpainting training
        
        Arguments:
            height {int} -- Mask height
            width {width} -- Mask width
        
        Keyword Arguments:
            channels {int} -- Channels to output (default: {3})
            rand_seed {[type]} -- Random seed (default: {None})
            filepath {[type]} -- Load masks from filepath. If None, generate masks with OpenCV (default: {None})
        """

        self.height = height
        self.width = width
        self.channels = channels
        self.filepath = filepath

        # If filepath supplied, load the list of masks within the directory
        self.mask_files = []
        if self.filepath:
            filenames = list(os.listdir(self.filepath))
            self.mask_files = [f for f in filenames if any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
            print(f">> Found {len(self.mask_files)} masks in {self.filepath}")        

        # TODO: Seed for reproducibility
        #if rand_seed:
        #    seed(rand_seed)
        seed(time.time())

    def _generate_mask(self, height=None, width=None, channels=None):
        if height is not None:
            self.height = height

        if width is not None:
            self.width = width

        if channels is not None:
            self.channels = channels

        img = np.zeros((self.height, self.width, self.channels), np.uint8)

        # Set size scale
        if self.width < 64 or self.height < 64:
            raise RuntimeError("Width and Height of mask must be at least 64!")

        thickness = randint(15, 60)
        # Draw random lines
        for _ in range(randint(6, 12)):
            x1, x2 = randint(1, self.width), randint(1, self.width)
            y1, y2 = randint(1, self.height), randint(1, self.height)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        return 1 - img

    def sample(self, height=None, width=None, channels=None, random_seed=None):
        #if random_seed:
        #    seed(random_seed)
        return self._generate_mask(height, width, channels)


class ImageChunker(object):
    def __init__(self, rows, cols, overlap):
        self.rows = rows
        self.cols = cols
        self.overlap = overlap
    
    def perform_chunking(self, img_size, chunk_size):
        """
        Given an image dimension img_size, return list of (start, stop) 
        tuples to perform chunking of chunk_size
        """
        chunks, i = [], 0
        while True:
            chunks.append((i*(chunk_size - self.overlap/2), i*(chunk_size - self.overlap/2)+chunk_size))
            i+=1
            if chunks[-1][1] > img_size:
                break
        n_count = len(chunks)        
        chunks[-1] = tuple(x - (n_count*chunk_size - img_size - (n_count-1)*self.overlap/2) for x in chunks[-1])
        chunks = [(int(x), int(y)) for x, y in chunks]
        return chunks
    
    def get_chunks(self, img, scale=1):
        """
        Get width and height lists of (start, stop) tuples for chunking of img.
        """
        x_chunks, y_chunks = [(0, self.rows)], [(0, self.cols)]        
        if img.shape[0] > self.rows:
            x_chunks = self.perform_chunking(img.shape[0], self.rows)
        else:
            x_chunks = [(0, img.shape[0])]
        if img.shape[1] > self.cols:
            y_chunks = self.perform_chunking(img.shape[1], self.cols)
        else:
            y_chunks = [(0, img.shape[1])]
        return x_chunks, y_chunks    
    
    def dimension_preprocess(self, img, padding=True):
        """
        In case of prediction on image of different size than 512x512,
        this function is used to add padding and chunk up the image into pieces
        of 512x512, which can then later be reconstructed into the original image
        using the dimension_postprocess() function.
        """

        # Assert single image input
        assert len(img.shape) == 3, "Image dimension expected to be (H, W, C)"

        # Check if we are adding padding for too small images
        if padding:

            # Check if height is too small
            if img.shape[0] < self.rows:
                padding = np.ones((self.rows - img.shape[0], img.shape[1], img.shape[2]))
                img = np.concatenate((img, padding), axis=0)

            # Check if width is too small
            if img.shape[1] < self.cols:
                padding = np.ones((img.shape[0], self.cols - img.shape[1], img.shape[2]))
                img = np.concatenate((img, padding), axis=1)

        # Get chunking of the image
        x_chunks, y_chunks = self.get_chunks(img)

        # Chunk up the image
        images = []
        for x in x_chunks:
            images.extend(img[x[0]:x[1], y[0]:y[1], :] for y in y_chunks)
        images = np.array(images)
        return images
    
    def dimension_postprocess(self, chunked_images, original_image, scale=1, padding=True):
        """
        In case of prediction on image of different size than 512x512,
        the dimension_preprocess  function is used to add padding and chunk 
        up the image into pieces of 512x512, and this function is used to 
        reconstruct these pieces into the original image.
        """

        # Assert input dimensions
        assert len(original_image.shape) == 3, "Image dimension expected to be (H, W, C)"
        assert len(chunked_images.shape) == 4, "Chunked images dimension expected to be (B, H, W, C)"

        # Check if we are adding padding for too small images
        if padding:

            # Check if height is too small
            if original_image.shape[0] < self.rows:
                new_images = [img[0 : scale * original_image.shape[0], :, :] for img in chunked_images]

                chunked_images = np.array(new_images)

            # Check if width is too small
            if original_image.shape[1] < self.cols:
                new_images = [img[:, 0 : scale * original_image.shape[1], :] for img in chunked_images]

                chunked_images = np.array(new_images)

        # Put reconstruction into this array
        new_shape = (
            original_image.shape[0]*scale,
            original_image.shape[1]*scale,
            original_image.shape[2]
        )
        reconstruction = np.zeros(new_shape)

        # Get the chunks for this image    
        x_chunks, y_chunks = self.get_chunks(original_image)

        i = 0
        s = scale
        for x in x_chunks:
            for y in y_chunks:

                prior_fill = reconstruction != 0
                chunk = np.zeros(new_shape)
                chunk[x[0]*s:x[1]*s, y[0]*s:y[1]*s, :] += chunked_images[i]
                chunk_fill = chunk != 0

                reconstruction += chunk
                reconstruction[prior_fill & chunk_fill] = reconstruction[prior_fill & chunk_fill] / 2

                i += 1

        return reconstruction