The objective of this project is to create a helper module that will preprocess your image data for deep learning networks (resize, rotate, transform, etc.)

Features:
-Split the specified image directory files into train, test, val(optional)
-Resize image files into specified sizes
-**Crop portion of the images to a specified size (done with faces, should revisit the technique in the future)

As a note it is probably good to make sure that the memory (hard drive) on your computer can handle at least 2x of the current image directory size