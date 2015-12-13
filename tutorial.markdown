Left Ventricle Segmentation Tutorial
------------------------------------

*written by Alex Newton, Data Scientist, Booz Allen Hamilton*

<link rel="stylesheet" href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/8.9.1/styles/default.min.css"></link>
<link rel="stylesheet" href="highlight/styles/default.css"></link>
<script src="//cdnjs.cloudflare.com/ajax/libs/highlight.js/8.9.1/highlight.min.js"></script>
<script src="highlight/highlight.pack.js"></script>
<script>hljs.initHighlightingOnLoad();</script>

This tutorial will show you how to implement a fully automatic method for segmenting the left ventricle (LV) of the heart from cardiac MRI studies, and use the result to estimate the [end-diastolic] and [end-systolic] volumes to calculate [ejection fraction], an important metric of cardiovascular health.

1. [Overview](#overview)
2. [Limitations](#limitations)
3. [Dependencies](#dependencies)
4. [Walkthrough](#walkthrough)
  * [Step 1: Loading Datasets](#step1)
    * [Step 1.1: An overview of the segmentation process](#step1_1)
  * [Step 2: Calculate ROIs](#step2)
    * [Step 2.1: Regression Filtering](#step2_1)
    * [Step 2.2: Post-process regression](#step2_2)
    * [Step 2.3: Getting ROIs](#step2_3)
  * [Step 3: Calculate Areas](#step3)
    * [Step 3.1: Locating the LV Blood Pool](#step3_1)
      * [Step 3.1.1: Finding the Best Angle](#step3_1_1)
      * [Step 3.1.2: Finding the Threshold Point](#step3_1_2)
    * [Step 3.2: Propagating Segments](#step3_2)
  * [Step 4: Calculate Total Volume](#step4)
5. [Analysis](#analysis)
6. [Conclusion](#conclusions)

<a name=overview></a> Overview
------------------------------
This implementation follows the methodology laid out by [Lin et al. (2006)][lin-paper], with a few modifications and extensions to make it better suited for the current challenge. Notable differences from the original algorithm include:

1. The original paper assumed that the orientation of the heart in the image was given as a prior, when this is not the case in the DSB datasets. This implementation instead searches for the correct orientation by correlating lines sampled from the image with a [matched filter] which is based on our expectation of what the LV should look like.
2. The original paper was only concerned with the segmentation of MRI images, not the estimation of LV volumes; in order to produce a volume, this implementation uses metadata from the original DICOM files using the segmented LV regions from the MRI study as the bases for a series of conical frusta.
3. The original paper did not specify how it obtained a circular region of interest (ROI) after performing centroid regression. This implementation uses a metric based on optimizing the ratio between the proportion of a circle's contents and its radius.

<a name=limitations></a> Limitations
------------------------------------
To address the above concerns, I have taken a simplified modeling approach with a number of limitations:

1. The model only checks possible orientations going through the center of the ROI. If the centroid for a given slice ends up not having a line which goes through both ventricles, the threshold value will be meaningless. In addition, because the model checks along this line for a connected region to declare the left ventricle, not finding it can lead to picking an incorrect structure of drastically incorrect size, and propagating this mistake upwards during the propagation step.

2. The model's matched filter function is not as robust as it could be, and it although it usually finds some line going through the ventricles, it does not find what most observers would consider the "best" line; that is, a line going straight through the centers of both ventricles, leaving few possible candidates for the myocardial septum. 

Despite these limitations, this model has reasonably competent performance on some studies, and produces segmentations which can then be visually inspected in order to identify problem areas in the implementation.

<a name=dependencies></a> Dependencies
--------------------------------------
This implementation was written in Python 2.7.10 and relies on the following libraries:

*  [NumPy 1.9.2](http://www.numpy.org/)
*  [SciPy 0.15.1](http://scipy.org/)
*  [OpenCV 2 3.0.0](http://opencv.org/)
*  [matplotlib 1.4.3](http://matplotlib.org/)
*  [pydicom 0.9.9](http://www.pydicom.org/)

NumPy, SciPy, and matplotlib are all included in [Anaconda], a data science-focused Python platform which provides easy installation of all these libraries and more on several operating systems, and also comes with the extremely handy [IPython Notebook](http://ipython.org/notebook.html). OpenCV 2, on Windows, does not come pre-installed, but because Anaconda is still a full-featured Python installation, it's easy to install it via the [installer](http://opencv.org/downloads.html). Similarly, pydicom is a package available through [PyPI](https://pypi.python.org/pypi), so installing it is as simple as typing:

```bash
pip install pydicom
```

Please note that the implementation was written using a Windows Anaconda 2.3.0 installation. There's no guarantee that the listed versions of the above libraries will behave nicely in other operating systems.

[end-diastolic]: https://en.wikipedia.org/wiki/End-diastolic_volume
[end-systolic]: https://en.wikipedia.org/wiki/End-systolic_volume
[ejection fraction]: https://en.wikipedia.org/wiki/Ejection_fraction
[lin-paper]: http://www.researchgate.net/publication/6452142_Automated_Detection_of_Left_Ventricle_in_4D_MR_Images_Experience_from_a_Large_Study
[matched filter]: https://en.wikipedia.org/wiki/Matched_filter
[JSON]: https://en.wikipedia.org/wiki/JSON
[Anaconda]: https://www.continuum.io/why-anaconda

<a name=walkthrough></a> Walkthrough
-------------------------------------

Now that your installation is set up, we can begin walking through the code for the implementation, the entirety of which is available [here]. While you're at it, you should also grab the competition data [here] and extract it to some directory. This code will look for a `train.csv` file inside the competition data's directory, so make sure you download that as well.

This walkthrough will assume proficiency with Python and a familiarity with the basic workings of NumPy. To begin, we'll import everything we'll need for the rest of the code:

```python
#
# segment.py
#
import cv2
import numpy as np
import dicom
import json
import os
import shutil
import sys
import random
from matplotlib import image
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_erosion
from scipy.fftpack import fftn, ifftn
from scipy.signal import argrelmin, correlate
from scipy.spatial.distance import euclidean
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
```

Running this code fragment by itself is a good way to see if you have the dependencies listed above working. Next, we'll declare some constants that we'll use later.

```python
# number of bins to use in histogram for gaussian regression
NUM_BINS = 100
# number of standard deviations past which we will consider a pixel an outlier
STD_MULTIPLIER = 2
# number of points of our interpolated dataset to consider when searching for
# a threshold value; the function by default is interpolated over 1000 points,
# so 250 will look at the half of the points that is centered around the known
# myocardium pixel
THRESHOLD_AREA = 250
# number of pixels on the line within which to search for a connected component
# in a thresholded image, increase this to look for components further away
COMPONENT_INDEX_TOLERANCE = 20
# number of angles to search when looking for the correct orientation
ANGLE_SLICES = 36
```

The meaning of these may not be clear just yet, even from the comments, but they've all been put in one place so that they can be easily tweaked. Finally, a utility function so that the script produces readable output about its progress:

```python
def log(msg, lvl):
    string = ""
    for i in range(lvl):
        string += " "
    string += msg
    print string
```

### <a name=step1></a>Step 1: Loading datasets

The competition dataset consists of over 1,000 complete cardiac MRI series. These come with a [two chamber view](http://www.vhlab.umn.edu/atlas/cardiac-mri/2-chamber-right/index.shtml), a [four chamber view](http://www.vhlab.umn.edu/atlas/cardiac-mri/4-chamber/index.shtml), and a series of longitudinal slices perpendicular to the heart's long axis known as the [short-axis stack](https://www.med-ed.virginia.edu/courses/rad/cardiacmr/Anatomy/Short.html). For this tutorial we will only be using the short-axis stack.

The dataset is organized in a very regular manner. The top-level directory consists of a number of subdirectories, one for each patient. Inside each is a single folder, named "study", that in turn contains all of the available views for that MRI study. In particular, there's one two chamber view (prefixed with `2ch`), one four chamber view (prefixed with `4ch`), and a number of short-axis views comprising a short-axis stack (prefixed with `sax`). These do not follow a strict incremental scheme, but you can assume that their numerical ordering matches their spatial ordering, and that sequentially adjacent slices are equidistant throughout.

The top-level method `auto_segment_all_datasets()` is responsible for running the algorithm on all of the datasets, and it looks like this:

```python
def auto_segment_all_datasets():
    d = sys.argv[1]
    studies = next(os.walk(os.path.join(d, "train")))[1] + next(
        os.walk(os.path.join(d, "validate")))[1]

    labels = np.loadtxt(os.path.join(d, "train.csv"), delimiter=",",
                        skiprows=1)

    label_map = {}
    for l in labels:
        label_map[l[0]] = (l[2], l[1])

    num_samples = None
    if len(sys.argv) > 2:
        num_samples = int(sys.argv[2])
        studies = random.sample(studies, num_samples)
    if os.path.exists("output"):
        shutil.rmtree("output")
    os.mkdir("output")

    accuracy_csv = open("accuracy.csv", "w")
    accuracy_csv.write("Dataset,Actual EDV,Actual ESV,Predicted EDV,"
                       "Predicted ESV\n")
    submit_csv = open("submit.csv", "w")
    submit_csv.write("Id,")
    for i in range(0, 600):
        submit_csv.write("P%d" % i)
        if i != 599:
            submit_csv.write(",")
        else:
            submit_csv.write("\n")

    for s in studies:
        if int(s) <= 500:
            full_path = os.path.join(d, "train", s)
        else:
            full_path = os.path.join(d, "validate", s)

        dset = Dataset(full_path, s)
        print "Processing dataset %s..." % dset.name
        p_edv = 0
        p_esv = 0
        try:
            dset.load()
            segment_dataset(dset)
            if dset.edv >= 600 or dset.esv >= 600:
                raise Exception("Prediction too large")
            p_edv = dset.edv
            p_esv = dset.esv
        except Exception as e:
            log("***ERROR***: Exception %s thrown by dataset %s" % (str(e), dset.name), 0)
        submit_csv.write("%d_systolic," % int(dset.name))
        for i in range(0, 600):
            if i < p_esv:
                submit_csv.write("0.0")
            else:
                submit_csv.write("1.0")
            if i == 599:
                submit_csv.write("\n")
            else:
                submit_csv.write(",")
        submit_csv.write("%d_diastolic," % int(dset.name))
        for i in range(0, 600):
            if i < p_edv:
                submit_csv.write("0.0")
            else:
                submit_csv.write("1.0")
            if i == 599:
                submit_csv.write("\n")
            else:
                submit_csv.write(",")
        (edv, esv) = label_map.get(int(dset.name), (None, None))
        if edv is not None:
            accuracy_csv.write("%s,%f,%f,%f,%f\n" % (dset.name, edv, esv, p_edv, p_esv))

    accuracy_csv.close()
    submit_csv.close()
```

This function relies on the dataset directory given to the program on the command line, and searches it for subdirectories containing the studies. For example, a typical invocation (on the command line) might look like this, assuming you're running the script in the same directory as the top-level competition data folder:

```bash
$ python segment.py .
```

In addition, an optional argument can be supplied after the file. If supplied, this numerical argument indicates the size of a random sample to take from the dataset. At roughly 30 seconds per study, it takes several hours to run it on the full datasets; if you're just playing around with it, this is a good alternative to waiting around for the whole thing to finish:

```bash
$ python segment.py . 20
```

The above example chooses 20 of the studies at random and runs only those. The names of the studies in the output folder match those in the original dataset, so you can cross-reference them.

The `auto_segment_all_datasets` function will output two files in addition to the folder `output` containing segmented images and masks: `accuracy.csv` and `submission.csv`. `accuracy.csv` contains the correct ESV and EDV values for each dataset side-by-side with those predicted by the program; it is useful for evaluating the accuracy of the model.

For each entry in the `sets` list, we create a new `Dataset` object which is responsible for loading up one batch of files to give to the segmentation routine. It also sets up an "output" dir which will contain all of the segmented images, as well as an output file containing the end-systolic volume (ESV) and end-diastolic volume (EDV), and the computed EF.

Here is the `Dataset` class:

```python
class Dataset(object):
    dataset_count = 0

    def __init__(self, directory, subdir):
        # deal with any intervening directories
        while True:
            subdirs = next(os.walk(directory))[1]
            if len(subdirs) == 1:
                directory = os.path.join(directory, subdirs[0])
            else:
                break

        slices = []
        for s in subdirs:
            m = re.match("sax_(\d+)", s)
            if m is not None:
                slices.append(int(m.group(1)))

        slices_map = {}
        first = True
        times = []
        for s in slices:
            files = next(os.walk(os.path.join(directory, "sax_%d" % s)))[2]
            offset = None

            for f in files:
                m = re.match("IM-(\d{4,})-(\d{4})\.dcm", f)
                if m is not None:
                    if first:
                        times.append(int(m.group(2)))
                    if offset is None:
                        offset = int(m.group(1))

            first = False
            slices_map[s] = offset

        self.directory = directory
        self.time = sorted(times)
        self.slices = sorted(slices)
        self.slices_map = slices_map
        Dataset.dataset_count += 1
        self.name = subdir

    def _filename(self, s, t):
        return os.path.join(self.directory,"sax_%d" % s, "IM-%04d-%04d.dcm" % (self.slices_map[s], t))

    def _read_dicom_image(self, filename):
        d = dicom.read_file(filename)
        img = d.pixel_array
        return np.array(img)

    def _read_all_dicom_images(self):
        f1 = self._filename(self.slices[0], self.time[0])
        d1 = dicom.read_file(f1)
        (x, y) = d1.PixelSpacing
        (x, y) = (float(x), float(y))
        f2 = self._filename(self.slices[1], self.time[0])
        d2 = dicom.read_file(f2)

        # try a couple of things to measure distance between slices
        try:
            dist = np.abs(d2.SliceLocation - d1.SliceLocation)
        except AttributeError:
            try:
                dist = d1.SliceThickness
            except AttributeError:
                dist = 8  # better than nothing...

        self.images = np.array([[self._read_dicom_image(self._filename(d, i))
                                 for i in self.time]
                                for d in self.slices])
        self.dist = dist
        self.area_multiplier = x * y

    def load(self):
        self._read_all_dicom_images()
```

The Dataset class has two important functions. First, when it is initialized, it extrapolates a list of time values and slice numbers by walking through the directory given. Second, when called upon to load its data, it will figure out the proper paths and filenames, and combine the data from the different files into one big NumPy array. Here's a usage example:

```python
>>> from segment import *
>>> d = Dataset("train/27", "27")
>>> d.load()  # complete data now stored in d.images
```

The specifics of the datatypes here are important. In the `_read_dicom_image(filename)` method, the attribute `pixel_array` yields a (height x width) array of 16-bit integers representing grayscale intensities. In the list comprehension performed in `_read_all_dicom_images()`, this is performed for each time and each slice, so the resulting array (in the `images` attribute) will be a 4-dimensional NumPy array of shape (number of slices x number of times x height x width). As you will see soon, the ability to treat the entire dataset as a multidimensional array of intensities is crucial. 

The only other tricky part here is in `_read_all_dicom_images` where it obtains the value of `PixelSpacing` and computes the value of `dist`. These values are metadata which are kept inside the DICOM file headers, and they are assumed to be consistent across all images in the study. Note that, due to variance in MRI equipment, not all fields are present for every study, so a few methods are tried when calculating `dist`. We load this metadata because we'll need it in our final area/volume calculations to convert from the segmented pixels to millimeters and finally milliliters.

_Note:_ There are a couple of irregularities in the data that will cause the Dataset class to fail to load some of the studies. Specifically, there are (at least) two types of irregular data that will cause Dataset to fail:

1. **Short-axis slices with multiple viewports.** Some datasets (e.g. #123) contain in their SAX folders not only the short axis stack images, but a number of other views of the heart presented in sync. This presents a problem to Dataset, which expects there to be only one slice in any folder. In addition, it poses the problem of how to determine which of these views is the short-axis stack, which is beyond the scope of this tutorial.

Here's an example of how multi-view data looks in Mango, a popular DICOM viewer for Windows. The proper short-axis images are in the bottom right.

![Multiview Data](images/multiview_data.png)

2. **Studies where the size of images varies.** Some datasets (e.g. #148) change the dimension of the individual MRI images partway through the study. This causes NumPy, which relies on the data being the same size when it constructs the `Dataset.images` array, to improperly define the shape. Fixing this problem would require resizing each individual image, either by cropping or by padding them out, further complicating the `Dataset` code and adding another avenue for potential error.

### <a name=step1_1></a> Step 1.1: An overview of the segmentation process

The only function we have not yet defined above is `segment_dataset`, and it's an important one: this function is responsible for coordinating the entire algorithm over each dataset. Because of this, it also conveniently serves as a sort of outline for the segmentation process, so it is worth going over in detail.

```python
def segment_dataset(dataset):
    images = dataset.images
    dist = dataset.dist
    areaMultiplier = dataset.area_multiplier
    # shape: num slices, num snapshots, rows, columns
    log("Calculating rois...", 1)
    rois, circles = calc_rois(images)
    log("Calculating areas...", 1)
    all_masks, all_areas = calc_all_areas(images, rois, circles)
    log("Calculating volumes...", 1)
    area_totals = [calc_total_volume(a, areaMultiplier, dist)
                   for a in all_areas]
    log("Calculating ef...", 1)
    edv = max(area_totals)
    esv = min(area_totals)
    ef = (edv - esv) / edv
    log("Done, ef is %f" % ef, 1)

    save_masks_to_dir(dataset, all_masks)

    output = {}
    output["edv"] = edv
    output["esv"] = esv
    output["ef"] = ef
    output["areas"] = all_areas.tolist()
    f = open("output/%s/output.json" % dataset.name, "w")
    json.dump(output, f, indent=2)
    f.close()
    dataset.edv = edv
    dataset.esv = esv
    dataset.ef = ef
```

The fundamental innovation of this algorithm is to realize that, in any given series of MRI images for a single slice, the only part which will be changing significantly is the heart itself; therefore, we can treat the series of images for a slice as a 2-dimensional signal which varies over time, or a 2D+T signal, and determine which pixels have the strongest response to the cardiac frequency by applying a [Fourier Transform](https://en.wikipedia.org/wiki/Fourier_transform) and examining the harmonics. This is how the algorithm automates the process of determining ROI from the complete image, cutting out most of the entire image and leaving only a circle-shaped region of nonzero pixels believed to contain the heart. This is all accomplished in the `calc_rois(images)` function.

Once a region of interest has been identified, the next step is to separate the blood (which tends to have high intensities) from the myocardial tissue (which tends to have low intensities). The method for doing this is to first identify a line which goes through both ventricles and the interventricular septum, then use the intensities along that line to find a point on the border between blood and tissue.

The value at this point is then used to threshold the original images, and the connected component closest to this point near the image centroid is taken to be the LV. The component is then projected onto the rest of the slices with the goal of finding regions of similar size and location, which will serve as segmentations for the LV in other slices. This entire computation--going from ROIs to segmentations--is accomplished in the `calc_all_areas(images, rois, circles)` function.

Finally, once the segmentation is complete, and the LV is segmented on each image of each slice, the volume of the LV as a whole at each time is computed using the DICOM metadata retrieved back in step 1 in the function `calc_total_volume(a, areaMultiplier, dist)`. The end-diastolic volume (EDV) is when the LV is at its most expanded, so it's taken to be the maximum of these values; the end-systolic volume (ESV) is when it's at its most contracted, so it's taken to be the minimum. The EF is then calculated in the standard fashion (EF = (EDV - ESV) / EDV). For the user's convenience, this (and the list of pixel-wise areas for each slice, for each time) is saved to a file called "output.json" in the output folder corresponding to that dataset.

(In addition, the segmentation masks, and the original segmented images with the LV segment outlined are saved in the function `save_masks_to_dir` which is not as critical to the main algorithm and will be laid out later.)

This is a lot to take in all at once, so the next three steps will focus on the three primary functions called during `segment_dataset`: namely, `calc_rois`, `calc_all_areas`, and `calc_total_volume`.

### <a name=step2></a> Step 2: Calculate ROIs

Calculating the ROI is a crucial step in the image segmentation algorithm. Reducing the image to the ROI generally reduces the number of pixels under consideration by approximately an order of magnitude, which can make a huge difference in both processing speed and accuracy of the result. In this implementation, the process of determining ROIs is performed in the function `calc_rois(images)`, which is given below:

```python
def calc_rois(images):
    (num_slices, _, _, _) = images.shape
    log("Calculating mean...", 2)
    dc = np.mean(images, 1)

    def get_H1(i):
        log("Fourier transforming on slice %d..." % i, 3)
        ff = fftn(images[i])
        first_harmonic = ff[1, :, :]
        log("Inverse Fourier transforming on slice %d..." % i, 3)
        result = np.absolute(ifftn(first_harmonic))
        log("Performing Gaussian blur on slice %d..." % i, 3)
        result = cv2.GaussianBlur(result, (5, 5), 0)
        return result

    log("Performing Fourier transforms...", 2)
    h1s = np.array([get_H1(i) for i in range(num_slices)])
    m = np.max(h1s) * 0.05
    h1s[h1s < m] = 0

    log("Applying regression filter...", 2)
    regress_h1s = regression_filter(h1s)
    log("Post-processing filtered images...", 2)
    proc_regress_h1s, coords = post_process_regression(regress_h1s)
    log("Determining ROIs...", 2)
    rois, circles = get_ROIs(dc, proc_regress_h1s, coords)
    return rois, circles
```

To take a look at the effects of `calc_rois`, you can use this example code:

```python
>>> from segment import *
>>> d = Dataset("train/27", "000000")
>>> d.load()
>>> image_example = d.images[0][0]  # image from first slice at first time
>>> image.imsave("image_example.png", image_example)  # from matplotlib.image imported in segment
>>> rois, circles = calc_rois(d.images)
  Calculating mean...
  Performing Fourier transforms...
# ...
# lots of calculations that take a while
# ...
   Getting ROI in slice 9...
   Getting ROI in slice 10...
>>> roi_example = rois[0]  # region of interest in first slice (this is a subset of the dc image)
>>> image.imsave("roi_example.png", roi_example)
>>> print circles[0]  # coordinates of circle against which ROI was taken
[(69.50797361879323, 103.31905719068067) 46]
```

This code results in two images being saved. Here is the first, "image_example.png":

![Image example](images/image_example.png)

Notice that the high intensity noise around the borders of the image is very large compared to the important signal here, which is the left ventricle. Due to the linear color palette employed by matplotlib's visualizations, it is very difficult to visually distinguish the LV from its surroundings. The result of the `calc_rois` method makes it much easier to focus:

![ROI example](images/roi_example.png)

The intensities in this image are the same, but because everything outside of the ROI has been set to 0, the contrast between the LV and its immediate surroundings is much more pronounced. Note that compared to the original image, the boundaries of these structures seem longer and blurrier; this is because the ROI is taken inside the DC component of the original signal.

The [DC component](https://en.wikipedia.org/wiki/DC_bias) of the signal is equal to the mean grayscale value for each pixel in each slice over time; it is the first thing calculated in the `calc_rois` method. NumPy makes this very easy by allowing us to take the mean over an arbitrary axis; in this case, taking the mean over the second axis reduces the shape of image (number of slices x number of times x height x width) to just (number of slices x height x width), averaging the individual intensities while preserving the axes.

A typical DC image might look something like this:

![DC example](images/dc_example.png)

Note that the borders of the LV and RV are blurry and indistinct; this is because these structures, unlike most of the image, change size and shape over time.

Next, we need to calculate the H1 component of each slice (where H stands for [Hilbert space](https://en.wikipedia.org/wiki/Hilbert_space)), which is done in the `get_H1` function. We treat each slice as a separate 2D+T signal of shape (number of times x height x width), and take the Fourier transform over the first axis. Since our original signal consists of a discrete and finite set of samples, we can use the `fftn` (n-dimensional fast Fourier transform) given in SciPy's `fftpack` module. Then, we use only the first harmonic of the transform and transform it back into data about our original signal using the `ifftn`, which reverses the transformation.

The result of `ifftn` is a complex number, but because our original signal was real the angle information represented by the elements of the result is useless. We only care about the magnitude, so np.absolute is used to reduce the complex numbers to real ones. Finally, we use OpenCV to smooth the resulting image. Once all the H1 components for the slices are collected, values less than 5% of the maximum throughout the whole volume are discarded to reduce noise.

The H1 images look like this:

![H1 example](images/h1_example.png)

Note that nearly the entire chest cavity, save the heart and some adjacent structures, has vanished from the image; even the high-intensity noise around the edges which is visible in the DC image has vanished completely. The reason for this is because the H1 component represents the response of each pixel to the fundamental 2D+T signal; in other words, we assume that the heartbeat can be modeled as a waveform, and that the MRI series captures exactly one period. Then, if we consider a pixel just outside the LV blood pool at end systole, it goes from dark (myocardium) to light (inside the blood pool) at end diastole, because the region containing blood expands outwards. The pixel gets darker again as the heart approaches end systole. This means that a graph of said pixel's intensity will resemble a waveform with the very same frequency as the heartbeat, thus it will have a large response to this frequency in the Fourier transform.

### <a name=step2_1></a> Step 2.1: Regression Filtering

Although acquiring the H1 component has done a great deal of the work in determining the region of interest, it still leaves something to be desired when it comes to noise. From the example given above, you can see that there are still some nonzero pixels in the bottom right area of the image. From my (admittedly rather limited) understanding of the depicted anatomy, this is due to the fact that other blood-carrying structures such as the aorta also grow and shrink according to the cardiac frequency, albeit not as drastically. The method I adopted, again taken from Lin et al. (2006), is to assume that the relevant points (weighted by intensity) are normally distributed about a 3D line going through the center of the heart, and to filter them accordingly.

This implementation performs this process in the `regression_filter(imgs)` function, given here:

```python
def regression_filter(imgs):
    condition = True
    iternum = 0
    while(condition):
        log("Beginning iteration %d of regression..." % iternum, 3)
        iternum += 1
        imgs_filtered = regress_and_filter_distant(imgs)
        c1 = get_centroid(imgs)
        c2 = get_centroid(imgs_filtered)
        dc = np.linalg.norm(c1 - c2)
        imgs = imgs_filtered
        condition = (dc > 1.0)  # because python has no do-while loops
    return imgs
```

The general idea behind this function is to iteratively refine our guess for the 3D line by removing pixels that are far away from it. This process of removing individual pixels from the H1 images continues until the 3D centroid of the stack (defined as the average coordinates, weighted by pixel intensity) no longer moves significantly; that is, the distance between the old centroid and the new centroid after removing pixels is less than 1.0. Here are the functions that calculate centroids:

```python
def get_centroid(img):
    nz = np.nonzero(img)
    pxls = np.transpose(nz)
    weights = img[nz]
    avg = np.average(pxls, axis=0, weights=weights)
    return avg
```

Note the use of `np.transpose` on the result of `np.nonzero`. This is a standard idiom and is [suggested](http://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html) in the documentation for NumPy's `nonzero` function. It allows us to get an array of coordinates, and will be used several times throughout the code.

Also note that NumPy's flexibility allows us to calculate the centroid regardless of whether you pass `get_centroid` a single 2D image or a 3D stack of images, and it will be used as such depending on the context.

The workhorse function in `regression_filter` is `regress_and_filter_distant`, which takes as its input a stack of images and returns the same stack of images but with distant pixels removed. It looks like this:

```python
def regress_and_filter_distant(imgs):
    centroids = np.array([get_centroid(img) for img in imgs])
    raw_coords = np.transpose(np.nonzero(imgs))
    (xslope, xintercept, yslope, yintercept) = regress_centroids(centroids)
    (coords, dists, weights) = get_weighted_distances(imgs, raw_coords, xslope,
                                                      xintercept, yslope,
                                                      yintercept)
    outliers = get_outliers(coords, dists, weights)
    imgs_cpy = np.copy(imgs)
    for c in outliers:
        (z, x, y) = c
        imgs_cpy[z, x, y] = 0
    return imgs_cpy
```

First, the centroid of each slice is determined; then a line is fitted through them via linear regression. After that, the distance from each pixel to the line is calculated, and these distances are taken along with the intensities of each pixel as weights to form a weighted histogram. Finally, distant outlier pixels are removed from the image by having their intensity set to 0.

`regress_centroids(centroids)` handles the first part of this process:

```python
def regress_centroids(cs):
    num_slices = len(cs)
    y_centroids = cs[:, 0]
    x_centroids = cs[:, 1]
    z_values = np.array(range(num_slices))

    (xslope, xintercept, _, _, _) = linregress(z_values, x_centroids)
    (yslope, yintercept, _, _, _) = linregress(z_values, y_centroids)

    return (xslope, xintercept, yslope, yintercept)
```

It's a relatively simple function that takes the x-coordinates and y-coordinates of all the centroids. The slices are each assigned to a z-coordinate, and the 3D line is assembled from the parameters of linear regression between x and z and between y and z.

Once this line is calculated, its parameters are passed into `get_weighted_distances(imgs, coords, xs, xi, ys, yi)`: 

```python
def get_weighted_distances(imgs, coords, xs, xi, ys, yi):
    a = np.array([0, yi, xi])
    n = np.array([1, ys, xs])

    zeros = np.zeros(3)

    def dist(p):
        to_line = (a - p) - (np.dot((a - p), n) * n)
        d = euclidean(zeros, to_line)
        return d

    def weight(p):
        (z, y, x) = p
        return imgs[z, y, x]

    dists = np.array([dist(c) for c in coords])
    weights = np.array([weight(c) for c in coords])
    return (coords, dists, weights)
```

This function uses a bit of basic linear algebra to calculate the distance from each nonzero pixel to the line. It returns a list of coordinates, and same-sized lists of distances and weights.

Once the distances and weights are in hand, the function `get_outliers(coords dists, weights)` is called to determine which are far away:

```python
def get_outliers(coords, dists, weights):
    fivep = int(len(weights) * 0.05)
    ctr = 1
    while True:
        (mean, std, fn) = gaussian_fit(dists, weights)
        low_values = dists < (mean - STD_MULTIPLIER*np.abs(std))
        high_values = dists > (mean + STD_MULTIPLIER*np.abs(std))
        outliers = np.logical_or(low_values, high_values)
        if len(coords[outliers]) == len(coords):
            weights[-fivep*ctr:] = 0
            ctr += 1
        else:
            return coords[outliers]
```

`get_outliers` contains code that tries to cut off some data points and try again if outliers cause the Gaussian function to match something completely nonsensical and make the entirety of the given points outliers. Another version of this technique is used within the `gaussian_fit` function below in case the regression doesn't converge.

A [Gaussian function](https://en.wikipedia.org/wiki/Gaussian_function) is fitted over the data, then all points which are 2 standard deviations above or below the mean (assuming they are normally distributed, this means about 5% of them) are removed. The function that fits the Gaussian, `gaussian_fit(dists, weights)` looks like this:

```python
def gaussian_fit(dists, weights):
    # based on http://stackoverflow.com/questions/11507028/fit-a-gaussian-function
    (x, y) = histogram_transform(dists, weights)
    fivep = int(len(x) * 0.05)
    xtmp = x
    ytmp = y
    fromFront = False
    while True:
        if len(xtmp) == 0 and len(ytmp) == 0:
            if fromFront:
                # well we failed
                idx = np.argmax(y)
                xmax = x[idx]
                p0 = [max(y), xmax, xmax]
                (A, mu, sigma) = p0
                return mu, sigma, lambda x: gauss(x, A, mu, sigma)
            else:
                fromFront = True
                xtmp = x
                ytmp = y

        idx = np.argmax(ytmp)
        xmax = xtmp[idx]

        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))

        p0 = [max(ytmp), xmax, xmax]
        try:
            coeff, var_matrix = curve_fit(gauss, xtmp, ytmp, p0=p0)
            (A, mu, sigma) = coeff
            return (mu, sigma, lambda x: gauss(x, A, mu, sigma))
        except RuntimeError:
            if fromFront:
                xtmp = xtmp[fivep:]
                ytmp = ytmp[fivep:]
            else:
                xtmp = xtmp[:-fivep]
                ytmp = ytmp[:-fivep]
```

First, the data is assembled into a weighted histogram (as in, each bin is the total weights of distances falling into that bin). Then, SciPy's general optimization function, curve_fit, is used to fit the three parameters of the Gaussian over the given data. It rarely occurs that the curve_fit function will be unable to find a suitable function within a certain number of iterations, in which case it throws a `RuntimeException`. In my experience, this is usually because there's a distant cluster of points on one of the "tails" of the function forming a small "hump.” The algorithm first tries coercing the function into a more Gaussian shape by removing the most distant points. If that doesn't work, it tries again with the closest points. 

The last piece necessary to complete the regression function is the `histogram_transform` function:

```python
def histogram_transform(values, weights):
    hist, bins = np.histogram(values, bins=NUM_BINS, weights=weights)
    bin_width = bins[1] - bins[0]
    bin_centers = bins[:-1] + (bin_width / 2)

    return (bin_centers, hist)
```

By default, NumPy's `histogram` function returns the bin edges; that is, the values are grouped so that `hist[0]` contains values falling between `bins[0]` and `bins[1]`, `hist[1]` contains those falling between `bins[1]` and `bins[2]`, _et cetera_. We need to change this from bin edges to bin centers; this is accomplished by taking all but the last edge (i.e., the left edges of every bin) and offsetting them by the width of one bin.

### <a name=step2_2></a> Step 2.2: Post-process regression

But wait, there's more! Before we say goodbye to the method of centroid regression, we need to filter the H1 slices one last time. This time, instead of considering the distance of all nonzero pixels and assembling one large distribution, we assemble a distribution for each slice and filter them separately. Thankfully our code is written in a modular enough fashion that we can assemble a function that will accomplish this, `post_process_regression(imgs)`, entirely out of the previous functions and library functions:

```python
def post_process_regression(imgs):
    (numimgs, _, _) = imgs.shape
    centroids = np.array([get_centroid(img) for img in imgs])
    log("Performing final centroid regression...", 3)
    (xslope, xintercept, yslope, yintercept) = regress_centroids(centroids)
    imgs_cpy = np.copy(imgs)

    def filter_one_img(zlvl):
        points_on_zlvl = np.transpose(imgs[zlvl].nonzero())
        points_on_zlvl = np.insert(points_on_zlvl, 0, zlvl, axis=1)
        (coords, dists, weights) = get_weighted_distances(imgs, points_on_zlvl,
                                                          xslope, xintercept,
                                                          yslope, yintercept)
        outliers = get_outliers(coords, dists, weights)
        for c in outliers:
            (z, x, y) = c
            imgs_cpy[z, x, y] = 0

    log("Final image filtering...", 3)
    for z in range(numimgs):
        log("Filtering image %d of %d..." % (z+1, numimgs), 4)
        filter_one_img(z)

    return (imgs_cpy, (xslope, xintercept, yslope, yintercept))
```

The only part of this which should seem arcane here is the use of `np.insert`; it's only necessary because the get_weighted_distances function requires its points to have z-values, and because we called `imgs[zlvl].nonzero()` instead of `imgs.nonzero()` we only get two coordinates instead of three, and have to add the third ourselves.

### <a name=step2_3></a> Step 2.3: Getting ROIs

Armed with a list of relevant pixels and a line of regression going through them, we can now calculate circular ROIs on each of the DC images. The function which takes care of this is called `get_ROIs(originals, h1s, regression_params)`:

```python
def get_ROIs(originals, h1s, regression_params):
    (xslope, xintercept, yslope, yintercept) = regression_params
    (num_slices, _, _) = h1s.shape
    results = []
    circles = []
    for i in range(num_slices):
        log("Getting ROI in slice %d..." % i, 3)
        o = originals[i]
        h = h1s[i]
        ctr = (xintercept + xslope * i, yintercept + yslope * i)
        r = circle_smart_radius(h, ctr)
        tmp = np.zeros_like(o)
        floats_draw_circle(tmp, ctr, r, 1, -1)
        results.append(tmp * o)
        circles.append((ctr, r))

    return (np.array(results), np.array(circles))
```

The parameters to this function are the original DC images, the filtered H1 images we've been computing, and the parameters for the final 3D line drawn through them. Each ROI will be centered around the point where the line intersects that slice; the only task that remains for us is to determine the radius. A suitable criterion for the circle defined by the radius is that it should be the largest possible radius, but it must also maximize the ratio of nonzero pixels to pixels, or try to be as "full" as possible. I chose the product of these two quantities: radius times the proportion of nonzero pixels in the circle. This is accomplished in the `circle_smart_radius(img, center)` function, and a helper called `filled_ratio_of_circle(img, center, r)`:

```python
def circle_smart_radius(img, center):
    domain = np.arange(1, 100)
    (xintercept, yintercept) = center

    def ratio(r):
        return filled_ratio_of_circle(img, (xintercept, yintercept), r)*r

    y = np.array([ratio(d) for d in domain])
    most = np.argmax(y)
    return domain[most]

def filled_ratio_of_circle(img, center, r):
    mask = np.zeros_like(img)
    floats_draw_circle(mask, center, r, 1, -1)
    masked = mask * img
    (x, _) = np.nonzero(mask)
    (x2, _) = np.nonzero(masked)
    if x.size == 0:
        return 0
    return float(x2.size) / x.size
```

This function calculates the metric for all radii between 1 and 100, then returns the radius with the highest metric. Experimentally, this metric tends to be highest around 70 to 80 for most ROIs. If we insert a command to graph each candidate radius against its metric, we get a graph that might look something like this:

![Smart radius example](images/smart_radius_graph.png)

The thought process behind this metric was that we should obtain a circle that is mostly signal, in the hopes that we can keep only relevant structures in the ROI, but should also endeavor to be as large as possible, so we capture all relevant structures.

Both `filled_ratio_of_circle` and `get_ROIs` use a utility function called `floats_draw_circle` which behaves like OpenCV's `circle` function, drawing a circle of the given color into a two-dimensional array, but takes floating point arguments and converts them into integers via rounding:

```python
def floats_draw_circle(img, center, r, color, thickness):
    (x, y) = center
    x, y = int(np.round(x)), int(np.round(y))
    r = int(np.round(r))
    cv2.circle(img, center=(x, y), radius=r, color=color, thickness=thickness)
```

The result of this entire computation is a set of the DC images with all pixels outside the ROI removed; it looks something like this:

![ROI example](images/roi_example.png)

### <a name=step3></a> Step 3: Calculate Areas

Once we have calculated ROI for all slices, we can begin actual segmentation. This is accomplished in the function `calc_all_areas(images, rois, circles)`:

```python
def calc_all_areas(images, rois, circles):
    closest_slice = get_closest_slice(rois)
    (_, times, _, _) = images.shape

    def calc_areas(time):
        log("Calculating areas at time %d..." % time, 2)
        mask, mean = locate_lv_blood_pool(images, rois, circles, closest_slice,
                                          time)
        masks, areas = propagate_segments(images, rois, mask, mean,
                                          closest_slice, time)
        return (masks, areas)

    result = np.transpose(map(calc_areas, range(times)))
    all_masks = result[0]
    all_areas = result[1]
    return all_masks, all_areas
```

Calling this function is easy once you have the results of `calc_rois` available. It takes the original images, the ROI images, and a list of circles (in the same format as output by calc_rois, which is to say a list of ((x, y), r) tuples), and returns two n-dimensional arrays of masks and areas (in pixels), respectively, with dimensions (time x slice). Here is an example invocation:

```python
>>> from segment import *
>>> d = Dataset("train/27", "27")
>>> d.load()
>>> rois, circles = calc_rois(d.images)
#
# ... some log messages are printed ...
#
>>> masks, areas = calc_all_areas(d.images, rois, circles)
>>> image.imsave("mask_example.png", masks[0][0])
>>> print areas[0][0]
3009
>>> np.count_nonzero(masks[0][0]) == areas[0][0]
True
```

The output, "mask_example.png", looks like this:

![Mask example](images/mask_example.png)

Each mask array, such as `masks[0][0]` which was saved to "mask_example.png" in the code above, is a single-channel 2D image with the dimensions of the original study. The only values in this array are 255 and 0; values of 255 indicate that the corresponding pixel in the original is part of the left ventricle, and values of 0 indicate that it is not (due to matplotlib's color scale functionality, this is rendered to the file on a scale from blue to red). Using the code in `save_masks_to_dir`, we can draw these solid masks as outlines on top of the original black and white images, as in this example:

![Hollow mask example](images/hollow_mask_example.png)

It can also be confirmed that the areas are just the number of nonzero pixels in the mask.

The idea behind segmenting here is to locate the LV blood pool on the middle slice by finding a threshold value, `mean`, and converting all slices at that time to binary images by thresholding at that value. Then, the algorithm selects one of those regions to be the LV and "propagates" upward and downward by selecting the connected component most similar to the previous, using the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index), resulting in a set of binary images that comprise the segmentation proper, and a list of areas (which is just the number of pixels in each segment). This is repeated for each time, collected, and returned.

Before anything else is calculated, the get_closest_slice utility function determines which slice is closest to the 3d centroid:

```python
def get_closest_slice(rois):
    ctrd = get_centroid(rois)
    closest_slice = int(np.round(ctrd[0]))
    return closest_slice
```

### <a name=step3_1></a> Step 3.1: Locating the LV Blood Pool

As mentioned above, the first step in the process of segmenting the image stack is to locate the LV blood pool on the middle slice. The function responsible for this, and for coming up with a threshold value, is `locate_lv_blood_pool(images, rois, circles, closest_slice, time)`:

```python
def locate_lv_blood_pool(images, rois, circles, closest_slice, time):
    best, best_fn = find_best_angle(rois[closest_slice],
                                    circles[closest_slice])
    mean, coords, idx = find_threshold_point(best, best_fn)
    thresh, img_bin = cv2.threshold(images[closest_slice,
                                           time].astype(np.float32),
                                    mean, 255.0, cv2.THRESH_BINARY)
    labeled, num = label(img_bin)
    x, y = coords[idx]

    count = 0
    # Look along the line for a component. If one isn't found within a certain
    # number of indices, just spit out the original coordinate.
    while labeled[y][x] == 0:
        idx += 1
        count += 1
        x, y = coords[idx]
        if count > COMPONENT_INDEX_TOLERANCE:
            idx -= count
            x, y = coords[idx]
            break

    if count <= COMPONENT_INDEX_TOLERANCE:
        component = np.transpose(np.nonzero(labeled == labeled[y][x]))
    else:
        component = np.array([[y, x]])

    hull = cv2.convexHull(component)
    squeezed = hull
    if count <= COMPONENT_INDEX_TOLERANCE:
        squeezed = np.squeeze(squeezed)
    hull = np.fliplr(squeezed)

    mask = np.zeros_like(labeled)
    cv2.drawContours(mask, [hull], 0, 255, thickness=-1)
    return mask, mean
```

`locate_lv_blood_pool` is called once for each time in the study, on the closest slice to the centroid only. It is called like so:

```
>>> from segment import *
>>> d = Dataset("train/27", "27")
>>> d.load()
>>> rois, circles = calc_rois(d.images)
>>> closest_slice = get_closest_slice(rois)  # slice 4
>>> mask, mean = locate_lv_blood_pool(d.images, rois, circles, closest_slice, 0)
>>> image.imsave("blood_pool_mask.png", mask)
>>> print mean
134.883047414
```

Here's the contents of "blood_pool_mask.png":

![Blood pool mask](images/blood_pool_mask.png)

The mask, in "blood_pool_mask.png", looks similar to the one finally returned in `calc_all_areas`; in fact, it's the one stored in `masks[0][4]` of that same array. The difference between the two masks is that the one calculated in `locate_lv_blood_pool` is calculated from "scratch", so to speak, and then all the masks for the slices above and below it are created from it via the `propagate_segment` function, which is treated in detail below. In addition, it's required to threshold all these images via a value calculated in `locate_lv_blood_pool`, which is returned as `mean`.

One of the big departures of this implementation from the original paper is that the orientation of the heart is not taken as a prior. To compensate for this fact, we sample a variety of different angles for the orientation and choose the one which most closely matches a prior idea of what a line through both ventricles should look like. 

This is accomplished by the `find_best_angle(rois, circles)` function; the result is an angle in radians, `best`, and a tuple consisting of a spline function, its domain, and the coordinate corresponding to each element in the domain, referred to as `best_fn`. These are passed into `find_threshold_point(best, best_fn)` which picks the point from the line where the intensity increases most rapidly (i.e., going from dark myocardium to bright blood) returns that point. It also returns the coordinates around it.

The image is then thresholded accorded to this value, and connected components are labeled. Here is an image demonstrating the result of thresholding and labeling (where different colors represent unique components):

![Labeling Example](images/thresh_labeled.png)

The remainder of the function is dedicated to finding the closest component intersected by the line, in the direction of the line, by going through each coordinate of the line and using the first component it finds, up to a threshold given by `COMPONENT_INDEX_TOLERANCE`. The threshold is necessary because if there is no detected LV component or the orientation is far off the mark, it will generally travel to the far end of the image and select a completely irrelevant region. If it fails, the region is considered to be a single pixel at the location of the threshold point, in the hopes that it might propagate upward to an actual region in the next step and salvage some of the missing volume. It then draws a convex hull around this component, and returns a binary image with the pixels inside the contour set.

### <a name=step3_1_1></a> Step 3.1.1: Finding the Best Angle

Let's dive into more detail about the `find_best_angle` function:

```python
def find_best_angle(roi, circ):
    ((cx, cy), r) = circ
    results = np.zeros(ANGLE_SLICES)
    fns = [None for i in range(ANGLE_SLICES)]

    COS_MATCHED_FILTER_FREQ = 2.5

    def score_matched(trimx, trimy):
        # first, normalize this data
        newtrimx = np.linspace(0.0, 1.0, np.size(trimx))
        minimum = np.min(trimy)
        maximum = np.max(trimy) - minimum
        newtrimy = (trimy - minimum) / maximum

        filt = 1 - ((np.cos(COS_MATCHED_FILTER_FREQ*2*np.pi*newtrimx)) /
                    2 + (0.5))
        cr = correlate(newtrimy, filt, mode="same")
        return np.max(cr)

    for i in range(ANGLE_SLICES):
        trimx, trimy, trimcoords = get_line(roi, cx, cy, np.pi*i/ANGLE_SLICES)
        score2 = score_matched(trimx, trimy)
        results[i] = score2
        fns[i] = (UnivariateSpline(trimx, trimy), trimx, trimcoords)

    best = np.argmax(results)
    return (best * np.pi / ANGLE_SLICES, fns[best])
```

The goal of `find_best_angle` is to populate the `results` and `fns` array with one score and one function, respectively, for each of `ANGLE_SLICES` different angles, on the range [0, pi). (It is unnecessary to calculate any other angles as they would just mirror an existing angle.) It does this by running the "score_matched" function over each angle and then selecting the angle with maximum score. It also calculates an interpolated spline function which will be used later.

The `score_matched` function assigns a score to each angle based on a matched filter. By default, this is a cosine function oscillating between 0 and 1 with frequency `COS_MATCHED_FILTER_FREQ` which defaults to 2.5, which has two large high-intensity "hills" (corresponding to the peaks in intensity for the RV and LV blood pools) and then a sort of tail which covers frequently encountered structures adjacent to the ventricles (see the ROI example at the end of step 2 for an example).

The only function that this function depends on is `get_line`, which draws a line with a certain angle through a point and samples the points along it. The result is a list of data points (`trimx`, `trimy`) representing all the pixel intensities along the line, in the order in which they were sampled. `trimcoords` is an array of the same size which gives a pair of image coordinates corresponding to each sampled pixel.

Here is `get_line`:

```python
def get_line(roi, cx, cy, theta):
    (h, w) = roi.shape
    (x0, x1, y0, y1) = get_line_coords(w, h, cx, cy, theta)

    intensities = []
    coords = []

    def collect(x, y):
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        intensities.append(roi[y, x])
        coords.append((x, y))

    bresenham(x0, x1, y0, y1, collect)

    def geti(idx):
        return intensities[idx]

    getiv = np.vectorize(geti)
    x = np.arange(0, len(intensities))
    y = getiv(x)
    first, last = trim_zeros_indices(y)
    trimy = y[first:last]
    trimcoords = coords[first:last]

    trimx = np.arange(0, trimy.size)
    return (trimx, trimy, trimcoords)
```

`get_line_coords` is a method which converts a point and an angle through it to a line given by integer start and end coordinates, bounded by `w` and `h`. `get_line_coords`, as well as `line_thru` which it depends on, are given here:

```python
def line_thru(bounds, center, theta):
    (xmin, xmax, ymin, ymax) = bounds
    (cx, cy) = center

    if np.cos(theta) == 0:
        return (cx, ymin, cx, ymax)
    slope = np.tan(theta)

    x0 = xmin
    y0 = cy - (cx - xmin) * slope
    if y0 < ymin:
        y0 = ymin
        x0 = max(xmin, cx - ((cy - ymin) / slope))
    elif y0 > ymax:
        y0 = ymax
        x0 = max(xmin, cx - ((cy - ymax) / slope))

    x1 = xmax
    y1 = cy + (xmax - cx) * slope
    if y1 < ymin:
        y1 = ymin
        x1 = min(xmax, cx + ((ymin - cy) / slope))
    elif y1 > ymax:
        y1 = ymax
        x1 = min(xmax, cx + ((ymax - cy) / slope))

    return (x0, x1, y0, y1)


def get_line_coords(w, h, cx, cy, theta):
    coords = np.floor(np.array(line_thru((0, w-1, 0, h-1), (cx, cy), theta)))
    return coords.astype(np.int)
```

Once the coordinates are obtained, [Bresenham's line algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm) is used to sample points from the line. The reason that a custom implementation of Bresenham's algorithm is used instead of (for example) `cv2.line` is because it is important to establish an ordering between the drawn points, so that we can treat the sampling across the line as a function and subsequently use it in cross-correlation. Here is my implementation of Bresenham's algorithm:

```python
def bresenham(x0, x1, y0, y1, fn):
    # using some pseudocode from
    # https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm
    # and also https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    steep = abs(y1-y0) > abs(x1-x0)
    if steep:
        x0, x1, y0, y1 = y0, y1, x0, x1
    if x0 > x1:
        x0, x1, y0, y1 = x1, x0, y1, y0

    def plot(x, y):
        if steep:
            fn(y, x)
        else:
            fn(x, y)

    dx = x1 - x0
    dy = y1 - y0

    D = 2*np.abs(dy) - dx
    plot(x0, y0)
    y = y0

    for x in range(x0+1, x1+1):  # x0+1 to x1
        D = D + 2*np.abs(dy)
        if D > 0:
            y += np.sign(dy)
            D -= 2*dx
        plot(x, y)
```

`bresenham` takes as one of its arguments an arbitrary function `fn` which is called for each coordinate. This allows you to use it for a variety of purposes; during the development of this model it was also used to draw coordinates as well as sample them. Here is an image of a line drawn with Bresenham's algorithm in action:

![Bresenham example](images/line.png)

Finally, the points on the line are trimmed using `trim_zeros_indices`, so that the part of the line outside of the ROI (which will have zero intensity) is removed from the array. This is an adaptation of NumPy's `trim_zeros` function which returns the indices at which it would trim instead of performing the trim, so that we can trim both `y` and `coords` in a way that they still correspond:

```python
def trim_zeros_indices(has_zeros):
    first = 0
    for i in has_zeros:
        if i == 0:
            first += 1
        else:
            break

    last = len(has_zeros)
    for i in has_zeros[::-1]:
        if i == 0:
            last -= 1
        else:
            break

    return first, last
```

### <a name="step3_1_2"></a> Step 3.1.2: Finding the Threshold Point

After the best orientation has been determined and the intensity along that line has been sampled, the next step is to find a suitable threshold value. Following the original paper, this will consist of finding a local maximum in the first derivative of our sample. This is performed in the function `find_threshold_point`:

```python
def find_threshold_point(best, best_fn):
    fn, trimx, trim_coords = best_fn
    dom = np.linspace(np.min(trimx), np.max(trimx), 1000)
    f = fn(dom)
    mins = argrelmin(f)

    closest_min = -1
    closest_dist = -1
    for m in np.nditer(mins):
        dist = np.abs(500 - m)
        if closest_min == -1 or closest_dist > dist:
            closest_min = m
            closest_dist = dist

    fnprime = fn.derivative()
    restrict = dom[np.max(closest_min-THRESHOLD_AREA, 0):
                   closest_min+THRESHOLD_AREA]
    f2 = fnprime(restrict)

    m1 = restrict[np.argmax(f2)]
    mean = fn(m1)

    idx = np.min([int(np.floor(m1))+1, len(trim_coords)-1])
    return (mean, trim_coords, idx)
```

First, the function, its domain, and a mapping back to coordinates are unpacked from the tuple. Then the closest relative minimum of that function to the center of the domain is found (this would be the darkest point of the ventricular septum). Here is an example, taken from the line sampling shown above in the Bresenham function:

![Minimum example](images/graph0.png)

If you correlate the function here with the Bresenham image, you'll note that the point chosen is actually not between the ventricles, but is close. After that, the derivative in an area around this point (`restrict`, which consists of the 250 closest points on either side) is searched for a maximum; this is why it doesn't matter too much if the initial guess for the septum is wrong. It is hoped that the maximum of the derivative is the point at which the intensity is changing most quickly between myocardium and blood, and so provides a better threshold value than, say, the mean value between the highest and lowest intensities. Here's the derivative for the function pictured above:

![Derivative example](images/graphprime.png)

The corresponding point on the original graph demonstrates that a point on the border between the blood and the septum has been found:

![Corresponding point example](images/graph.png)

The threshold value obtained, the list of image coordinates, and an index into those coordinates are provided as return values so that `locate_lv_blood_pool` can iterate through coordinates, starting at that index, looking for a component.

### <a name=step3_2></a> Step 3.2: Propagating Segments

Instead of following this same methodology on all slices, we used the assumed property of spatial consistency and instead "propagate" our LV segmentation up and down through the other slices. This works as follows: given the LV segmentation of a previous slice, we compute its measure of similarity with each connected component on the new slice using the Jaccard index on sets of pixels. In this way, a new region is chosen, defined as the convex hull of the component with the maximum similarity to the existing component, and the process continues with the next slice. Here is the code for this process, `propagate_segments`:

```python
def propagate_segments(images, rois, base_mask, mean, closest_slice, time):
    def propagate_segment(i, mask):
        thresh, img_bin = cv2.threshold(images[i,
                                               time].astype(np.float32),
                                        mean, 255.0, cv2.THRESH_BINARY)

        labeled, features = label(img_bin)

        region1 = mask == 255
        max_similar = -1
        max_region = 0
        for j in range(1, features+1):
            region2 = labeled == j
            intersect = np.count_nonzero(np.logical_and(region1, region2))
            union = np.count_nonzero(np.logical_or(region1, region2))
            similar = float(intersect) / union
            if max_similar == -1 or max_similar < similar:
                max_similar = similar
                max_region = j

        if max_similar == 0:
            component = np.transpose(np.nonzero(mask))
        else:
            component = np.transpose(np.nonzero(labeled == max_region))
        hull = cv2.convexHull(component)
        hull = np.squeeze(hull)
        if hull.shape == (2L,):
            hull = np.array([hull])
        hull = np.fliplr(hull)

        newmask = np.zeros_like(img_bin)

        cv2.drawContours(newmask, [hull], 0, 255, thickness=-1)

        return newmask

    (rois_depth, _, _) = rois.shape
    newmask = base_mask
    masks = {}
    areas = {}
    masks[closest_slice] = base_mask
    areas[closest_slice] = np.count_nonzero(base_mask)
    for i in range(closest_slice-1, -1, -1):
        newmask = propagate_segment(i, newmask)
        masks[i] = newmask
        areas[i] = np.count_nonzero(newmask)

    newmask = base_mask
    for i in range(closest_slice+1, rois_depth):
        newmask = propagate_segment(i, newmask)
        masks[i] = newmask
        areas[i] = np.count_nonzero(newmask)

    return masks, areas
```

`masks` and `areas` hold binary masks and the areas of those masks, respectively. This function first iterates downward through the slices starting from closest_slice, propagating the segments as it goes. The inner function, `propagate_segment`, which is responsible for executing one iteration of the process, contains code that handles the case where no similar region is found (i.e., measures of similarity are all zero). In this case, it just reuses the same mask as the last slice.

For example, take this image:

![Projection example](images/projection_example.png)

In this case, the red area represents the mask of the previous slice, which would be the argument `mask` to the inner function `propagate_segment`. The blue area represents the current slice after it's been thresholded with the mean value given. The magenta area represents pixels shared by the mask from the previous slice and the thresholded pixels of the current slice. For each connected component in the blue layer, its intersection with the red (i.e., the magenta pixels only) is divided by the union with the red (i.e., all red and magenta pixels, and the blue pixels belonging to the same connected component) to yield the Jaccard index. This is done for every component (the vast majority will not intersect the red layer at all, yielding a Jaccard index of 0). The one with the highest score is selected as the most similar component, and the process repeats with the next slice.

Segments are propagated for the image stack at every time. At the end of `calc_all_areas`, the result is a segmentation for every original MRI image, and the areas (in pixels) of those segmentations. 

### <a name=step4></a> Step 4: Calculate Total Volume

At the end of the last step, the segmentation task is complete; all that remains is to calculate the total volume via a series of quick calculations on the areas:

```
def calc_total_volume(areas, area_multiplier, dist):
    slices = np.array(sorted(areas.keys()))
    modified = [areas[i] * area_multiplier for i in slices]
    vol = 0
    for i in slices[:-1]:
        a, b = modified[i], modified[i+1]
        subvol = (dist/3.0) * (a + np.sqrt(a*b) + b)
        vol += subvol / 1000.0  # conversion to mL
    return vol
```

This function multiplies the areas for one stack by the area multiplier, which was calculated way back in step 1. It's the product of the millimeter distance between pixels of the image, so it's equivalent to the height and width of a single pixel in millimeters. In this way we convert our total areas in pixels into total areas in square millimeters. Next, we apply the formula for a frustum to estimate the volume (assuming the two regions are roughly similar and circular), using the distance between slices in step 1 as the height, then divide by a factor of 1000 to get from cubic millimeters to milliliters. 

After the volume is computed, `segment_dataset` calculates ESV as the minimum of these values and EDV as the maximum, then calculates ejection fraction `(EF = (EDV-ESV)*100/EDV)`. The output of the algorithm is saved in the `output/` folder, with the results for individual datasets saved in folders bearing the same name, and the output images sorted by time. This happens in the `save_masks_to_dir` method:

```python
def save_masks_to_dir(dataset, all_masks):
    os.mkdir("output/%s" % dataset.name)
    for t in range(len(dataset.time)):
        os.mkdir("output/%s/time%02d" % (dataset.name, t))
        for s in range(len(dataset.slices)):
            mask = all_masks[t][s]
            image.imsave("output/%s/time%02d/slice%02d_mask.png" %
                         (dataset.name, t, s), mask)
            eroded = binary_erosion(mask)
            hollow_mask = np.where(eroded, 0, mask)
            colorimg = cv2.cvtColor(dataset.images[s][t],
                                    cv2.COLOR_GRAY2RGB)
            colorimg = colorimg.astype(np.uint8)
            colorimg[hollow_mask != 0] = [255, 0, 255]
            image.imsave("output/%s/time%02d/slice%02d_color.png" %
                         (dataset.name, t, s), colorimg)
```

Here, you can see the resulting segmentation:

![Segmentation Result](images/result.png)

Finally, we need a little bit more code so that this runs as a standalone script as well as a module you can tinker with, in addition to seeding random for you in case you're taking a random sample:

```python
if __name__ == "__main__":
    random.seed()
    auto_segment_all_datasets()
```

When this script is run via `python segment.py train/`, for instance, it will automatically segment all the datasets in that subdirectory. Alternatively, if python is run in interactive mode (for example, `python -i` then `>>> from segment import *`), you can play around with the functions individually.

<a name=analysis></a> Analysis
------------------------------

On average, our model's mean absolute error (MAE) for ejection fraction is 0.1599; however, the MAE for a model which just guesses the median each time is 0.077. Our model performs better than choosing an EF at random from a uniform distribution with an expected MAE of 0.269. While our model is outperforming random guessing, the fact that we are not currently beating the simple median calculation means that there is still more work to be done for increasing the model's accuracy. Our model, informed by movement of the cardiovascular system, should offer opportunities for further refinement and improvement that are not present in a simple statistical model. 

To learn more about where one might improve the model, let's look at the model's errors and outliers. There are a number of outliers that are clear failures of the model given the physical constraints. On dataset 1, the actual EDV is 261 mL, but the segmentation yields an EDV of 3861 mL. If the latter prediction were accurate, the volume of these single chambers of the heart would be about a gallon! Examining a few images from the dataset gives a good idea why:

![Outlier example](images/uhoh.png)

It seems that the segmentation picked up on a few structures adjacent to, but unrelated to, the left ventricle. This is due to a incorrect grayscale value chosen to threshold the images, leading to the component that would represent the LV becoming mistakenly connected to other, unrelated components; this problem is exacerbated by the fact that the algorithm takes the convex hull of its selected component to deal with papillary muscles, leading to one enormous "blob" of area that drives up the volume. The grayscale threshold value might be incorrect due to a number of factors. One likely cause is that the process that determines the orientation and derives, from the orientation, a threshold value is very sensitive to if a line going through the ventricles does not sufficiently resemble its matched filter.

However, in spite of all this, the EF predictions of the model are more consistent. This is because the same threshold value is used for the whole study, because it's derived from the DC images which are averaged over time; therefore, the EDV will most likely suffer from the same systematic thresholding problems as the ESV, leading to a similar ejection fraction if the two values err with a similar ratio. 

To illustrate the phenomenon, I plotted the EF predictions against the actual EF values:

![Comparison](images/comparison.png)

As you can see in this image, many of the data points are still roughly clustered around the line depicting where the actual EF values correspond to the predicted EF values, although the discrepancy grows as the EF increases. Additionally, the predicted EF values tend to run significantly lower than the real ones. The reason for this is that the two volumes, the EDV and ESV, are similar. I hypothesize that this is because the volume of the structures picked up due to the irregular choice of threshold is large compared to the ventricles, and do not change as much over the cardiac cycle.

<a name=conclusions></a> Conclusions
------------------------------------

This tutorial is intended to be an introduction to this Data Science Bowl challenge, not a competitive solution. Although this system performs decently well on a few datasets, there are certainly limitations (see [Limitations](#limitations) and discussion in [Analysis](#analysis), above). Below, I've compiled a list of potential improvements one might use as a starting point when adapting this model for the competition:

1. The parameters most likely to affect the performance of the model, considering it the problem outlined above, are those that will affect the region of interest, namely `STD_MULTIPLIER` and to a lesser degree `NUM_BINS`. With a differently sized region of interest, it's likely that the algorithm will decide on a different orientation as the lines get more or less similar to the desired cosine function (see below).

2. The algorithm's matched frequency filter (a cosine function with hardcoded frequency, as described in [Finding the Best Angle](#step3_1_1)) does not typically determine what a human observer would call the "best" orientation; that is, a line which would pass through the centers of the left and right ventricles, creating two large evenly-sized "humps" in the graph of pixel intensities across the line. I had first attempted a few other models such as bimodal Gaussian functions, quartic functions, and a method which abandoned matched filters altogether and looked for the two local maxima with the lowest minimum value between them. I expect that a better method exists for ascertaining orientation than a matched-frequency filter; at the very least, I imagine that it's possible to come up with a better function to model the type of curve we're looking for. Alternatively, it might be a better idea to discard the orientation altogether and determine another means of thresholding the image and finding the relevant component.

3. Occasionally, due to a less-than-ideal threshold or just noisy data, a few stray pixels will connect an undesired structure to the component corresponding to the LV. Because the LV component is convex-hulled in order to determine the final segmentation, even small structures can cause the volume to be vastly over-reported for certain slices, and the effect of being inadvertently connected to a large structure can be disastrous. I considered using [mathematical morphology](https://en.wikipedia.org/wiki/Mathematical_morphology); specifically, applying one iteration of binary erosion to the thresholded image in order to separate out barely-connected components (i.e., SciPy's binary_erosion function which is also used to help draw the segmentation outline on the output mages), then applying binary dilation to restore the original size of the component, in a process called "opening". However, I determined that many of the cases required as many as 3 or 4 iterations of both processes to fully delete the link between components. It was negatively impacting the original shape and interfering with propagation as well as lengthening the runtime of the algorithm, so I decided to abandon the process altogether. This problem is something of which anyone who adopts the system should be aware; my best idea for fixing it is playing with the structuring element used to determine connected components (see the documentation for SciPy's [label function](http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.measurements.label.html#scipy.ndimage.measurements.label)).

I hope you have enjoyed my tutorial and good luck in the competition. 
