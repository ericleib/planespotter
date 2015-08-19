# PlaneSpotter
This repository contains: 
 - A web-crawler (based on the [Scrapy](http://scrapy.org/) Python library) to download pictures of aircraft from various websites.
 - An [OpenIMAJ](http://www.openimaj.org/) project to train an image classifier with traditional Machine Learning technics (along with utility classes to process the output of the crawler).
 - A (more advanced) [Theano](http://deeplearning.net/software/theano/) script to train a Convolutional Neural Network (CNN) image.
 - A minimalist Python web server to host the CNN classifier

# Instructions

## Prerequisites

You will need to download and install [Python 2.7](https://www.python.org/downloads/), or preferably a scientific Python distribution, such as [Anaconda](http://continuum.io/downloads).

Also required is the [Java 1.8 Development Kit](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html).

## Image crawler

Install Scrapy:

    pip install scrapy

Run the crawler (cd into `planespotter/scrapy`):

    scrapy crawl airliners -o planes.json > log.txt

*Note: This will download potentially millions of (small) pictures on your hard-drive, taking a lot of time. Performing this on a SSD will greatly speed-up the process*

## OpenIMAJ

In Eclipse, import the `openimaj_classifier` folder as an "Existing Project into Workspace".

*Note: OpenIMAJ is based on Maven. The project has a great number of (probably unused) dependencies. So Maven will download a lot of libraries from the Internet to perform the first build (afterward it will be transparent).*

There are three main classes in the project that you can run:
 - [tk.thebrightstuff.JsonProcessor](https://github.com/ericleib/planespotter/blob/master/openimaj_classifier/src/main/java/tk/thebrightstuff/JsonProcessor.java): This utility class takes as input one (or more) json files created by the crawler, and reformats them into one single text file (required by both OpenIMAJ **and** Theano).
 - [tk.thebrightstuff.Sorter](https://github.com/ericleib/planespotter/blob/master/openimaj_classifier/src/main/java/tk/thebrightstuff/Sorter.java): This utility class processes an image folder created by the crawler (or a tar version of it) to create a more file-system-efficient folder structure (required by both OpenIMAJ **and** Theano).
 - [tk.thebrightstuff.AircraftApp](https://github.com/ericleib/planespotter/blob/master/openimaj_classifier/src/main/java/tk/thebrightstuff/AircraftApp.java): This class trains the **image annotator**. Various inputs are required, such as the path were the **image folder** is stored on the disk, and how many pictures should be used for the training. After training, all the data is saved in a `data.txt` file (which can be reloaded later), and the classifier is tested against a set of pictures.

## Theano

Install theano:

    pip install theano

To train the CNN on you GPU (much more efficient), you also need to have a good NVidia graphic card, and install [Cuda](https://developer.nvidia.com/cuda-downloads) and g++. On Windows you will probably need to install Visual Studio (See [this post](http://stackoverflow.com/questions/31892519/link-error-with-cuda-7-5-in-windows-10-from-theano-project-msvcrt-lib-error-l) for an example of setup).

Run the script (cd into `theano_conv_net`):

    python Theano_aircraft.py

*Note: The CNN will take a very long time to train, depending on your hardware, the size of the dataset and other settings that you can tune in the script.*

## Web server

The Theano script should save a `model-values.save` file inside `webapp/results`. You are ready to run the server!

Run the server (cd into `webapp`):

    python server.py

*Note: Depending on your setup you may need to run the server as an administrator.*

Visit the web application at [localhost](http://localhost/).

*Note: If you want to run the server on a separate computer, you will need to install Python and Theano as well on this computer.*
