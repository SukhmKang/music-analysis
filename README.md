# music-analysis

**Ishan Balakrishnan & Sukhm Kang**

Project that detects beats within an inputted song (.wav file) and finds its tempo (beats per minute). Annotations for the tesing datasets are available in a parseable format at [this][../blob/main/Annotations.zip] link.

## Built With

**Python** (https://www.python.org/) \
**Pandas** (https://pandas.pydata.org/) \
**Wave** (https://docs.python.org/3/library/wave.html) \
**SciPy** (https://scipy.org/) \
**MatPlotLib** (https://matplotlib.org/)

## Algorithm Summary

Please see our academic paper, "Global Tempo Estimation Algorithm for Popular Music with Noise-Filtration and Spectral Clustering Algorithm," for an in-depth summary of the implementation, methodology, tuning, and testing of the algorithm.

## Results

| Dataset | Type (Training/Testing) | Algorithm Accuracy |
| --- | --- | --- |
| SPOTIFY TOP 100 2016 | TRAINING | 87/100 |
| SPOTIFY TOP 100 2018 | TRAINING | 93/100 |
| BILLBOARD HOT 100 YEAR-END 2021 | TESTING | 91/100 |
| BILLBOARD HOT 100 APRIL 2022 | TESTING | 89/100 |
| SPOTIFY '00s ROCK ANTHEMS | TESTING | 85/100 |
| GIANT STEPS | TESTING | 562/662 |

## Authors & Contact Information

This project was authored by **Sukhm Kang** and **Ishan Balakrishnan**.

**Sukhm Kang**\
Mathematics @ The University of Chicago\
https://www.linkedin.com/in/sukhm-kang


**Ishan Balakrishnan**\
Computer Science & Business @ University of California, Berkeley\
https://www.linkedin.com/in/ishanbalakrishnan

Feel free to reach out to either one of us by email @ ishan.balakrishnan(at)berkeley.edu or sukhmkang(at)uchicago.edu! 
