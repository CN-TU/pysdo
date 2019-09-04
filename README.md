# pysdo
pysdo is a Python implementation of the Sparse Data Observers (SDO) outlier detection algorithm.

Features
---------
* PCA-based observer count selection
* Histogram-based sampling
* Custom distance metrics
* Chunked multi-threaded operation to obtain near-optimum runtime performance
* Tree-based nearest observer search to obtain logarithmic runtime increase with the observer count


Installation
------------
pysdo can be installed using pip by running

```pip install git+https://github.com/CN-TU/pysdo```



Usage
--------
pysdo uses the same interface as outlier detectors implemented in [scikit-learn](https://scikit-learn.org/stable/modules/outlier_detection.html). The following example loads a dataset from `my_dataset.csv` and outputs the 10% most outlying samples' indices:
```
import pysdo
import pandas
X = pandas.read_csv('my_dataset.csv')
detector = pysdo.SDO(contamination=0.1)
labels = detector.fit_predict(X)
print ("Outliers:", [ i for i in range(labels.size) if labels[i]])
```
In this example, the observer count will be chosen automatically using Principal Component Analysis (PCA). However, it is highly recommended to choose at least the observer count manually relying on preknowledge about the dataset.

Here is an example which manually sets an observer count of 500, returns outlier scores rather than binary labels and utilizes all available CPU cores:
```
import pysdo
import pandas
X = pandas.read_csv('my_dataset.csv')
detector = pysdo.SDO(k=500, return_scores=True, n_jobs=-1)
scores = detector.fit_predict(X)
print ("Outlier scores:", scores)
```

References
-----------
F. Iglesias Vázquez, T. Zseby and A. Zimek, "Outlier Detection Based on Low Density Models," 2018 IEEE International Conference on Data Mining Workshops (ICDMW), Singapore, Singapore, 2018, pp. 970-979.  
DOI: [10.1109/ICDMW.2018.00140](https://doi.org/10.1109/icdmw.2018.00140)
