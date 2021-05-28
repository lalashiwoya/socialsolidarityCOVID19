# Data Analysis

### Annotation agreement as Cohens Kappa and F1
python annotationF1.py
- prints out the agreement for Tobias and Daniel as well as the one over the whole Dataset

### Create Figures that show the label distribution
python create_figures.py
- creates figures over the whole dataset and per hashtag and stores them in the Analysis folder

### Look closer into specific timeframe
python create_timeframe_sample.py
- creates csvs with the label distribution and the 10 most common hashtags per day
- creates csvs with daily samples in interesting timeframes of dips of the SoliScore

### Create a file with all misclassified Tweets
python error_analysis.py
- creates file containing all wrongly classified Tweets

