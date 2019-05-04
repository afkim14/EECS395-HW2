# EECS 395 Assignment #2
### Probe Data Analysis for Road Slope
### Andres Kim, YaNing Wang, Stacy Montgomery

## Directory Structure
```
|   map_matching.py
├── test (test directory to test functionality of map_matching.py)
│   └── Partition6467LinkData.csv
│   └── Partition6467ProbePoints.csv
└── Partition6467MatchedPoints.csv (distribution code output)
└── Partition6467MatchedPoints_Slopes.csv (distribution code output)
```

## Setup
This script is run with Python 3. Please make sure you have dependencies installed.
```
pip3 install numpy
pip3 install pandas
pip3 install csv
```

## Running the Script
Make sure you have a directory compromised of images for smear detection. Then run the following:
```
python3 map_matching.py [probe_data.csv] [link_data.csv]
```
For example, we have also included a test directory for usage testing.
```
python3 map_matching.py test/Partition6467ProbePoints.csv test/Partition6467LinkData.csv
```

The script will output two csv files: `Partition6467MatchedPoints.csv` and `Partition6467MatchedPoints_Slopes.csv`. `Partition6467MatchedPoints.csv` will contain map-matched probe data, with each probe data having an associated link as well as the requested distances to the link. `Partition6467MatchedPoints_Slopes.csv` will contain the slopes for each of the map-matched probe data points.
