# README

## Concept



## Data sources

The analysis presented here uses the cpg0004-lincs dataset from Way et al. (2022) (https://doi.org/10.1016/j.cels.2022.10.001), in which the same 1,571 compounds were profiled across 6 doses in A549 cells with both Cell Painting and LINCS L1000. 

Before running the main script, run the download_data.py script. By default, the data is written to the ./data folder.

```
python download_data.py
```

## Dependencies

Install the uv package manager, then run

```
uv sync
```

to create a virtual environment and install the dependencies. If preferred, you can also use pip:

```
pip install -r requirements 
```

## Instructions

Simply run the the run_similarity_analysis.py script. Output will be written to the results/ folder

If using uv, run

```
uv run python run_similarity_analysis.py 
```

Otherwise:

```
python run_similarity_analysis.py 
```


