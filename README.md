# CS591C1
This is a repository for the final project of CS591C2. Below are descriptions of the code we implemented.

### data_visualization.ipynb

- This file is responsible for creating the dense subgraphs of the Deezer data sets. 
- Graph files (.gml) are written to `./graphs/[country]_graph.gml`, and can be opened in a graph visualization program such as Gephi.
- csv files of the subsets are formatted like the original data set are written to `./dataset/deezer_clean_data/top_5000_edges_subset/`

To run:

0. Run Jupyter notebook (`jupyter notebook`) in the same directory as this file.
1. Adjust the country parameter in the 2nd code cell. Options: 'RO', 'HU', 'HR'. Then Run All code cells.
2. The subset's average degree, average degree in the original subset, and number of unique users is printed out in the "Create Graphs" section.
3. The degree distribution visualization for the subset is also displayed in "Create Graphs"


### main.py

- This file contains functions to process data, run baseline methods, evaluate the results, and outputs the figures.
- It assumes that all related directories are correct in names.


To run:

0. Run as it is.
1. Run the main.m file to obtain the TarDPR results.
2. Comment out line 519 - 526, uncomment line 530 and line 531 to process the results.

### main.m

- This file is used to run TarDPR experiments.
- It requires all neccesary files and data to exist correctly.
- Some filename inside must be changed for each run.

To run:

0. Simply run it.
