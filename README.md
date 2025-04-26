# Portraying Large Language Models as Machines, Tools, or Companions Affects What Mental Capacities Humans Attribute to Them

![Teaser photo of experimental setup. Participants are assigned to one of four conditions where they watch videos portraying LLMs as machines, tools, or companions, or watch no video. Then all participants take the same mental capacity attribution survey] (teaser.png)
## Abstract

How do people determine whether non-human entities have thoughts and feelings — an inner mental life? Prior work has proposed that people use compact sets of dimensions (e.g.,body-heart-mind) to form beliefs about familiar kinds, but how do they generalize to novel entities? Here we investigate emerging beliefs about the mental capacities of large language models (LLMs) and how those beliefs are shaped by how LLMs are portrayed. Participants (N = 470) watched brief videos that encouraged them to view LLMs as either machines, tools, or companions. We found that the companion group more strongly endorsed statements regarding a broad array of mental capacities that LLMs might possess relative to the machine and tool groups, suggesting that people’s beliefs can be rapidly shaped by context. Our study highlights the need to explore the factors shaping people’s beliefs about emerging technologies to promote accurate public understanding.

This study was [pre-registered](https://aspredicted.org/vgdm-gjrm.pdf) with AsPredicted and approved by IRB at Princeton University (#17151).

## Directory Structure

portraying_llms/
├── code/
|   ├── src/ # Python and R files to conduct analysis
|   ├── visualization_notebooks/ # Jupyter notebooks for creating graphs for papers
|   └── analysis.ipynb # Main analysis notebook
├── data/
|   ├── raw/ # Raw data from Qualtrics, separated into baseline and experimental groups
|   └── files/ # Supplementary files to help with data processing
├── survey/Appendix.pdf # Information regarding survey questions, pilot study, and links to stimuli

## Reproducing Analyses

### Create Conda Environment
If you do not have conda, install it following instructions on the [website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
In the terminal,
1. Create a conda environment: `conda env create -f environment.yml`
2. Activate the environment with `conda activate portrayingLLMs`

### Running Analysis Code

The main analyses can be found in `code/analysis.ipynb`. Visualizations for graphs found in the paper can be produced using the `code/visualization_notebooks/figures.ipynb`. Trying to run `figure.ipynb` before `analysis.ipynb` will not work. Resulting analyses can be found in a new folder called `analysis`. It is recommended to sequentially run each cell in `analysis.ipynb` to ensure all variables are defined. Exploratory analyses are at the bottom of the file.
