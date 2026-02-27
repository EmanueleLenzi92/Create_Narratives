# Algorithm for Narrative Creation

This repository contains the code for the first module of an automatic workflow designed to build a Knowledge Graph of 454 narratives on mountain value chains developed within the MOVING (https://www.moving-h2020.eu/) project.

## Purpose

This module transforms textual data (stored in a CSV file or in raw txt files) into structured narratives divided into events.  
The output is used as the foundation for the next stages of Knowledge Graph construction.

## Input

The algorithm accepts:

1. **CSV file**  
   The MOVING dataset containing textual information about mountain value chains.  
   A `mappingtable.csv` file defines how dataset columns are converted into narrative events.

2. **TXT file**  
   A plain-text narrative that is automatically divided into paragraphs using a local LLM (using Ollama (https://ollama.com/). Ollama is required).

## Output

The module generates structured CSV narrative files inside the `/stories` folder.

### If input is CSV

- One output file per row
- Events are built using predefined mappings
- Output format: title,description,image

### If input is TXT

- One output file per text
- Each paragraph becomes an event
- Output format: title,description

## How to Run
In the main.py file, write:

```python
run("MOVING_VCs_DATASET_FINAL_V2.csv")
```

or

```python
run("narrative.txt")
```
