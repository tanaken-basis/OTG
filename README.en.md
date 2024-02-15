# OTG (Optimal Transport Grouping)

[English](README.en.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [日本語](README.jp.md)

## Overview
This is a Python program for generating groupings as uniform as possible by using the optimal transport technique.

## Usage
- By running "otg_gradio.py" in Python, you can start the web application created by Gradio for automatically generating groupings. You can adjust parameters for automated generation, execute calculations to generate results, and save calculated results to files. This web application can be run on a local machine.
- By running "otg_flet.py" in Python, you can start the GUI application created by Flet for automatically generating groupings. You can adjust parameters for automated generation, execute calculations to generate results, and save calculated results to files.
- The same program is also written in the notebook file "otg.ipynb". 

## Notes on using Python programs
- In this code, we use the Python library UMAP to visualize data. Please refer to https://github.com/lmcinnes/umap for installation information.
- In this code, we use the Python library Gradio to create the web app. Please refer to https://github.com/gradio-app/gradio for installation information.
- In this code, we use the Python library Flet to create the GUI app. Please refer to https://github.com/flet-dev/flet for installation information.

## Samples of execution screens

#### Gradio
![alt text](otg_gradio-1.jpg)

#### Flet
![alt text](otg_flet-1.jpg)
