# OTG (Optimal Transport Grouping)

[English](README.en.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [日本語](README.jp.md)

## Overview
This is a Python program for generating groupings as uniform as possible by using the optimal transport technique.

## Usage
- Without a Python runtime environment
    - On Windows, download "[otg_gradio_win.7z](https://github.com/tanaken-basis/otg/raw/master/otg_gradio_win.7z)" and decompress it, double-click "otg_gradio.exe" to start the web application for automatically generating groupings. It may take 2-3 minutes to start the web application. This web application runs on a local machine.
    - On Mac (Apple silicon), download "[otg_gradio_mac.7z](https://github.com/tanaken-basis/otg/raw/master/otg_gradio_mac.7z)" and decompress it, double-click "otg_gradio" to start the web application for automatically generating groupings. It may take 2-3 minutes to start the web application. This web application runs on a local machine.
- Using Python programs
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
