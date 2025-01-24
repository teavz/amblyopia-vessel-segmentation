# Towards image-based detection of Amblyopia using deep learning

## Authors  

Rijul S. Soans, PhD  
Susana T. L. Chung, OD, PhD  

Sight Enhancement Laboratory ([SELAB])  
Herbert Wertheim School of Optometry & Vision Science  
University of California, Berkeley, USA.

## Features

- Companion code for our research showing altered retinal vasculature can be used towards image-based detection of amblyopia
- Precise retinal vessel segmentation using Spatial Attention-UNet ([SA-UNet paper])

## Software Dependencies

The framework requires the following software:

- [Python 3.9.2] - General purpose programming language for Vessel Segmentation
- [MATLAB R2022b] - Programming & Numeric Computing platform for Analyses

## Installation  

1. Install Python from the link provided above.
2. Install MATLAB from the link provided above.
3. Clone this repository.
4. Install dependencies for Python by: ``pip install -r requirements.txt`` OR for conda users:
   ```
   conda env create -f environment.yml
   conda activate amblyopia_iroct_env
   ```
5. Run the segmentation on an example image:
    ```
    python run_segmentation.py --image sample_images/AMB_001_OD.png --model models/SA_UNet.h5
    ```
   
## License
MIT License

Copyright 2025  &copy; Rijul S. Soans, &copy; Susana T. L. Chung  

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.   


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job.)

   [MATLAB R2022b]: <https://www.mathworks.com/products/new_products/release2022b.html>
   [SELAB]: <https://selab.berkeley.edu/>
   [SA-UNet paper]: <https://ieeexplore.ieee.org/document/9413346>
   [Python 3.9.2]: <https://www.anaconda.com/download>

