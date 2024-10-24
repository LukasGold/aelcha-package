<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/aelcha.svg?branch=main)](https://cirrus-ci.com/github/<USER>/aelcha)
[![ReadTheDocs](https://readthedocs.org/projects/aelcha/badge/?version=latest)](https://aelcha.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/aelcha/main.svg)](https://coveralls.io/r/<USER>/aelcha)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/aelcha.svg)](https://anaconda.org/conda-forge/aelcha)
[![Monthly Downloads](https://pepy.tech/badge/aelcha/month)](https://pepy.tech/project/aelcha)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/aelcha)
-->

[![PyPI-Server](https://img.shields.io/pypi/v/aelcha.svg)](https://pypi.org/project/aelcha/)
[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# aElChA

> A Python package providing the code for Automated Electrochemical Analysis (aElChA)

A longer description of your project goes here...

## Installation
```cmd
pip install aelcha
```

## Usage

### Excel as User Interface
* Open the File_Selection.xlsx from the examples folder
* Enter the files you like to process in the first sheet "Selection"
* Enter the parameters for the analysis
    * General parameters in the second sheet "Configuration"
    * File specific parameters in the first sheet "Selection"
* Run the script
```cmd
python <path_to_repository>/examples/script.py
```
### Python as User Interface
```python
from aelcha.core import process_file
from aelcha.user_interface import SelectionRow, Configuration
from aelcha.common import MaccorPreprocessingOption


config = Configuration(
    input_dir_default='path/to/input',
    export_dir_default='path/to/output',
    input_source_type=MaccorPreprocessingOption.mims_client1
)
row = SelectionRow(
    index=0,
    file_name='file_name.txt',
    sample_name='sample_name',
)
process_file(row, config)
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
