
# Contributing to 123D

Thank you for your interest in contributing to 123D! This guide will help you get started with the development process.

## Getting Started

### 1. Clone the Repository

```sh
git clone git@github.com:DanielDauner/d123.git
cd d123
```

### 2. Install the pip-package

```sh
conda env create -f environment.yml --name d123_dev # Optional
conda activate d123_dev
pip install -e .[dev]
pre-commit install
```

.. note::
    We might remove the conda environment in the future, but leave the file in the repo during development.


### 3. Managing dependencies

One principal of 123D is to keep *minimal dependencies*. However, various datasets require problematic (or even outdated) dependencies in order to load or preprocess the dataset. In this case, you can add optional dependencies to the `pyproject.toml` install file. You can follow examples of Waymo or nuPlan. These optional dependencies can be install with

```sh
pip install -e .[dev,waymo,nuplan]
```
where you can combined the different optional dependencies.

The optional dependencies should only be required for data pre-processing. If a dataset allows to load sensor data dynamically from the original dataset, please encapsule the import accordingly, e.g.

```python
import numpy as np
import numpy.typing as npt

def load_camera_from_outdated_dataset(file_path: str) -> npt.NDArray[np.uint8]:
    try:
        from outdated_dataset import load_camera_image
    except ImportError:
        raise ImportError(
            "Optional dependency 'outdated_dataset' is required to load camera images from this dataset. "
            "Please install it using: pip install .[outdated_dataset]"
        )
    return load_camera_image(file_path)
```


## Code Style and Formatting

We maintain consistent code quality using the following tools:
- **[Black](https://black.readthedocs.io/)** - Code formatter
- **[isort](https://pycqa.github.io/isort/)** - Import statement formatter
- **[flake8](https://flake8.pycqa.github.io/)** - Style guide enforcement
- **[pytest](https://docs.pytest.org/)** - Testing framework for unit and integration tests
- **[pre-commit](https://pre-commit.com/)** - Framework for managing and running Git hooks to automate code quality checks


.. note::
    If you're using VSCode, it is recommended to install the [Black](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter), [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort), and [Flake8](https://marketplace.visualstudio.com/items?itemName=ms-python.flake8) plugins.



### Editor Setup

**VS Code Users:**
If you're using VSCode, it is recommended to install the following plguins:
- [Black](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) - see above.
- [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort) - see above.
- [Flake8](https://marketplace.visualstudio.com/items?itemName=ms-python.flake8) - see above.
- [autodocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) - Creating docstrings (please set `"autoDocstring.docstringFormat": "sphinx-notypes"`).
- [Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker) - A basic spell checker.


**Other Editors:**
Similar plugins are available for most popular editors including PyCharm, Vim, Emacs, and Sublime Text.


## Documentation Requirements

### Docstrings
- **Development:** Docstrings are encouraged but not strictly required during active development
- **Release:** All public functions, classes, and modules must have comprehensive docstrings before release
- **Format:** Use [Sphinx-style docstrings](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html)

**VS Code Users:** The [autoDocstring extension](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) can help generate properly formatted docstrings.

### Type Hints
- **Required:** All function parameters and return values must include type hints
- **Style:** Follow [PEP 484](https://peps.python.org/pep-0484/) conventions

### Sphinx documentation

All datasets should be included in the `/docs/datasets.md` documentation. Please follow the documentation format of other datasets.

You can install relevant dependencies for editing the public documentation via:
```sh
pip install -e .[docs]
```

It is recommended to uses [sphinx-autobuild](https://github.com/sphinx-doc/sphinx-autobuild) (installed above) to edit and view the documentation. You can run:
```sh
sphinx-autobuild docs docs/_build/html
```

## Adding new datasets
TODO


## Questions?

If you have any questions about contributing, please open an issue or reach out to the maintainers.
