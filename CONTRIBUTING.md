# Contributing

Any contributions to 123D are welcome! This guide both serves as internal tutorial and can help you get started with the development process.

## Table of contents

1. [Ways to contribute](#ways-to-contribute)
2. [Development](#development)
2. [Submitting a pull-request](#submitting-a-pull-request)


## Ways to contribute

If you want to get involved and improve 123D, there are several ways to contribute, that include but are not limited to:

- **Support new datasets:** test

- **Improve documentation:**

- **Adding or extend features:** speed, coverage, tooling, adding new modalities.

- **Support new datasets:**



## Development

### Getting started

You can get started by
```sh
git clone git@github.com:kesai-labs/py123d.git
cd py123d
```

```sh
conda create -n py123d_dev python=3.12; conda activate py123d_dev  # Optional
pip install uv
uv pip install -e .[dev,docs]
pre-commit install
```

The above installation should also include linting, formatting, type-checking in the pre-commit.
We use [`ruff`](https://docs.astral.sh/ruff/) as linter/formatter, for which you can run:
```sh
ruff check --fix .
ruff format .
```
Type checking is not strictly enforced, but ideally added with [`pyright`](https://github.com/microsoft/pyright).

### Conventions

TODO

## Submitting a pull-request

TODO
