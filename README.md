<!--
<p align="center">
  <img src="https://github.com/biopragmatics/bioontologies/raw/main/docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
  Bioontologies
</h1>

<p align="center">
    <a href="https://github.com/biopragmatics/bioontologies/actions?query=workflow%3ATests">
        <img alt="Tests" src="https://github.com/biopragmatics/bioontologies/workflows/Tests/badge.svg" />
    </a>
    <a href="https://pypi.org/project/bioontologies">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/bioontologies" />
    </a>
    <a href="https://pypi.org/project/bioontologies">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/bioontologies" />
    </a>
    <a href="https://github.com/biopragmatics/bioontologies/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/bioontologies" />
    </a>
    <a href='https://bioontologies.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/bioontologies/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://codecov.io/gh/biopragmatics/bioontologies/branch/main">
        <img src="https://codecov.io/gh/biopragmatics/bioontologies/branch/main/graph/badge.svg" alt="Codecov status" />
    </a>  
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /> 
    </a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
    </a>
    <a href="https://github.com/biopragmatics/bioontologies/blob/main/.github/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"/>
    </a>
</p>

Tools for biomedical ontologies.

## 💪 Getting Started

This package lets you get OBO Graphs from ontologies based on their OWL
files, OBO files, or [Bioregistry](https://bioregistry.io) prefixes. Internally,
it uses [ROBOT](https://robot.obolibrary.org) to convert from these formats
to [OBO Graph JSON](https://github.com/geneontology/obographs).

```python
import bioontologies

# Get an ontology and convert to OBO Graph object via an OWL IRI
owl_iri = "http://purl.obolibrary.org/obo/go.owl"
parse_results = bioontologies.convert_to_obograph(owl_iri)

# Get an ontology and convert to OBO Graph object via an OBO IRI
obo_iri = "http://purl.obolibrary.org/obo/go.obo"
parse_results = bioontologies.convert_to_obograph(obo_iri)

# Get an ontology by its Bioregistry prefix
parse_results = bioontologies.get_obograph_by_prefix("go")
go_graph_document = parse_results.graph_document
```

## 🚀 Installation

The most recent release can be installed from
[PyPI](https://pypi.org/project/bioontologies/) with:

```bash
$ pip install bioontologies
```


The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/biopragmatics/bioontologies.git
```

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.md](https://github.com/biopragmatics/bioontologies/blob/master/.github/CONTRIBUTING.md) for more information on getting involved.

## 👋 Attribution

### ⚖️ License

The code in this package is licensed under the MIT License.

<!--
### 📖 Citation

Citation goes here!
-->

### 🎁 Support

The Bioregistry was developed by the [INDRA Lab](https://indralab.github.io), a part of the
[Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/about/)
and the [Harvard Program in Therapeutic Science (HiTS)](https://hits.harvard.edu)
at [Harvard Medical School](https://hms.harvard.edu/).

### 💰 Funding

The development of this package is funded by the DARPA Young Faculty Award W911NF2010255 (PI: Benjamin M. Gyori).

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.

## 🛠️ For Developers

<details>
  <summary>See developer instructions</summary>


The final section of the README is for if you want to get involved by making a code contribution.

### Development Installation

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/biopragmatics/bioontologies.git
$ cd bioontologies
$ pip install -e .
```

### 🥼 Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/biopragmatics/bioontologies/actions?query=workflow%3ATests).

### 📖 Building the Documentation

The documentation can be built locally using the following:

```shell
$ git clone git+https://github.com/biopragmatics/bioontologies.git
$ cd bioontologies
$ tox -e docs
$ open docs/build/html/index.html
``` 

The documentation automatically installs the package as well as the `docs`
extra specified in the [`setup.cfg`](setup.cfg). `sphinx` plugins
like `texext` can be added there. Additionally, they need to be added to the
`extensions` list in [`docs/source/conf.py`](docs/source/conf.py).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses [Bump2Version](https://github.com/c4urself/bump2version) to switch the version number in the `setup.cfg`,
   `src/bioontologies/version.py`, and [`docs/source/conf.py`](docs/source/conf.py) to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel using [`build`](https://github.com/pypa/build)
3. Uploads to PyPI using [`twine`](https://github.com/pypa/twine). Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion minor` after.
</details>
