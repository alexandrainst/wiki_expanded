<a href="https://github.com/alexandrainst/wiki_expanded">
<img
    src="https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/alexandra/alexandra-logo.jpeg"
	width="239"
	height="175"
	align="right"
/>
</a>

# Expanded Wikipedia Dataset

Long context Wikipedia dataset constructed by expanding links in Wikipedia articles.

## Installation

```bash
make install
```

### WikiExtractor

We use the [WikiExtractor](https://github.com/attardi/wikiextractor) module to extract data from a Wikipedia dump.

```bash
cd .. && git clone https://github.com/attardi/wikiextractor && \
cd wiki_expanded
```

To make this module work, see [this issue](https://github.com/attardi/wikiextractor/issues/336#issuecomment-2700154486) (the solution in [this issue](https://github.com/attardi/wikiextractor/issues/336#issuecomment-2400360799) did not work for me).


## Usage

1. Extract data from a [Wikipedia dump](https://dumps.wikimedia.org/dawiki/latest/):

```bash
python -m ../wikiextractor.WikiExtractor dawiki-latest-pages-articles.xml.bz2 --links --output=data/raw/text
```

2. Process the extracted data to build four json files that will be used to construct the expanded Wikipedia dataset.

```bash
python src/scripts/process.py
```

3. Build the expanded Wikipedia dataset.

```bash
python src/scripts/build_dataset.py
```

______________________________________________________________________
[![Code Coverage](https://img.shields.io/badge/Coverage-0%25-red.svg)](https://github.com/alexandrainst/wiki_expanded/tree/main/tests)
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/wiki_expanded)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/alexandrainst/wiki_expanded/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/wiki_expanded)](https://github.com/alexandrainst/wiki_expanded/commits/main)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/alexandrainst/wiki_expanded/blob/main/CODE_OF_CONDUCT.md)

Developer:

- Oliver Kinch (oliver.kinch@alexandra.dk)


### Adding and Removing Packages

To install new PyPI packages, run:
```
uv add <package-name>
```

To remove them again, run:
```
uv remove <package-name>
```

To show all installed packages, run:
```
uv pip list
```
