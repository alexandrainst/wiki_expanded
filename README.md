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

## Extract data from a Wikipedia dump

We use [Dumpster Dive](https://github.com/dumpster-dive/dumpster-dive) to extract the data from a Wikipedia dump.

### 1️⃣ Install dependencies

Install [nodejs](https://nodejs.org/en/) (at least `v6`), [mongodb](https://docs.mongodb.com/manual/installation/) (at least `v3`)

```bash
# install this script
npm install -g dumpster-dive # (that gives you the global command `dumpster`)
# start mongo up
mongod --config /opt/homebrew/etc/mongod.conf
```

### 2️⃣ Prepare the Wikipedia dump

#### 2a. Set language code
```bash
# Choose your target language (e.g., "da" for Danish, "en" for English)
# See https://dumps.wikimedia.org/ for available languages
LANG="da"
```

#### 2b. Download Wikipedia dump
```bash
# Download the latest Wikipedia dump for your chosen language
# Note: This can be several GB and may take time depending on your connection
wget "https://dumps.wikimedia.org/${LANG}wiki/latest/${LANG}wiki-latest-pages-articles.xml.bz2"
```

#### 2c. Extract the dump
```bash
# Unzip the compressed XML file
bzip2 -d "./${LANG}wiki-latest-pages-articles.xml.bz2"
```

#### 2d. Load data into MongoDB
```bash
# Parse Wikipedia XML and load into MongoDB (ensure MongoDB is running)
dumpster "./${LANG}wiki-latest-pages-articles.xml" \
  --infoboxes=false --citations=false --categories=false --images=false --links=true --plaintext=true
```

> **Note:** This step requires MongoDB to be running. Start it with the command from step 1.

#### 2e. Export processed data
```bash
# Export the processed data from MongoDB to JSON
mongoexport --db="${LANG}wiki" --collection=pages --out="${LANG}wiki_pages.jsonl"
```

### 3️⃣ Process the extracted data
Build five JSON files that will be used to construct the expanded Wikipedia dataset.

```bash
python src/scripts/process.py
```

### 4️⃣ Build the expanded Wikipedia dataset
Build the expanded Wikipedia dataset by expanding links in the articles.

```bash
python src/scripts/build_dataset.py --include-strategy=prepend --max-link-expansions=10 --num-tokens-threshold=15000
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
