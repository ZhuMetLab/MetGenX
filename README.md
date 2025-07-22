
![MetGenX Logo](logo.png)
# MetGenX CLI Manual

MetGenX is a structure-informed generative model for metabolite annotation based on MS2 spectra. This command-line tool processes a single MS2 spectra file (.msp/.mgf) and outputs generated structures.

---
## Requirements
- Python >= 3.10
- Dependencies:
  - `numpy==1.26.4`
  - `pandas==2.2.3`
  - `faiss-cpu==1.8.0`
  - `transformers==4.42.3`
  - `torch==2.4.1`
  - `rdkit==2023.9.5`
  - `gensim==4.3.2`
  - `lightgbm==4.5.0`
  - `pytorch_lightning==2.2.5`
  - `six==1.16.0`
  - `more_itertools==10.5.0`
  - `scipy==1.12.0`
  - `jpype1==1.5.0`

- The model requires Java SE Development Kit 11.0.23 (JDK 11.0.23)

---
## Installation

### 1. Clone the project

You can install MetGenX by cloning the repository:
download the code from [GitHub](https://github.com/ZhuMetLab/MetGenX)

```bash
$ git clone https://github.com/ZhuMetLab/MetGenX.git
cd MetGenX
```
### 2.  Install Dependencies
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
### 3. Download Model Weights and Databases
Download the following weights and databases from [zenodo](https://doi.org/10.5281/zenodo.15709467)
copy the weights and database dir into the project structure:
<pre>MetGenX/
- weights/
- database/</pre>



## Usage

### Basic Command

```bash
python run.py --spec_path <input_file.mgf> [options]

# Run demo data in positive mode
python run.py --spec_path ./test/demo_positive.mgf  --mode Restricted --output ./test/generation_results_restricted.csv
```

## Arguments

| Argument      | Type     | Default                   | Description                                 |
|---------------|----------|---------------------------|---------------------------------------------|
| `--spec_path` | `str`    | **(Required)**            | Path to input `.mgf` file of MS2 spectra    |
| `--polarity`  | `str`    | `"positive"`              | Spectrum polarity: `positive` or `negative` |
| `--mode`      | `str`    | `"Free"`                  | Generation mode: `Free` or `Restricted`     |
| `--output`    | `str`    | `"generation_results.csv"`| Output CSV file path                        |
| `--db_cutoff` | `float`  | `0.4`                     | Similarity cutoff for template filtering    |

## MS2 spectra format
### mgf
```bash
BEGIN IONS
Name= <Name>
IONMODE= <ionization mode>
PEPMASS= <precurosr m/z>
Formula= <neutral formula>
m/z1 intensity1
m/z2 intensity2
m/z3 intensity3
...
END IONS
```

### msp
```bash
Name: <Name>
IONMODE: <ionization mode>
PEPMASS: <precurosr m/z>
Formula: <neutral formula>
Num Peaks: <number of peaks>
m/z1 intensity1
m/z2 intensity2
m/z3 intensity3
...
```


---
## Troubleshooting

- **"No formula provided in spectra data"**  
  Ensure each spectrum in the `.mgf` file includes a `formula` field in its metadata.

---
## Maintainers
[@Hongmiao Wang](https://github.com/waterom)