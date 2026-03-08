
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
python -m venv metgenx
source metgenx/bin/activate   # On Windows: metgenx\Scripts\activate
pip install -r requirements.txt
```
### 3. Download Model Weights and Databases
Download the following weights and databases from [zenodo](https://doi.org/10.5281/zenodo.18907370)
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

## Train and evaluate your custom model (optional)

Here we offer a demo workflow for training and evaluating the model using a publicly accessible dataset, MassSpecGym. 

Training MetGenX involves three main steps:
- Preparing your training data
- Constructing the Training Dataset
- Training the Model

### 1. Preparing your training data
Prepare your training data in a directory structure as the following:
<pre> {datasetname}/
- MS2_spectra.mgf
- metaData.csv
- splits.tsv (optional)
</pre>

**Note:**
- The ID in MS2 data should be the same as the ID in metaData.csv
- If no splits file is provided, the split fold should be included in the metaData.

### 2. Constructing the Training Dataset
Modeify the parameters in `./Scripts/01_Construct_dataset.py` as follows:

Run the following command:
```bash
  python ./Scripts/01_Construct_dataset.py
```

If the training dataset is constructed successfully, the following files will be generated in the `./results/{datasetname}` directory:
- input_dataset.dataset
- Query_index
- temp
- weights

> **Note:** During training of the spectra embedding model, we observed some randomness that may lead to slight differences in the training results. To reproduce our exact experimental results, you can use the pre-trained `SpecEmbed_model` weights provided by us. Place the weights in:
>
> ```
> # Our pre-trained weights in [./results/MassSpecGym/weights/word2vec/SpecEmbed_model](https://doi.org/10.5281/zenodo.18907370)
> ./results/{datasetname}/weights/word2vec/SpecEmbed_model
> ```

### 3. Training the Model

Run the following command:
```bash
  . /Scripts/02_model_training.sh
```

Parameters:

| Argument                   | Type     | Default         | Description                                            |
|-----------------------------|----------|-----------------|--------------------------------------------------------|
| `--datasetname`             | `str`    | `"MassSpecGym"` | Name of the dataset used for training (consistent with the directory name)        |
| `--path_train`              | `str`    | `None`          | Path to the training dataset                           |
| `--checkpoint_path`         | `str`    | `None`          | Path to pre-trained checkpoint to initialize the model |
| `--batch_size`              | `int`    | `64`            | Training batch size                                    |
| `--num_workers`             | `int`    | `4`             | Number of DataLoader worker processes                  |
| `--lr`                      | `float`  | `5e-6`          | Learning rate                                          |
| `--accelerator`             | `str`    | `"gpu"`         | Training device: `'gpu'` or `'cpu'`                    |

**Note:**
- The pretrained weights and fine-tuned weights for MassSpecGym dataset can be downloaded from [here](https://doi.org/10.5281/zenodo.18907370).(./MassSpecGym/MassSpecGym_weights/)
  - Pretraining: the model was pretrained on approximately 3M molecules by structure similarity.
  - MetGenX: the similarity threshold of 0.4 was used in the template search.
  - MetGenX_full: the similarity threshold was removed in the template search.
- If you want to change the hyperparameters, please modify the parameters in following files:
  - Parameters for model: `./weight/generation/config.json`
  - Parameters for database used: `./weight/generation/config_database.json`
  - Parameters for generation process: `./weight/generation/config_generation.json`
- The trained model weights will be saved in `./results/{datasetname}/weights/generation/Trained_Weight.pth`

## Model evaluation
The evaluation process of MetGenX has been modified to adapt the standard format of [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym).
If you want to evaluate the model, please install massspecgym package. (this is not necessary for using MetGenX)

**Note:**

The MassSpecGym package was constructed using **Python 3.11**. To avoid conflicts, if you want to use it in your current environment, you can run the following commands: 
  ```bash
    pip install massspecgym --no-deps
    pip install -r requirements_msg.txt
  ```
Alternatively, you can follow the instructions provided by MassSpecGym to create a separate environment for evaluation:
  ```bash
    conda create -n massspecgym python==3.11
    conda activate massspecgym
    pip install massspecgym
  ```
Note that installing MassSpecGym is not necessary for training or using MetGenX.

To evaluate the model, please run the following command:
```bash
  . Scripts/MassSpecGym/Evaluation_denovo.sh 
```
or
```bash
  . Scripts/MassSpecGym/Evaluation_retrieval.sh
```

---
The evaluation supports two modes:
#### 1. De novo molecule generation (database-free)
To evaluate the model on database-free mode, the constructed dataset can be directly used.

#### 2. Retrieval molecule generation (database-restricted)
To evaluate the model on database-restricted mode, the candidate list should be provided. The demo candidate list is available in [here](https://doi.org/10.5281/zenodo.18907370).
See ./MassSpecGym/MassSpecGym_retrieval_candidates_formula_canoical.json
The candidate list should be in the following format:
> {"True SMILES": ["Candidate_SMILES1", "Candidate_SMILES2", "Candidate_SMILES3", ...]}

Convert the dataset into the retrieval dataset:
```bash
  python ./Scripts/MassSpecGym/Convert.py 
```
The converted dataset will be saved in `./results/{datasetname}/input_dataset_retrival.dataset`

---

Parameters:

| Argument                | Type     | Default                                             | Description                                                                                              |
|-------------------------|----------|-----------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `--datasetname`         | `str`    | `None`                                              | Name of the dataset                                                                                      |
| `--Evaluation_mode`     | `str`    | `"denovo"`                                          | Evaluation mode: `'denovo'` for database-free evaluation, `'retrieval'` for database-restricted evaluation |
| `--dataset`             | `str`    | **(Required)**                                     | Path to dataset for evaluation (.dataset).                                                               |
| `--checkpoint`          | `str`    | **(Required)**                                     | Path to model checkpoint file                                                                            |
| `--output_dir`          | `str`    | **(Required)**                                     | Directory to save test results                                                                           |

The evaluation results will be saved in `./results/{datasetname}/eval_result/`

### To do
- [ ] Create a one-step model training and evaluation pipeline


---
## Troubleshooting

- **"No formula provided in spectra data"**  
  Ensure each spectrum in the `.mgf` file includes a `formula` field in its metadata.

---
## Maintainers
[@ Hongmiao Wang](https://github.com/waterom)