# Phenomobile - Plant Phenotyping Data Processing Pipeline

A comprehensive command-line tool for processing hyperspectral, thermal, and RGB images to extract plant phenotyping features and train machine learning models.

## 🚀 Features

- **Dataset Creation**: Process hyperspectral, thermal, and RGB images from local or server sources
- **Dataset Merging**: Merge parameter datasets with reference datasets using configurable strategies
- **ML Training**: Train and evaluate machine learning models for phenotyping tasks
- **Cross-platform**: Works on Windows, Linux, and macOS

## 📦 Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd phenomobile
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/macOS
   .venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r src/requirements.txt
   ```

### Dependencies

See `src/requirements.txt` for complete list of packages:
- `datasets==4.6.1` - Dataset manipulation
- `matplotlib==3.10.8` - Plotting and visualization
- `numpy==2.4.2` - Numerical operations
- `pandas==3.0.1` - Data analysis
- `Pillow==12.1.1` - Image processing
- `python-dotenv==1.2.1` - Environment variable management
- `scikit-learn==1.8.0` - Machine learning
- `scipy==1.17.1` - Scientific computing
- `seaborn==0.13.2` - Statistical visualization
- `skimage==0.0` - Image processing
- `spectral==0.24` - Hyperspectral data processing
- `statsmodels==0.14.6` - Statistical models
- `webcolors==25.10.0` - Color utilities
- `xgboost==3.2.0` - Gradient boosting

## 🖥️ Usage

### Basic Commands

```bash
# Create datasets
python main.py create --hs --th --rgb

# Create datasets with NDI calculation
python main.py create --hs --th --ndi_tuple "583.85,507.56"

# Merge datasets
python main.py merge --params_dataset data.csv --ref_dataset ref.csv

# Train ML models
python main.py train --dataset complete.csv --features "ndvi,ndi,anthocyanin" --target anthocyanin
```

## 📁 Data Input Structure

### Local Data Organization

The creation of the dataset will be applied on the raw data whether it is
saved in the local folder or in the cloud.
So for example when you type in the cli command like:

```bash
python main.py create --rgb
```
the dataset will be created from the raw RGB images stored in the local folder or in the server.
you might need to specify the path to the raw data in the main config file
(note: currently there is no implementation for creating the dataset of hyperspectral and thermal images form local folder, but it can be done from the server)

The structure of the raw data is vary depend on the source of the data:



### Local Data Structure

#### RGB Images

The script will look for the raw data in the following structure:

```
phenomobile/
├── images/                    # Raw image data
│   ├── local_rgb_folder/      # RGB images
│   │   ├── date_folder/       # Date-based folders
│   │   │   ├── images/      # Image files
│   │   │   └── annotations.csv  # Bounding box annotations
│   ├── hs_images/            # Hyperspectral images
│   └── th_images/            # Thermal images
├── DATA/                      # Processed datasets
└── config/                     # Configuration files
    ├── main_config.json      # Global settings
    └── anthocyanin_config.json # Project-specific settings
```

### Server Connection

#### Connection Configuration

first to enable to connect to the server, you need to configure the `.env` file with:

```env
SMB_USERNAME=your_username
SMB_PASSWORD=your_password
SMB_SERVER=server_address
SMB_SHARE=share_name
REMOTE_FOLDER=path/to/data
year= YYYY (for example: 2025)
date= MM-DD (for example: 03-12)
```

#### Server Folder Organization

```
<PROJECT_NAME>/
├── <YEAR>/                    # Year-based folders (e.g., 2025/)
│   ├── <DATE>/               # Date-based folders (e.g., 03-12/)
│   │   ├── RGB/             # RGB images
│   │   │   ├── <NUMBER_X1>/  # Experiment folders (e.g., 1/, 2/, 3/)
│   │   │   ├── <NUMBER_X2>/
│   │   │   └── <NUMBER_Xn>/
│   │   ├── HS/              # Hyperspectral images
│   │   │   ├── <NUMBER_X1>/
│   │   │   ├── <NUMBER_X2>/
│   │   │   └── <NUMBER_Xn>/
│   │   └── TH/              # Thermal images
│   │       ├── <NUMBER_X1>/
│   │       ├── <NUMBER_X2>/
│   │       └── <NUMBER_Xn>/
│   └── <ANNOTATION_FILE>.csv  # Bounding box annotations (optional)
```

**Folder Types**:
- **RGB/**: RGB images for processing
- **HS/**: Hyperspectral images (.hdr/.img format)
- **TH/**: Thermal images (.tiff/.csv format)
- **<NUMBER_X*>**: Individual experiment folders
- **<ANNOTATION_FILE>.csv**: Bounding box annotations for object detection

## ⚙️ Configuration

### Main Configuration (`config/main_config.json`)

```json
{
  "paths": {
    "download_folder": "images",
    "home_directory_name": "phenomobile",
    "FlirImageExtractor_path": null
  },
  "data_source": {
    "type": "local",
    "description": "Default data source is server (credentials in .env)"
  }
}
```

### Project Configuration (`config/anthocyanin_config.json`)

```json
{
  "metadata": {
    "project_name": "Anthocyanin_BENI_ATAROT"
  },
  "parameters": {
    "target": "Anthocyanin",
    "group_by_col": "catalog id",
    "kind_of_merged": "as_it"
  },
  "experiment_settings": {
    "ANNOTATION_FILE": "bounding_boxes_BENI_ATAROT.csv",
    "SPLIT_IMAGE_TO_OBJECTS": true,
    "dataset_folder": "DATA"
  },
  "categories": {
    "RED_white_blue_led_ids": ["R1", "R2", "R3", "R4", "R5"],
    "GREEN_white_blue_led_ids": ["G1", "G2", "G3", "G4", "G5"],
    "RED_white_led_ids": ["R6", "R7", "R8", "R9", "R10"],
    "GREEN_white_led_ids": ["G6", "G7", "G8", "G9", "G10"],
    "RED_Shade_ids": ["R11", "R12", "R13", "R14", "R15"],
    "GREEN_Shade_ids": ["G11", "G12", "G13", "G14", "G15"],
    "RED_Control_ids": ["C1", "C2", "C3", "C4", "C5"],
    "GREEN_Control_ids": ["C6", "C7", "C8", "C9", "C10"]
  }
}
```

## 🎯 CLI Commands

### Dataset Creation

```bash
# Create RGB dataset
python main.py create --rgb

# Create hyperspectral dataset
python main.py create --hs

# Create thermal dataset  
python main.py create --th

# Create multiple datasets
python main.py create --hs --th --rgb

# Create with NDI calculation
python main.py create --hs --ndi_tuple "583.85,507.56"

# Split images to objects
python main.py create --rgb --split_objects
```

**Options**:
- `--hs`: Create hyperspectral dataset
- `--th`: Create thermal dataset
- `--rgb`: Create RGB dataset
- `--ndi_tuple`: NDI wavelength tuple (e.g., "583.85,507.56")
- `--create_ndi_table`: Create NDI tables for hyperspectral images
- `--split_objects`: Split images to objects using annotations

### Dataset Merging

```bash
# Merge datasets
python main.py merge --params_dataset data.csv --ref_dataset ref.csv
```

**Options**:
- `--params_dataset`: Path to parameters dataset CSV file
- `--ref_dataset`: Path to reference dataset CSV file

### ML Training

```bash
# Train models
python main.py train --dataset complete.csv --features "ndvi,ndi,anthocyanin" --target anthocyanin
```

**Options**:
- `--dataset`: Path to training dataset
- `--features`: Comma-separated feature list
- `--target`: Target variable name
- `--task`: Task type ('regression' or 'classification')
- `--model`: Specific model to train

## 🔧 Configuration Parameters

### Data Source Settings

- **`data_source.type`**: `"local"` or `"server"`
  - `"local"`: Use local files in `images/` directory
  - `"server"`: Use SMB server connection (requires `.env` file)

### Project Parameters

- **`target`**: Target variable for ML training (e.g., "Anthocyanin")
- **`group_by_col`**: Column name for merging datasets (e.g., "catalog id")
- **`kind_of_merged`**: Merge strategy (`"as_it"` or `"sample_params"`)
- **`categories`**: LED ID mappings for different experimental conditions

### Experiment Settings

- **`ANNOTATION_FILE`**: Bounding box annotation filename
- **`SPLIT_IMAGE_TO_OBJECTS`**: Whether to split images into individual objects
- **`dataset_folder`**: Output directory for processed datasets

## 🏗️ Project Structure

```
phenomobile/
├── src/                          # Source code
│   ├── cli/                   # Command-line interface
│   │   ├── commands.py        # Command handlers
│   │   ├── workflows.py        # Workflow orchestration
│   │   └── anthocyanin_workflow.py  # Project-specific workflows
│   ├── core/                   # Core processing logic
│   │   ├── datasets_creation/  # Dataset creation classes
│   │   │   ├── dataset_creation.py
│   │   │   ├── rgb_ds_creation.py
│   │   │   ├── hyper_spectral_ds_creation.py
│   │   │   └── thermal_ds_creation.py
│   │   └── datasets_merge/    # Dataset merging classes
│   │       ├── merge_parameter_ds_with_ref_ds.py
│   │       └── merge_lettuce_dataset.py
│   ├── image_processing/         # Image processing utilities
│   │   ├── rgb_image.py
│   │   ├── hyper_spectral_image.py
│   │   └── image_processing.py
│   ├── ml/                     # Machine learning
│   │   ├── training.py
│   │   └── anthocyanin_training.py
│   ├── utils/                   # Utilities
│   │   ├── arguments.py        # CLI argument parsing
│   │   ├── config.py          # Configuration management
│   │   └── logger.py          # Logging utilities
│   └── main.py                 # Main entry point
├── config/                     # Configuration files
│   ├── main_config.json      # Global settings
│   └── anthocyanin_config.json # Project-specific
├── DATA/                       # Processed datasets (output)
├── images/                     # Raw data input
│   ├── local_rgb_folder/     # RGB images by date
│   ├── hs_images/            # Hyperspectral images
│   └── th_images/            # Thermal images
├── examples/                   # Example notebooks
├── requirements.txt              # Python dependencies
└── README.md                   # This file
```

## 🔍 Troubleshooting

### Common Issues

**Permission Denied Errors**:
- Ensure write permissions to `DATA/` folder
- Run as administrator if needed
- Check antivirus software blocking file access

**Module Not Found**:
- Install missing dependencies: `pip install -r src/requirements.txt`
- Activate virtual environment

**Environment File Not Found**:
- Create `.env` file in project root for server access
- Set `data_source.type` to `"local"` for local processing

**Path Issues**:
- Use forward slashes for paths in config files
- Ensure raw data folders exist before processing
- Check file extensions and naming conventions

### Installation Requirements

**FLIR Image Extractor**:
- Required for thermal image processing
- Install FLIR SDK and configure `FlirImageExtractor_path` in config

**SMB Connection**:
- Verify server credentials in `.env` file
- Test network connectivity to SMB server
- Check share permissions and folder structure

## 📝 Development

### Adding New Projects

1. Create project-specific config in `config/` folder
2. Add project workflow in `src/cli/`
3. Define LED categories in config
4. Update main.py to register new workflow

### Code Style

- Follow Python PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for new functions
- Include error handling and logging

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request with description

For questions and support, please open an issue in the repository.
