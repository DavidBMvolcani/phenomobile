# Phenomobile Developer Manual

## Table of Contents

### General Documentation
1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Class Hierarchy](#class-hierarchy)
4. [CLI Command Flow Diagrams](#cli-command-flow-diagrams)
5. [Core Components](#core-components)
6. [Configuration Management](#configuration-management)
7. [Development Guidelines](#development-guidelines)
8. [Examples and Usage Patterns](#examples-and-usage-patterns)
9. [Testing Guidelines](#testing-guidelines)
10. [Performance Considerations](#performance-considerations)
11. [Debugging and Troubleshooting](#debugging-and-troubleshooting)

12. [Anthocyanin Project: Detection of Anthocyanin in Lettuces](#anthocyanin-project-detection-of-anthocyanin-in-lettuces)

## 1. Architecture Overview

Phenomobile follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │   main.py   │  │ commands.py │  │ arguments.py│       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  Workflow Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │  workflows  │  │  training   │  │ plotting    │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   Core Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │dataset_creat│  │dataset_merge│  │ datasets    │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                Processing Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │image_proc   │  │hyper_spec   │  │ rgb/thermal │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                 Utility Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │  config.py  │  │  logger.py  │  │  utils      │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## 2. Directory Structure

```
phenomobile/
├── src/
│   ├── cli/                     # Command-line interface
│   │   ├── commands.py         # Command handlers
│   │   ├── workflows.py        # Workflow orchestration
│   │   └── anthocyanin_workflow.py # Project-specific workflows
│   ├── core/                    # Core business logic
│   │   ├── datasets_creation/  # Dataset creation modules
│   │   │   ├── dataset_creation.py
│   │   │   ├── hyper_spectral_ds_creation.py
│   │   │   ├── thermal_ds_creation.py
│   │   │   ├── rgb_ds_creation.py
│   │   │   └── merge_parameters_ds.py
│   │   └── datasets_merge/      # Dataset merging modules
│   │       ├── merge_lettuce_dataset.py
│   │       └── merge_parameter_ds_with_ref_ds.py
│   ├── image_processing/        # Image processing classes
│   │   ├── image_processing.py  # Base image processing
│   │   ├── hyper_spectral_image.py
│   │   ├── rgb_image.py
│   │   └── thermal_image.py
│   ├── ml/                      # Machine learning
│   │   └── training.py          # Training classes
│   ├── plotting/                # Visualization
│   │   ├── base_plots.py
│   │   └── anthocyanin_plots.py
│   ├── utils/                   # Utilities
│   │   ├── config.py           # Configuration management
│   │   ├── logger.py           # Logging utilities
│   │   └── arguments.py        # Argument parsing
│   ├── datasets/                # Dataset classes
│   │   └── lettuce_dataset.py
│   └── main.py                  # Entry point
├── config/                      # Configuration files
│   ├── main_config.json        # Main configuration
│   └── anthocyanin_config.json # Project-specific config
├── datasets/                    # Data storage
├── examples/                    # Example notebooks
├── documentation/               # Documentation
└── README.md                    # User documentation
```

## 3. Class Hierarchy

### Core Base Classes

```python
# Abstract Base Classes
ABC
├── ml.training.Training (ABC)
│   └── ml.anthocyanin_training.AnthocyaninTraining
└── cli.workflows.DatasetMergeWorkflow (ABC)
    └── cli.anthocyanin_workflow.AnthocyaninDatasetMergeWorkflow
```

### Image Processing Hierarchy

```python
# Image Processing Classes
image_processing.ImageProcessing
├── image_processing.HyperSpectralImage
├── image_processing.RgbImage
└── image_processing.ThermalImage
```

### Dataset Creation Classes

```python
# Dataset Creation
core.datasets_creation.HyperSpectralDsCreation
core.datasets_creation.ThermalDsCreation
core.datasets_creation.RgbDsCreation
core.datasets_creation.MergeParameterDs
```

### Workflow Classes

```python
# Workflow Classes
cli.workflows.DatasetCreationWorkflow
cli.workflows.MLTrainingWorkflow
cli.workflows.PlottingWorkflow
└── cli.anthocyanin_workflow.*
    ├── AnthocyaninMLTrainingWorkflow
    ├── AnthocyaninDatasetMergeWorkflow
    └── AnthocyaninPlottingWorkflow
```

### Configuration Management

```python
# Configuration
utils.config.ConfigManager
```

## 4. CLI Command Flow Diagrams

### Create Command Flow

**ASCII Flow Diagram:**
```
User Command: python main.py create --hs --th
         ↓
      main.py
         ↓
   parse_and_validate
         ↓
   ConfigManager
         ↓
   execute_command
         ↓
 handle_create_command
         ↓
   get_workflow 'create'
         ↓
DatasetCreationWorkflow
         ↓
   create_datasets
         ↓
   ┌─────────┐
   │ HS?     ├─Yes→ HyperSpectralDsCreation → Process HS images → Save HS dataset
   └─────────┘
   ┌─────────┐
   │ TH?     ├─Yes→ ThermalDsCreation → Process TH images → Save TH dataset  
   └─────────┘
   ┌─────────┐
   │ RGB?    ├─Yes→ RgbDsCreation → Process RGB images → Save RGB dataset
   └─────────┘
         ↓
   Log completion
```


### Train Command Flow

**ASCII Flow Diagram:**
```
User Command: python main.py train --dataset data.csv
         ↓
      main.py
         ↓
   parse_and_validate
         ↓
   ConfigManager
         ↓
   execute_command
         ↓
 handle_train_command
         ↓
   get_workflow 'train'
         ↓
   Project type?
    ├─Anthocyanin→ AnthocyaninMLTrainingWorkflow
    └─Other      → MLTrainingWorkflow
         ↓
   train_models
         ↓
   Training class init
         ↓
   Load dataset
         ↓
   Task type?
    ├─Regression→ evaluate_regression_models
    └─Classification→ evaluate_classification_models
         ↓
   Generate results
         ↓
   Save CSV results
         ↓
   Log completion
```


### Merge Command Flow

**ASCII Flow Diagram:**
```
User Command: python main.py merge --params ds1 --ref ds2
         ↓
      main.py
         ↓
   parse_and_validate
         ↓
   ConfigManager
         ↓
   execute_command
         ↓
 handle_merge_command
         ↓
   Validate paths
         ↓
   get_workflow 'merge'
         ↓
   Project type?
    ├─Anthocyanin→ AnthocyaninDatasetMergeWorkflow
    └─Other      → NotImplementedError
         ↓
   merge_datasets
         ↓
   Load parameter dataset
         ↓
   Load reference dataset
         ↓
   Merge strategies
         ↓
   Save merged dataset
         ↓
   Log completion
```


## 5. Core Components

### 1. ConfigManager (`utils/config.py`)

**Purpose**: Centralized configuration management

**Key Methods**:
- `__init__(env_file, project_config)` - Initialize configuration
- `load_main_config()` - Load main configuration from JSON
- `load_project_config()` - Load project-specific configuration
- `load_environment()` - Load environment variables from .env file
- `get_dataset_path(dataset_name)` - Resolve dataset paths
- `ensure_output_dir()` - Create output directory if needed

**Usage Pattern**:
```python
config = ConfigManager(env_file='.env', project_config='anthocyanin_config.json')
dataset_path = config.get_dataset_path('my_dataset')
```

### 2. Training Classes (`ml/training.py`)

**Base Class**: `Training(ABC)`

**Key Methods**:
- `__init__(dataset_name, config, task, model)` - Initialize training
- `evaluate_regression_models(features, target)` - Train regression models
- `evaluate_classification_models(features, target)` - Train classification models
- `preprocess_data()` - Data preprocessing pipeline

**Child Classes**:
- `AnthocyaninTraining` - Anthocyanin-specific training logic

### 3. Image Processing Classes

**Base Class**: `ImageProcessing`

**Specialized Classes**:
- `HyperSpectralImage` - Process hyperspectral images, compute NDVI/NDI
- `RgbImage` - Process RGB images
- `ThermalImage` - Process thermal images

**Key Methods**:
- `compute_ndvi()` - Calculate Normalized Difference Vegetation Index
- `compute_ndi(wl1, wl2)` - Calculate Normalized Difference Index
- `split_image_to_objects()` - Segment images into objects

### 4. Dataset Creation Classes

**Classes**:
- `HyperSpectralDsCreation` - Create hyperspectral datasets
- `ThermalDsCreation` - Create thermal datasets  
- `RgbDsCreation` - Create RGB datasets
- `MergeParameterDs` - Merge parameter datasets

**Common Pattern**:
```python
ds_creator = HyperSpectralDsCreation(
    logger=logger,
    data_source='server',
    ndi_tuple=(583.85, 507.56),
    # ... other parameters
)
dataset_df = ds_creator.create_dataset()
```

## Configuration Management

### Configuration Files

1. **Main Configuration** (`config/main_config.json`)
   - Global settings
   - Data source configuration
   - Path definitions
   - Experiment settings

2. **Project Configuration** (`config/anthocyanin_config.json`)
   - Project-specific settings
   - Model parameters
   - Feature lists

### Environment Variables

When `data_source` is set to 'server', the following environment variables are required:

```bash
SMB_USERNAME=your_username
SMB_PASSWORD=your_password
SMB_SERVER=server_address
SMB_SHARE=share_name
REMOTE_FOLDER=remote_folder_path
year=2024
date=2024-12-01
```

### ConfigManager Pattern

The ConfigManager follows a hierarchical loading pattern:

1. Load main configuration (required)
2. Load project configuration (optional)
3. Load environment variables (if data_source='server')
4. Setup paths and validate configuration

## 7. Development Guidelines

### 1. Adding New Commands

To add a new CLI command:

1. **Add argument parser** in `utils/arguments.py`
2. **Create command handler** in `cli/commands.py`
3. **Register command** in `COMMAND_HANDLERS` dictionary
4. **Implement workflow** in `cli/workflows.py` or project-specific workflow

Example:
```python
# In commands.py
def handle_new_command(args: Dict, config: ConfigManager) -> None:
    logger = get_logger('new_command')
    workflow = get_workflow('new', config)
    workflow.execute_new_task(args)

# Register in COMMAND_HANDLERS
COMMAND_HANDLERS['new'] = handle_new_command
```

### 2. Adding New Image Types

To add a new image processing type:

1. **Create class** in `image_processing/` inheriting from `ImageProcessing`
2. **Implement required methods**:
   - `__init__()` - Initialize with image-specific parameters
   - `process_image()` - Main processing logic
   - `extract_features()` - Feature extraction
3. **Add to __init__.py** exports

### 3. Adding New Projects

To add support for a new project type:

1. **Create project config** in `config/`
2. **Create workflow classes** in `cli/` or `cli/project_workflow.py`
3. **Update workflow factory** in `cli/workflows.py`
4. **Add project-specific logic** as needed

### 4. Code Style Guidelines

- **Use relative imports** within packages: `from .module import Class`
- **Follow ConfigManager pattern** for all path resolution
- **Use logging** via `utils.logger` for consistent logging
- **Add type hints** for all public methods
- **Document classes** with clear parameter descriptions

## 8. Examples and Usage Patterns

### 1. Dataset Creation Pattern

```python
# Standard dataset creation
workflow = DatasetCreationWorkflow(config)
args = {
    'hs': True,
    'th': True,
    'rgb': False,
    'ndi_tuple': (583.85, 507.56),
    'create_ndi_table': True
}
workflow.create_datasets(args)
```

### 2. ML Training Pattern

```python
# Regression training
workflow = MLTrainingWorkflow(config)
args = {
    'dataset': 'path/to/dataset.csv',
    'features': ['ndvi', 'ndi', 'temperature'],
    'target': 'anthocyanin',
    'task': 'regression',
    'model': 'RandomForest'
}
results = workflow.train_models(args)
```

### 3. Image Processing Pattern

```python
# Hyperspectral image processing
hs_image = HyperSpectralImage(
    img=hyperspectral_data,
    wl=wavelengths,
    img_name='image_001',
    ndi_tuple=(583.85, 507.56)
)

# Calculate indices
ndvi = hs_image.ndvi_
ndvi_thresholded = hs_image.create_ndvi_threshold(threshold=0.3)

# Split into objects
hs_image.split_image_to_objects(
    image_num='001',
    ANNOTATION_PATH='path/to/annotations.csv'
)
```

### 4. Configuration Pattern

```python
# Initialize configuration
config = ConfigManager(
    env_file='.env',
    project_config='anthocyanin_config.json'
)

# Get paths
dataset_path = config.get_dataset_path('anthocyanin_dataset')
output_dir = config.ensure_output_dir()

# Access configuration values
data_source = config.get('data_source')
experiment_settings = config.get('experiment_settings', {})
```

### 5. Error Handling Pattern

```python
try:
    workflow = DatasetCreationWorkflow(config)
    workflow.create_datasets(args)
except FileNotFoundError as e:
    logger.error(f"Dataset file not found: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    if verbose:
        logger.debug(traceback.format_exc())
    sys.exit(1)
```

## 9. Testing Guidelines

### Unit Testing Pattern

```python
import unittest
from unittest.mock import Mock, patch
from utils.config import ConfigManager

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.config = ConfigManager(env_file='test.env')
    
    def test_get_dataset_path(self):
        path = self.config.get_dataset_path('test_dataset')
        self.assertIsInstance(path, str)
        
    @patch('os.path.exists')
    def test_load_main_config_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            ConfigManager()
```

### Integration Testing Pattern

```python
def test_create_command_integration():
    # Test full create command workflow
    args = {'hs': True, 'th': False, 'rgb': False}
    config = ConfigManager(env_file='test.env')
    
    workflow = DatasetCreationWorkflow(config)
    result = workflow.create_datasets(args)
    
    assert result is not None
    assert 'spectral_img_df' in result
```

## 10. Performance Considerations

### 1. Memory Management
- Use generators for large dataset processing
- Clear intermediate variables when possible
- Monitor memory usage with logging

### 2. Parallel Processing
- Use `multiprocessing` for CPU-intensive tasks
- Batch process images when possible
- Consider GPU acceleration for ML training

### 3. Caching Strategy
- Cache computed indices (NDVI, NDI)
- Store intermediate results to avoid recomputation
- Use pickle for model serialization

## 11. Debugging and Troubleshooting

### Common Issues

1. **Import Errors**: Check relative imports and package structure
2. **Path Issues**: Use ConfigManager for all path resolution
3. **Configuration Errors**: Validate JSON syntax and required fields
4. **Memory Issues**: Monitor dataset sizes and use chunking

### Debugging Tools

```python
# Enable verbose logging
python main.py --verbose create --hs

# Debug with pdb
import pdb; pdb.set_trace()

# Profile performance
import cProfile
cProfile.run('workflow.create_datasets(args)')
```


## 12. Anthocyanin Project: Detection of Anthocyanin in Lettuces

### Table of Contents for Anthocyanin Project
12.1. [Raw Data](#raw-data)
&nbsp;&nbsp;&nbsp;&nbsp;12.1.1. [Data Sources](#data-sources)
&nbsp;&nbsp;&nbsp;&nbsp;12.1.1.1. [HS: Hyper-spectral](#hs-hyper-spectral)
&nbsp;&nbsp;&nbsp;&nbsp;12.1.2. [Local Data Organization](#local-data-organization)
&nbsp;&nbsp;&nbsp;&nbsp;12.1.3. [Server Data Organization](#server-data-organization)
&nbsp;&nbsp;&nbsp;&nbsp;12.1.4. [Folder Types and Contents](#folder-types-and-contents)
&nbsp;&nbsp;&nbsp;&nbsp;12.1.5. [Server Connection Configuration](#server-connection-configuration)
&nbsp;&nbsp;&nbsp;&nbsp;12.1.6. [Data Access Patterns](#data-access-patterns)
&nbsp;&nbsp;&nbsp;&nbsp;12.1.7. [Experimental Categories](#experimental-categories)

### Raw Data

The Anthocyanin project processes multiple types of imaging data to detect anthocyanin levels in lettuce plants. The raw data is organized differently depending on whether it's stored locally or accessed from a remote server.

### 12.1.1. Data Sources

The project utilizes three main types of imaging data:

1. **RGB Images** - Standard color photographs for visual analysis
2. **Hyperspectral Images** - Multi-wavelength spectral data (.hdr/.img format)
3. **Thermal Images** - Temperature distribution data (.tiff/.csv format)

#### 12.1.1.1. HS: Hyper-spectral

In the Anthocyanin project, hyperspectral images contain multiple lettuce objects (plants) per image. The processing pipeline involves splitting these objects and computing vegetation indices for each individual plant.

**Object Splitting Process:**

The project uses a configuration-driven approach to split hyperspectral images into individual objects:

1. **Binary Image Creation** - First, a binary image is created in the `HyperSpectralImage` class using predefined methods (typically NDVI thresholding, configurable via project settings)

2. **Object Separation** - The binary image is sent to the `ImageProcessing` class, which uses bounding box coordinates from CSV annotation files to separate individual lettuce objects

3. **Data Storage** - Each object's binary pixel values are stored in a DataFrame called `obj_df` for further processing

**Bounding Box Integration:**

The splitting process relies on bounding box CSV files that contain:
- `image_name` - Image filename
- `bbox_x`, `bbox_y` - Top-left corner coordinates
- `bbox_width`, `bbox_height` - Dimensions of bounding box
- `label_name` - Object identifier

**Vegetation Indices Computation:**

After object separation, vegetation indices (NDVI, NDI) are computed for each individual lettuce object, allowing for plant-level analysis rather than image-level analysis. This enables:
- Individual plant health assessment
- Precise anthocyanin quantification per plant
- Better correlation with experimental conditions

**Configuration Control:**

The splitting behavior is controlled by:
- `SPLIT_IMAGE_TO_OBJECTS` - Boolean parameter in experiment settings
- `ANNOTATION_FILE` - Path to bounding box CSV file
- Rotation parameters for image orientation correction

### 12.1.2. Local Data Organization

For local processing, the raw data follows this structure:

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

**Note**: Currently, hyperspectral and thermal image processing from local folders is not implemented. These data types must be accessed from the server.

### 12.1.3. Server Data Organization

When using server-based data source (configured via `.env` file), the data is organized hierarchically:

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

### 12.1.4. Folder Types and Contents

- **RGB/**: Contains standard RGB image files for visual analysis
- **HS/**: Hyperspectral images in .hdr/.img format with spectral wavelength data
- **TH/**: Thermal images in .tiff/.csv format with temperature information
- **<NUMBER_X*>**: Individual experiment folders representing different experimental conditions or time points
- **<ANNOTATION_FILE>.csv**: Bounding box annotations for object detection and image segmentation

### 12.1.5. Server Connection Configuration

To access server data, configure the `.env` file with:

```env
SMB_USERNAME=your_username
SMB_PASSWORD=your_password
SMB_SERVER=server_address
SMB_SHARE=share_name
REMOTE_FOLDER=path/to/data
year= YYYY (for example: 2025)
date= MM-DD (for example: 03-12)
```

### 12.1.6. Data Access Patterns

1. **Local RGB Processing**: Direct file system access to local RGB images
2. **Server Access**: SMB connection for hyperspectral and thermal data
3. **Annotation Integration**: CSV files containing bounding box coordinates for object segmentation
4. **Date-Based Organization**: Data organized by experimental dates for temporal analysis

### 12.1.7. Experimental Categories

The Anthocyanin project uses specific LED treatment categories defined in `anthocyanin_config.json`:

- **RED_white_blue_led_ids**: Plants under red, white, and blue LED combinations
- **GREEN_white_blue_led_ids**: Plants under green, white, and blue LED combinations  
- **RED_white_led_ids**: Plants under red and white LED combinations
- **GREEN_white_led_ids**: Plants under green and white LED combinations
- **RED_Shade_ids**: Plants under red LED with shade conditions
- **GREEN_Shade_ids**: Plants under green LED with shade conditions
- **RED_Control_ids**: Control plants under red LED conditions
- **GREEN_Control_ids**: Control plants under green LED conditions

These categories help organize the experimental data for different lighting conditions and their effects on anthocyanin production in lettuce plants.

This manual provides a comprehensive guide for developers working with the Phenomobile codebase. It covers the architecture, class hierarchy, command flows, and development patterns needed to effectively maintain and extend the project.
