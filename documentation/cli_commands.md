# CLI commands

## merge example

(imgProc2) davidvol@bd-10-lanaa:~/projects/phenomobile/src$ python main.py merge  --hp-dataset 'hyper_sp_imgs_dataset_created_at_20251229_from_BneiAtarot_2026-01-04 15.csv' --ref-dataset 'Anthocyanin_BENI_ATAROT_2025-12-25.csv' --project 'lettuce_BENI_ATAROT'
[INFO] Starting phenomobile with command: merge
[INFO] Arguments: {'env_file': '.env', 'config': 'anthocyanin_config.json', 'verbose': False, 'command': 'merge', 'hp_dataset': 'hyper_sp_imgs_dataset_created_at_20251229_from_BneiAtarot_2026-01-04 15.csv', 'ref_dataset': 'Anthocyanin_BENI_ATAROT_2025-12-25.csv', 'project': 'lettuce_BENI_ATAROT'}
[INFO] Step: Starting dataset merge
[INFO] Step: Initializing dataset merge
new file created :/home/davidvol/projects/phenomobile/datasets/BneiAtarot/complete_ds_of_Anthocyanin_BENI_ATAROT_2025-12-25_2026-02-26 14.csv
[INFO] Merged dataset shape: (63, 22)
[INFO] Merged dataset columns: ['img_num', 'label_name', 'longitude', 'latitude', 'acquisition date', 'FVC', 'ndvi_mean', 'ndvi_std', 'ndvi_median', 'gndvi_mean', 'evi_mean', 'lai_mean', 'NDI_(583.85, 507.56)', 'category', 'catalog id', 'color', 'Leaf sample weight (mg)', 'Sample disc weight (g)', 'Anthocyanins OD-530 nm', 'Chlorophyll OD-657 nm', 'Chlorophyll interference', 'Anthocyanin']
[INFO] merge_datasets completed in 0.01 seconds
[INFO] Command 'merge' completed successfully



 
## train example
(imgProc2) davidvol@bd-10-lanaa:~/projects/phenomobile/src$ python main.py train --dataset 'complete_HS_ds_of_Anthocyanin_BENI_ATAROT_2025_12_ts_1767538506.77255.csv' --features 'NDI_(583.85, 507.56)','Anthocyanin' --target Anthocyanin 
[INFO] Starting phenomobile with command: train
[INFO] Arguments: {'env_file': '.env', 'config': 'anthocyanin_config.json', 'verbose': False, 'command': 'train', 'dataset': 'complete_HS_ds_of_Anthocyanin_BENI_ATAROT_2025_12_ts_1767538506.77255.csv', 'features': ['NDI_(583.85, 507.56)', 'Anthocyanin'], 'target': 'Anthocyanin', 'task': 'regression', 'model': None, 'filter_condition': None, 'filter_indicator': None, 'plot_separate': False}
[INFO] Step: Starting ML training
[INFO] Step: Initializing ML training
[INFO] Training dataset shape: (107, 22)
[INFO] Training dataset columns: ['img_num', 'label_name', 'longitude', 'latitude', 'acquisition date', 'FVC', 'ndvi_mean', 'ndvi_std', 'ndvi_median', 'gndvi_mean', 'evi_mean', 'lai_mean', 'NDI_(583.85, 507.56)', 'category', 'catalog id', 'color', 'Leaf sample weight (mg)', 'Sample disc weight (g)', 'Anthocyanins OD-530 nm', 'Chlorophyll OD-657 nm', 'Chlorophyll interference', 'Anthocyanin']

=== MODEL EVALUATION RESULTS ===
     target     predictable_features              model_name       R2     RMSE
Anthocyanin ['NDI_(583.85, 507.56)']           XGB Regressor 0.999908 0.002837
Anthocyanin ['NDI_(583.85, 507.56)'] Random Forest Regressor 0.940303 0.072202
Anthocyanin ['NDI_(583.85, 507.56)']       Linear Regression 0.530793 0.202421
qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in ""
[INFO] train_models completed in 37.32 seconds
[INFO] Command 'train' completed successfully


## train exmaple with light filter
(imgProc2) davidvol@bd-10-lanaa:~/projects/phenomobile/src$ python main.py train --dataset 'complete_HS_ds_of_Anthocyanin_BENI_ATAROT_2025_12_ts_1767538506.77255.csv' --features 'ndvi_mean','lai_mean','evi_mean','NDI_(583.85, 507.56)','Anthocyanin' --target Anthocyanin --filter light
[INFO] Starting phenomobile with command: train
[INFO] Arguments: {'env_file': '.env', 'config': 'anthocyanin_config.json', 'verbose': False, 'command': 'train', 'dataset': 'complete_HS_ds_of_Anthocyanin_BENI_ATAROT_2025_12_ts_1767538506.77255.csv', 'features': ['ndvi_mean', 'lai_mean', 'evi_mean', 'NDI_(583.85, 507.56)', 'Anthocyanin'], 'target': 'Anthocyanin', 'task': 'regression', 'filter': 'light', 'model': None, 'show_plots': False, 'plot_separate': False}
[INFO] Step: Starting ML training
[INFO] Step: Initializing ML training
[INFO] Training dataset shape: (107, 22)
[INFO] Training dataset columns: ['img_num', 'label_name', 'longitude', 'latitude', 'acquisition date', 'FVC', 'ndvi_mean', 'ndvi_std', 'ndvi_median', 'gndvi_mean', 'evi_mean', 'lai_mean', 'NDI_(583.85, 507.56)', 'category', 'catalog id', 'color', 'Leaf sample weight (mg)', 'Sample disc weight (g)', 'Anthocyanins OD-530 nm', 'Chlorophyll OD-657 nm', 'Chlorophyll interference', 'Anthocyanin']

=== MODEL EVALUATION RESULTS ===
     target                                          predictable_features              model_name       R2     RMSE         catalog id
Anthocyanin ['ndvi_mean', 'lai_mean', 'evi_mean', 'NDI_(583.85, 507.56)']           XGB Regressor 0.999983 0.001225                NaN
Anthocyanin ['ndvi_mean', 'lai_mean', 'evi_mean', 'NDI_(583.85, 507.56)'] Random Forest Regressor 0.972447 0.049052                NaN
Anthocyanin ['ndvi_mean', 'lai_mean', 'evi_mean', 'NDI_(583.85, 507.56)']       Linear Regression 0.555335 0.197056                NaN
Anthocyanin ['ndvi_mean', 'lai_mean', 'evi_mean', 'NDI_(583.85, 507.56)']           XGB Regressor 0.999995 0.000512 White and Blue Led
Anthocyanin ['ndvi_mean', 'lai_mean', 'evi_mean', 'NDI_(583.85, 507.56)'] Random Forest Regressor 0.945485 0.051356 White and Blue Led
Anthocyanin ['ndvi_mean', 'lai_mean', 'evi_mean', 'NDI_(583.85, 507.56)']       Linear Regression 0.704333 0.119602 White and Blue Led
Anthocyanin ['ndvi_mean', 'lai_mean', 'evi_mean', 'NDI_(583.85, 507.56)']           XGB Regressor 0.999999 0.000518          White Led
Anthocyanin ['ndvi_mean', 'lai_mean', 'evi_mean', 'NDI_(583.85, 507.56)'] Random Forest Regressor 0.954543 0.098174          White Led
Anthocyanin ['ndvi_mean', 'lai_mean', 'evi_mean', 'NDI_(583.85, 507.56)']       Linear Regression 0.606844 0.288721          White Led
Anthocyanin ['ndvi_mean', 'lai_mean', 'evi_mean', 'NDI_(583.85, 507.56)']           XGB Regressor 0.999989 0.000650              Shade
Anthocyanin ['ndvi_mean', 'lai_mean', 'evi_mean', 'NDI_(583.85, 507.56)'] Random Forest Regressor 0.906911 0.059197              Shade
Anthocyanin ['ndvi_mean', 'lai_mean', 'evi_mean', 'NDI_(583.85, 507.56)']       Linear Regression 0.703127 0.105714              Shade
Anthocyanin ['ndvi_mean', 'lai_mean', 'evi_mean', 'NDI_(583.85, 507.56)']           XGB Regressor 0.999988 0.000757            Control
Anthocyanin ['ndvi_mean', 'lai_mean', 'evi_mean', 'NDI_(583.85, 507.56)'] Random Forest Regressor 0.948774 0.049796            Control
Anthocyanin ['ndvi_mean', 'lai_mean', 'evi_mean', 'NDI_(583.85, 507.56)']       Linear Regression 0.695990 0.121309            Control
qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in ""
[INFO] train_models completed in 3.67 seconds
[INFO] Command 'train' completed successfully
