# Multi-modal learning for Impact-based forecasting of Droughts in Eastern Africa

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WinterSchool2026/ch14-multimodal-drought-forecasting/blob/main/DeepExtremes_SpatioTemporal_DL.ipynb)

## Description
In this challenge, we are exploring deep learning algorithms for forecasting the impact of
climate extremes on vegetation at high resolution. For this, we leverage an existing dataset
of co-aligned Sentinel-2 satellite images and meteorological reanalysis: the
DeepExtremeCubes. We will work with PyTorch and xarray and decide jointly what goals we
want to achieve during the week 🙂

## Recommended reading material
- [https://openaccess.thecvf.com/content/CVPR2024/html/Benson_Multi-modal_Learning_for_Geospatial_Vegetation_Forecasting_CVPR_2024_paper.html](https://openaccess.thecvf.com/content/CVPR2024/html/Benson_Multi-modal_Learning_for_Geospatial_Vegetation_Forecasting_CVPR_2024_paper.html)
- [https://www.nature.com/articles/s41597-025-04447-5](https://www.nature.com/articles/s41597-025-04447-5)

## Metadata Dataframe

```python
import pandas as pd                                                                                                                                                                                                
import s3fs                                                                                                                                                                                                        
                                                                                                                                                                                                                    
fs = s3fs.S3FileSystem(                                                                                                                                                                                            
    anon=True,                                                                                                                                                                                                     
    client_kwargs={"endpoint_url": "https://s3.bgc-jena.mpg.de:9000"},
)                                                                                                                                                                                                                  
                                                                                                                                                                                                                    
# Load the registry CSV directly from S3                                                                                                                                                                           
with fs.open("earthnet/deepextremes/registry.csv") as f:                                                                                                                                                           
    df = pd.read_csv(f, low_memory=False)                                                                                                                                                                          

print(f"Full registry: {len(df)} rows")                                                                                                                                                                            
print(f"Versions:\n{df['version'].str.strip().value_counts()}\n")                                                                                                                                                  
                                                                                                                                                                                                                    
# Subset to version 1.3 only (the ones currently on S3)                                                                                                                                                            
df_v13 = df[df["version"].str.strip() == "1.3"].copy()                                                                                                                                                             
print(f"Version 1.3 minicubes: {len(df_v13)}")
print(df_v13[["mc_id", "version", "class", "dominant_class"]].head())
                    
```
