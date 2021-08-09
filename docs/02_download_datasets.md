# Download datasets

All the datasets work with DGL 0.5.x or later. Please update the environment using the yml files in the root directory if the use of these datasets throw error(s).



<br>

## 1. TSP dataset
TSP size is 1.87GB.  

```
# At the root of the project
cd data/ 
bash script_download_TSP.sh
```
Script [script_download_TSP.sh](../data/script_download_TSP.sh) is located here. Refer to [benchmarking-gnns repo](https://github.com/graphdeeplearning/benchmarking-gnns) for details on preparation.

<br>

## 2. PATTERN/CLUSTER SBM datasets
PATTERN size is 1.98GB and CLUSTER size is 1.26GB.

```
# At the root of the project
cd data/ 
bash script_download_SBMs.sh
```
Script [script_download_SBMs.sh](../data/script_download_SBMs.sh) is located here. Refer to [benchmarking-gnns repo](https://github.com/graphdeeplearning/benchmarking-gnns) for details on preparation.

<br>

## 3. All datasets

```
# At the root of the project
cd data/ 
bash script_download_all_datasets.sh
```

Script [script_download_all_datasets.sh](../data/script_download_all_datasets.sh) is located here.
NOTE： We also provide pre-processed data in Goolge drive in the download script.  <br>
[processed_CLUSTER_data.pth](https://drive.google.com/u/1/uc?export=download&confirm=o8r0&id=1yMPVq6cysiddGznYWJy2Z73mWRwoBe4d) <br>
[processed_PATTERN_data.pth](https://drive.google.com/u/1/uc?export=download&confirm=sGyR&id=1WFvuQXXnzsH_XQwFfQ2MHFuYjTYmG48W) <br>
[processed_TSP_data.pth](https://drive.google.com/u/1/uc?export=download&confirm=mOyO&id=1xSA7487npv7q9FgxZKNUvd_VOOyChiDH)
  
<br><br><br>
