#!/bin/bash

dataset="PenDigits"
mkdir $dataset
wget https://timeseriesclassification.com/Downloads/${dataset}.zip
unzip ${dataset}.zip -d $dataset/
rm ${dataset}.zip

dataset="RightWhaleCalls"
wget https://timeseriesclassification.com/Downloads/${dataset}.zip
unzip ${dataset}.zip
rm ${dataset}.zip
mv WhaleCalls RightWhaleCalls

wget https://cloudstor.aarnet.edu.au/plus/index.php/s/pRLVtQyNhxDdCoM?path=%2FDataset%2FSITS_2006_NDVI_C%2FSITS1M_fold1

mv pRLVtQyNhxDdCoM?path=%2FDataset%2FSITS_2006_NDVI_C%2FSITS1M_fold1 crops.csv
