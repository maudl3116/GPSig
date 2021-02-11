#!/bin/bash
mkdir weather_data
cd weather_data
for year in {2004..2020} 
do
    wget https://www.bgc-jena.mpg.de/wetter/mpi_roof_${year}a.zip
    wget https://www.bgc-jena.mpg.de/wetter/mpi_saale_${year}a.zip
    wget https://www.bgc-jena.mpg.de/wetter/mpi_roof_${year}b.zip
    wget https://www.bgc-jena.mpg.de/wetter/mpi_saale_${year}b.zip
    unzip mpi_roof_${year}a.zip
    unzip mpi_roof_${year}b.zip
    unzip mpi_saale_${year}a.zip
    unzip mpi_saale_${year}b.zip
done

rm *.zip
