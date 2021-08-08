# Command to download dataset:
#   bash script_download_all_datasets.sh

############
# TSP
############

DIR=TSP/
cd $DIR

FILE=TSP.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/qga6q0gxx3wb8k0/TSP.pkl?dl=1 -o TSP.pkl -J -L -k
fi

FILE=processed_TSP_data.pth
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://drive.google.com/u/1/uc?export=download&confirm=mOyO&id=1xSA7487npv7q9FgxZKNUvd_VOOyChiDH -o processed_TSP_data.pth -J -L -k
fi

cd ..

############
# PATTERN and CLUSTER 
############

DIR=SBMs/
cd $DIR


FILE=SBM_CLUSTER.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/SBM_CLUSTER.pkl -o SBM_CLUSTER.pkl -J -L -k
fi

FILE=processed_CLUSTER_data.pth
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://drive.google.com/u/1/uc?export=download&confirm=o8r0&id=1yMPVq6cysiddGznYWJy2Z73mWRwoBe4d -o processed_CLUSTER_data.pth -J -L -k
fi

FILE=SBM_PATTERN.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/SBM_PATTERN.pkl -o SBM_PATTERN.pkl -J -L -k
fi

FILE=processed_PATTERN_data.pth
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://drive.google.com/u/1/uc?export=download&confirm=sGyR&id=1WFvuQXXnzsH_XQwFfQ2MHFuYjTYmG48W -o processed_PATTERN_data.pth -J -L -k
fi

cd ..