

# Command to download dataset:
#   bash script_download_SBMs.sh


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


