# Command to download dataset:
#   bash script_download_TSP.sh

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
