from data.data import LoadData 
import torch
import time

def preprocessing_data(DATASET_NAME):
    dataset = LoadData(DATASET_NAME)
    # add laplacian positional encoding
    st = time.time()
    print("[!] Adding Laplacian positional encoding.")
    dataset._add_laplacian_positional_encodings(8) # net_params['pos_enc_dim'] = 8
    print('Time LapPE:{}'.format(time.time()-st))

    # add jaccard similarity
    st = time.time()
    print("[!] Adding jaccard similarity to node pairs..")
    dataset._compute_jaccard_similarity()
    print('Time taken to add jaccard similarity to node pairs:{}'.format(time.time()-st))

    # save processed data
    processed_TSP_data = {
            "dataset": dataset,
            }
    data_dir='./data/TSP'
    torch.save(processed_TSP_data, '{}.pth'.format(data_dir + "/processed_TSP_data"))

dataset_name = ['TSP']
for name in dataset_name:
    preprocessing_data(name)
