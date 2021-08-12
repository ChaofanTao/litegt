from data.data import LoadData
import torch
import time

def preprocessing_data(DATASET_NAME):
    dataset = LoadData(DATASET_NAME)
    # add laplacian positional encoding
    st = time.time()
    print("[!] Adding Laplacian positional encoding.")
    if DATASET_NAME == 'SBM_PATTERN':
       dataset._add_laplacian_positional_encodings(2) # net_params['pos_enc_dim'] = 2
    else: 
        dataset._add_laplacian_positional_encodings(10) # net_params['pos_enc_dim'] = 10
    print('Time LapPE:{}'.format(time.time()-st))

    # add jaccard similarity
    st = time.time()
    print("[!] Adding jaccard similarity to node pairs..")
    dataset._compute_jaccard_similarity()
    print('Time taken to add jaccard similarity to node pairs:{}'.format(time.time()-st))

    # save processed data
    processed_SBMs_data = {
            "dataset": dataset,
            }
    data_dir='./data/SBMs'
    if DATASET_NAME == 'SBM_PATTERN':
        torch.save(processed_SBMs_data, '{}.pth'.format(data_dir + "/processed_PATTERN_data"))
    elif DATASET_NAME == 'SBM_CLUSTER':
        torch.save(processed_SBMs_data, '{}.pth'.format(data_dir + "/processed_CLUSTER_data"))

dataset_name = ['SBM_PATTERN', 'SBM_CLUSTER']
for name in dataset_name:
    preprocessing_data(name)
