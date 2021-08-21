import json
import torch
from dataset import create_classification_set, EmotionDataset
from classification_model import EmotionClassificationNet


def write_embeddings(model, dataset, filename):
    model.eval()
    model.double()
    emb_dict = dict()
    for sample in dataset:
        spec_features = sample[0]

        label = str(sample[1])
        if model.get_type() == "meta_learning":
            embedding = model.forward_once(spec_features)
        else:
            embedding = model.forward(spec_features)
        embedding = torch.squeeze(embedding, 0)
        
        embedding = embedding.tolist()
        # print(embedding)
        if label not in emb_dict.keys():
            emb_dict[label] = dict()
            emb_dict[label]['embeddings'] = [embedding]
            emb_dict[label]['label'] = label
        else:
            emb_dict[label]['embeddings'] += [embedding]

    jsonString = json.dumps(emb_dict)
    jsonFile = open(filename, "w")
    jsonFile.write(jsonString)
    jsonFile.close()
'''
dataset = EmotionDataset('/mount/arbeitsdaten/studenten1/advanced_ml/dengelva/meta_learning_emotion/data/pavoque/pavoque_across_500_dur_4_preemph_norm_0to1.json')
model = EmotionClassificationNet(26, 64, 2, 8, 20, 4, 26)
train, test = create_classification_set(dataset)
get_embeddings(model, 'state_dict_model_classification_iemocap_1000ep_20_08_21_11_28am.pt', test)
'''