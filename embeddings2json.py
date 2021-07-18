import json 
import torch

def get_embeddings(model, model_path, dataset):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.double()
    emb_dict = dict()
    for sample in dataset:
        spec_features = sample[0]

        label = str(sample[1])
        embedding = model.forward_once(spec_features)
        embedding = embedding.tolist()
        # print(embedding)
        if label not in emb_dict.keys():
            emb_dict[label] = dict()
            emb_dict[label]['embeddings'] = [embedding]
        else: 
            emb_dict[label]['embeddings'] += [embedding]

    jsonString = json.dumps(emb_dict)
    jsonFile = open("emotion_embeddings_7_5.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()
    
