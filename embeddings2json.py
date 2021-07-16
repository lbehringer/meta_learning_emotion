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
        print(type(embedding))
        print(type(label))
        if label not in emb_dict:
            emb_dict[label] = [embedding]
        else: 
            emb_dict[label] += [embedding]


    #aDict = {"a":54, "b":87}
    jsonString = json.dumps(emb_dict)
    jsonFile = open("emotion_embeddings.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()
    
