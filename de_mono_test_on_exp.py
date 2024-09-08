import transformers
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
from torch import nn
import torchaudio
import datasets
from datasets import load_from_disk
from utils import *
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# load fine-tuned classification model
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMOTION_LABEL={'Angry': 0, 'Happy': 1, 'Neutral': 2, 'Sad': 3}
# load models
cn_model = Wav2Vec2Model.from_pretrained(CN_MODEL_NAME, output_hidden_states=True).to(DEVICE)
de_model = Wav2Vec2Model.from_pretrained(DE_MODEL_NAME, output_hidden_states=True).to(DEVICE)
en_model = Wav2Vec2Model.from_pretrained(EN_MODEL_NAME, output_hidden_states=True).to(DEVICE)
cn_featext = Wav2Vec2FeatureExtractor.from_pretrained(CN_MODEL_NAME)
de_featext = Wav2Vec2FeatureExtractor.from_pretrained(DE_MODEL_NAME)
en_featext = Wav2Vec2FeatureExtractor.from_pretrained(EN_MODEL_NAME)
# my model path
de_monoling = torch.load("/work/tc062/tc062/zhan7721/layer_wise_analysis_current/trained_cls_model/de_trained_de_tested_de/0730_layer4.pt").to(DEVICE)

# load dataset
de_dataset = load_from_disk("./saved_dataset/de_exp_test")

# resample de data
tj_dataset = tj_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
de_dataset = de_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))

# map text labels to numbers
cn_dataset = cn_dataset.map(map_labels)
de_dataset = de_dataset.map(map_labels)
en_dataset = en_dataset.map(map_labels)

def collate_fn_mean_pool(batch=1):
    for item in batch:
        audios = torch.tensor(item['audio']['array']).unsqueeze(0).to(DEVICE)
        labels = item['label']
        audio_id = item['audio']['path']

    inputs = de_featext(audios, return_tensors="pt", sampling_rate=16000)
    input_values = inputs['input_values'].squeeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = de_model(input_values=input_values)

    representations = outputs.hidden_states
    pooled_features = torch.tensor([rep.squeeze(0).mean(dim=0).tolist() for rep in representations]).to(DEVICE)

    return {'pooled_feature': pooled_features, 'label': labels, 'audio_id': audio_id}


def load_dataset(dataset_split, dataset_name):
    """
    dataset_split: str; train, eval or test
    dataset_name: str; de, cn or en
    """
    if dataset_name == "de":
        print(f"Loading {dataset_name} {dataset_split} data...")
        ds = DataLoader(de_dataset[dataset_split], batch_size=1, shuffle=True, collate_fn=collate_fn_mean_pool) 
    elif dataset_name == "cn":
        print(f"Loading {dataset_name} {dataset_split} data...")
        ds = DataLoader(cn_dataset[dataset_split], batch_size=1, shuffle=True, collate_fn=collate_fn_mean_pool)
    elif dataset_name == "en": 
        print(f"Loading {dataset_name} {dataset_split} data...")
        ds = DataLoader(en_dataset[dataset_split], batch_size=1, shuffle=True, collate_fn=collate_fn_mean_pool) 
    else:
        raise ValueError(f"NO dataset named {dataset_name}.")
    
    return ds


def select_model(condition_name):
    """
    dataset_split: str; train, eval or test
    dataset_name: str; de, cn or en
    """
    if condition_name == "mono":
        tgt_layer = 4
        return de_monoling, tgt_layer
    elif condition_name == "f_cross":
        tgt_layer = 10
        return de_crossling_former, tgt_layer
    elif condition_name == "c_cross":
        tgt_layer = 5
        return de_crossling_current, tgt_layer    
    elif condition_name == "l2": 
        tgt_layer = 4
        return de_l2_en, tgt_layer
    else:
        raise ValueError(f"NO condition named {condition_name}.")
    

if __name__ == "__main__":

    # create parser
    parser = argparse.ArgumentParser(description="Modify model settings")

    # Arguments
    parser.add_argument('-c', '--condition', type=str, required=True, help='Which condition: mono, cross, l2')
    parser.add_argument('-f', '--featext', type=str, required=False, help='Which feature extractor and w2v2 model to use: de, cn or en')
    parser.add_argument('-T', '--test', type=str, required=True, help='Test datasets: de, cn or en')
    parser.print_help()

    args = parser.parse_args()

    # get layer representations in the dataloader
    test_output = load_dataset("train", args.test)
    model, tgt_layer = select_model(args.condition)
    
    # test phase
    model.eval()
    test_loss = 0.0
    emolabels_test = []
    predictions_test = []
    
    with torch.no_grad():
        for batch in tqdm(test_output, desc="Testing", leave=False):
            inputs = batch['pooled_feature'][tgt_layer].unsqueeze(0)
            labels = torch.tensor(batch['label'])
            outputs = model(inputs.to(DEVICE))
            emolabels_test.append(labels)
            predictions_test.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    # print test results: accuracy, and precision, recall & f1 for each emotion
    acc_test = accuracy_score(emolabels_test, predictions_test)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {acc_test:.4f}")
    macro_f1_test = f1_score(emolabels_test, predictions_test, average='macro')
    print(f"Macro F1-score: {macro_f1_test:.4f}")
    for emo, emo_idx in EMOTION_LABEL.items():
        precision, recall, f1score = get_acc_by_emo_sk(predictions_test, emolabels_test, emo_idx)
        print(f"Model performance on {emo} speech (in test): ")
        print(f"\tPrecision: {precision:.4f}, Recall: {recall:.4f}, F1_score: {f1score:.4f}")
    
    # draw confusion matrix
    conf_matrix = confusion_matrix(emolabels_test, predictions_test)

    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='viridis', ax=ax)

    ax.set_xlabel('Model predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f"Confusion Matrix of DE ({args.condition}) predictions on experiment {str(args.test).upper()} speech")
    ax.xaxis.set_ticklabels(EMOTION_LABEL)
    ax.yaxis.set_ticklabels(EMOTION_LABEL)
    fig.savefig(f"./figs/de_cond_{args.condition}_layer_{tgt_layer}_tested_{args.test}_exp.png")
    # conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=None)
