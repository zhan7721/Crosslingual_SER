from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
import datasets
from datasets import load_from_disk
from utils import *
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# load fine-tuned classification model
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE=8
EMOTION_LABEL={'Angry': 0, 'Happy': 1, 'Neutral': 2, 'Sad': 3}

# load models
cn_model = Wav2Vec2Model.from_pretrained(CN_MODEL_NAME, output_hidden_states=True).to(DEVICE)
de_model = Wav2Vec2Model.from_pretrained(DE_MODEL_NAME, output_hidden_states=True).to(DEVICE)
en_model = Wav2Vec2Model.from_pretrained(EN_MODEL_NAME, output_hidden_states=True).to(DEVICE)
cn_featext = Wav2Vec2FeatureExtractor.from_pretrained(CN_MODEL_NAME)
de_featext = Wav2Vec2FeatureExtractor.from_pretrained(DE_MODEL_NAME)
en_featext = Wav2Vec2FeatureExtractor.from_pretrained(EN_MODEL_NAME)

root_dir = "/work/tc062/tc062/zhan7721/wav2vec2_peft/saved_pts_fold"
fold_model = ['fold_0.pt', 'fold_1.pt', 'fold_2.pt', 'fold_3.pt', 'fold_4.pt']

# load dataset
tj_dataset = load_from_disk("./saved_dataset/tj")
cn_dataset = load_from_disk("./saved_dataset/preprocessed_cn_exp_test")
de_dataset = load_from_disk("./saved_dataset/preprocessed_de_exp_test")
en_dataset = load_from_disk("./saved_dataset/preprocessed_en_exp_test")

# resample de data
tj_dataset = tj_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
de_dataset = de_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))

# used on preprocessed dataset
def collate_fn_listbatch(batch):
    audios = [torch.tensor(item['audio']['array']) for item in batch]
    labels = torch.tensor([item['label'] for item in batch])

    return {"audios_list": audios, "labels": labels.to(DEVICE)}

# select models and datasets
def select_cond(model_name, condition_name):
    """
    dataset_split: str; train, eval or test
    dataset_name: str; de, cn or en
    """
    if model_name == "cn":
        if condition_name == "ml":
            model_path = os.path.join(root_dir, "cn_mono")
            return model_path, cn_featext, cn_dataset
        elif condition_name == "cl":
            model_path = os.path.join(root_dir, "cn_cross")
            return model_path, cn_featext, de_dataset
        elif condition_name == "tl":
            model_path = os.path.join(root_dir, "cn_trans")
            return model_path, cn_featext, de_dataset
        elif condition_name == "twostage":
            model_path = os.path.join(root_dir, "cn_trans_twostage")
            return model_path, cn_featext, de_dataset
        elif condition_name == "tlen": 
            model_path = os.path.join(root_dir, "cn_en_trans")
            return model_path, cn_featext, en_dataset
        else:
            raise ValueError(f"NO condition named {model_name}+{condition_name}.")
    elif model_name == "de":
        if condition_name == "ml":
            model_path = os.path.join(root_dir, "de_mono_various")
            return model_path, de_featext, de_dataset
        elif condition_name == "cl":
            model_path = os.path.join(root_dir, "de_cross")
            return model_path, de_featext, cn_dataset
        elif condition_name == "tl":
            model_path = os.path.join(root_dir, "de_trans")
            return model_path, de_featext, cn_dataset
        elif condition_name == "twostage":
            model_path = os.path.join(root_dir, "de_trans_twostage")
            return model_path, de_featext, cn_dataset
        elif condition_name == "tlen": 
            model_path = os.path.join(root_dir, "de_en_trans")
            return model_path, de_featext, en_dataset
        else:
            raise ValueError(f"NO condition named {model_name}+{condition_name}.")
    elif model_name == "en":
        if condition_name == "ml":
            model_path = os.path.join(root_dir, "en_mono")
            return model_path, en_featext, en_dataset
        else: 
            raise ValueError(f"NO condition named {model_name}+{condition_name}.")
    

if __name__ == "__main__":

    # create parser
    parser = argparse.ArgumentParser(description="Modify model settings")

    # Arguments
    parser.add_argument('-m', '--model', type=str, required=True, help='Which model: cn, en or de')
    parser.add_argument('-c', '--condition', type=str, required=True, help='Which condition: ml, cl, tl, twostage, tlen')
    parser.print_help()

    args = parser.parse_args()
    model_dir, featext, test_ds = select_cond(args.model, args.condition)

    # featext
    def do_featext(example):

        audio = torch.tensor(example['audio']['array']).unsqueeze(0).tolist()
        audio_feat = featext(audio, return_tensors="pt", sampling_rate=16000)
        example['audio']['array'] = audio_feat['input_values'].squeeze(0).tolist()

        return example
    
    test_ds = test_ds.map(do_featext)
    ds = DataLoader(test_ds['train'], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_listbatch)

    emolabels_allfold = []
    predictions_allfold = []
    fold_layer_acc = []
    fold_emo_precision = {'Angry': [], 'Happy': [], 'Neutral': [], 'Sad': []}
    fold_emo_recall = {'Angry': [], 'Happy': [], 'Neutral': [], 'Sad': []}
    fold_emo_f1 = {'Angry': [], 'Happy': [], 'Neutral': [], 'Sad': []}

    # test phase
    print(f"\nStart testing {args.model}: {args.condition}")
    for fold in fold_model:
        print(f"\nTest Phase {fold}: ")
        model_path = os.path.join(model_dir, fold)
        model = torch.load(model_path).to(DEVICE)
        model.eval()
        test_loss = 0.0
        emolabels_test = []
        predictions_test = []

        with torch.no_grad():
            for batch in tqdm(ds, desc="Testing", leave=False):
                inputs = batch['audios_list']
                labels = batch['labels']
                outputs = model(input_values=inputs)
                emolabels_test.extend(labels.cpu().numpy())
                predictions_test.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            
        # print test results: accuracy, and precision, recall & f1 for each emotion
        acc_test = accuracy_score(emolabels_test, predictions_test)
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {acc_test:.4f}")
        macro_f1_test = f1_score(emolabels_test, predictions_test, average='macro')
        print(f"Macro F1-score: {macro_f1_test:.4f}")
        fold_layer_acc.append("%.4f" % acc_test)
        emolabels_allfold += emolabels_test
        predictions_allfold += predictions_test

        for emo, emo_idx in EMOTION_LABEL.items():
            precision, recall, f1score = get_acc_by_emo_sk(predictions_test, emolabels_test, emo_idx)
            fold_emo_precision[emo].append("%.4f" % precision)
            fold_emo_recall[emo].append("%.4f" % recall)
            fold_emo_f1[emo].append("%.4f" % f1score)
            print(f"Model performance on {emo} speech (in test): ")
            print(f"\tPrecision: {precision:.4f}, Recall: {recall:.4f}, F1_score: {f1score:.4f}")
    
    print(f"\n{args.model}, {args.condition}, all folds layer accuracy: {fold_layer_acc}")
    print(f"{args.model}, {args.condition}, all emo precision: {fold_emo_precision}")
    print(f"{args.model}, {args.condition}, all emo recall: {fold_emo_recall}")
    print(f"{args.model}, {args.condition}, all emo f1score: {fold_emo_f1}")

    # draw confusion matrix
    conf_matrix = confusion_matrix(emolabels_allfold, predictions_allfold, normalize='true')
    print(f"Confusion matrix of {args.model} {args.condition}:")
    conf_matrix = np.round(conf_matrix, 3)
    print(conf_matrix)

    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='viridis', ax=ax)

    ax.set_xlabel('Model predicted labels')
    ax.set_ylabel('True labels')
    # ax.set_title(f"Confusion Matrix of {args.model} {args.condition} predictions")
    ax.xaxis.set_ticklabels(EMOTION_LABEL)
    ax.yaxis.set_ticklabels(EMOTION_LABEL)
    fig.savefig(f"./figs/{args.model}_{args.condition}.png")
