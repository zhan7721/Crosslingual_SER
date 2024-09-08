import transformers
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
from torch import nn
from datasets import load_from_disk
from utils import *
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
transformers.set_seed(SEED)

CN_MODEL_NAME="TencentGameMate/chinese-wav2vec2-base"
DE_MODEL_NAME="facebook/wav2vec2-base-de-voxpopuli-v2"
EN_MODEL_NAME="facebook/wav2vec2-base-960h"
DE_FOLD_DATASET_PATH="./saved_dataset/de_5fold"
CN_FOLD_DATASET_PATH="./saved_dataset/cn_5fold"
EN_FOLD_DATASET_PATH="./saved_dataset/en_5fold"
DATASET_LABEL=['train', 'eval', 'test']
FOLD_LABEL=['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
EMOTION_LABEL={'Angry': 0, 'Happy': 1, 'Neutral': 2, 'Sad': 3}
BATCH_SIZE=32
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_PATH=os.getcwd()

# Load models
cn_model = Wav2Vec2Model.from_pretrained(CN_MODEL_NAME, output_hidden_states=True).to(DEVICE)
de_model = Wav2Vec2Model.from_pretrained(DE_MODEL_NAME, output_hidden_states=True).to(DEVICE)
en_model = Wav2Vec2Model.from_pretrained(EN_MODEL_NAME, output_hidden_states=True).to(DEVICE)
cn_featext = Wav2Vec2FeatureExtractor.from_pretrained(CN_MODEL_NAME)
de_featext = Wav2Vec2FeatureExtractor.from_pretrained(DE_MODEL_NAME)
en_featext = Wav2Vec2FeatureExtractor.from_pretrained(EN_MODEL_NAME)

# Get speech representations and do mean pooling
def meanpool_cn(batch=1):
    for item in batch:
        audios = torch.tensor(item['audio']['array']).unsqueeze(0).to(DEVICE)
        labels = torch.tensor(item['label']).unsqueeze(0).to(DEVICE)

    inputs = cn_featext(audios, return_tensors="pt", sampling_rate=16000)
    input_values = inputs['input_values'].squeeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = cn_model(input_values=input_values)

    representations = outputs.hidden_states
    pooled_features = torch.tensor([rep.squeeze(0).mean(dim=0).tolist() for rep in representations]).to(DEVICE)

    return {'pooled_feature': pooled_features, 'label': labels}

def meanpool_de(batch=1):
    for item in batch:
        audios = torch.tensor(item['audio']['array']).unsqueeze(0).to(DEVICE)
        labels = torch.tensor(item['label']).unsqueeze(0).to(DEVICE)

    inputs = de_featext(audios, return_tensors="pt", sampling_rate=16000)
    input_values = inputs['input_values'].squeeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = de_model(input_values=input_values)

    representations = outputs.hidden_states
    pooled_features = torch.tensor([rep.squeeze(0).mean(dim=0).tolist() for rep in representations]).to(DEVICE)

    return {'pooled_feature': pooled_features, 'label': labels}

def meanpool_en(batch=1):
    for item in batch:
        audios = torch.tensor(item['audio']['array']).unsqueeze(0).to(DEVICE)
        labels = torch.tensor(item['label']).unsqueeze(0).to(DEVICE)

    inputs = en_featext(audios, return_tensors="pt", sampling_rate=16000)
    input_values = inputs['input_values'].squeeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = en_model(input_values=input_values)

    representations = outputs.hidden_states
    pooled_features = torch.tensor([rep.squeeze(0).mean(dim=0).tolist() for rep in representations]).to(DEVICE)

    return {'pooled_feature': pooled_features, 'label': labels}


def load_dataset(model_name, dataset_split, dataset_name, fold_name):
    """
    Load dataset according to arguments.
    dataset_split: str; train, eval or test
    dataset_name: str; de, cn or en
    """
    if dataset_name == "de":
        print(f"Loading {dataset_name} {dataset_split} data: {fold_name}...")
        dataset_path = os.path.join(DE_FOLD_DATASET_PATH, fold_name)
    elif dataset_name == "cn":
        print(f"Loading {dataset_name} {dataset_split} data: {fold_name}...")
        dataset_path = os.path.join(CN_FOLD_DATASET_PATH, fold_name)
    elif dataset_name == "en": 
        print(f"Loading {dataset_name} {dataset_split} data: {fold_name}...")
        dataset_path = os.path.join(EN_FOLD_DATASET_PATH, fold_name)  
    else:
        raise ValueError(f"NO dataset named {dataset_name}.")
    
    selected_ds = load_from_disk(dataset_path)

    if model_name == "cn":
        ds = DataLoader(selected_ds[dataset_split], batch_size=1, shuffle=True, collate_fn=meanpool_cn)
    elif model_name == "de":
        ds = DataLoader(selected_ds[dataset_split], batch_size=1, shuffle=True, collate_fn=meanpool_de)
    elif model_name == "en":
        ds = DataLoader(selected_ds[dataset_split], batch_size=1, shuffle=True, collate_fn=meanpool_en)
    else:
        raise ValueError(f"NO model named {model_name}.")   

    return ds

if __name__ == "__main__":

    # create parser
    parser = argparse.ArgumentParser(description="Modify model and dataset settings")

    # Arguments
    parser.add_argument('-m', '--model', type=str, required=True, help='Which model to use: de, cn or en')
    parser.add_argument('-t', '--train', type=str, required=True, help='Training datasets: de, cn or en')
    parser.add_argument('-e', '--eval', type=str, required=False, help='Evaluation datasets: de, cn or en')
    parser.add_argument('-T', '--test', type=str, required=True, help='Test datasets: de, cn or en')
    parser.print_help()

    args = parser.parse_args()

    # loop for every fold: read data and train model
    for fold in FOLD_LABEL:

        fold_layer_acc = []
        fold_emo_precision = {'Angry': [], 'Happy': [], 'Neutral': [], 'Sad': []}
        fold_emo_recall = {'Angry': [], 'Happy': [], 'Neutral': [], 'Sad': []}
        fold_emo_f1 = {'Angry': [], 'Happy': [], 'Neutral': [], 'Sad': []}

        print(f"\n======================= This is {fold} on {args.model} =======================")
        
        # load data
        print("\nLoad dataset: ")
        train_output = load_dataset(args.model, "train", args.train, fold)
        test_output = load_dataset(args.model, "test", args.test, fold)
        if args.eval:
            eval_output = load_dataset(args.model, "eval", args.eval, fold)

        # initialise the same cls head for each layer
        os.makedirs("init_cls_model", exist_ok=True)
        init_model = NeuralNet()
        torch.save(init_model, "init_model.pt")
        init_model_path = os.path.join(ROOT_PATH, "init_cls_model", "init_model.pt")

        # loop for every layer: init and train the cls head
        for layer in range(13):
            print(f"\nTraining layer {layer}...")
            # get the representations for certain layer, and rebatch for training
            train_cls = dataloader_for_cls(train_output, layer, batch_size=BATCH_SIZE)
            test_cls = dataloader_for_cls(test_output, layer, batch_size=BATCH_SIZE)
            if args.eval:
                eval_cls = dataloader_for_cls(eval_output, layer, batch_size=BATCH_SIZE)

            # load initialised model every time
            print(f"Loading initialized model for layer {layer}...")
            cls_model_i = torch.load(init_model_path)
            optimizer_i = torch.optim.AdamW(cls_model_i.parameters(), lr=1e-4, eps=1e-8, weight_decay=1e-5)
            criterion_i = nn.CrossEntropyLoss()
            
            best_acc = 0.0
            num_epochs = 100
            best_performance = 0.0
            lowest_eval_loss = 1e10

            # start traning the model
            for epoch in range(num_epochs):
                print(f"Epoch {epoch+1}/{num_epochs}")

                # training phase
                print("\nTraining Phase: ")
                cls_model_i.train()
                train_loss = 0.0
                emolabels_train = []
                predictions_train = []

                for inputs, labels in tqdm(train_cls, desc="Training", leave=True):
                    labels = labels.squeeze(1)
                    optimizer_i.zero_grad()
                    outputs = cls_model_i(inputs.to(DEVICE))
                    loss = criterion_i(outputs, labels.to(DEVICE))
                    loss.backward()
                    optimizer_i.step()
                    train_loss += loss.item()
                    emolabels_train.extend(labels.cpu().numpy())
                    predictions_train.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                
                # print training results: accuracy, and precision, recall & f1 for each emotion
                acc_train = accuracy_score(emolabels_train, predictions_train)
                macro_f1 = f1_score(emolabels_train, predictions_train, average='macro')
                print(f"Training loss: {train_loss:.4f}, Training accuracy: {acc_train:.4f}")
                print(f"Macro F1-score: {macro_f1:.4f}")
                for emo, emo_idx in EMOTION_LABEL.items():
                    precision, recall, f1score = get_acc_by_emo_sk(predictions_train, emolabels_train, emo_idx)
                    print(f"Model performance on {emo} speech (in training): ")
                    print(f"\tPrecision: {precision:.4f}, Recall: {recall:.4f}, F1_score: {f1score:.4f}")

                # eval phase
                print("\nEvaluation Phase: ")
                cls_model_i.eval()
                eval_loss = 0.0
                emolabels_eval = []
                predictions_eval = []

                if not args.eval:
                    eval_cls = test_cls
                    print("No EVAL dataset loaded. Use test set as eval set.")

                with torch.no_grad():
                    for inputs, labels in tqdm(eval_cls, desc="Evaluating", leave=True):
                        labels = labels.squeeze(1)
                        outputs = cls_model_i(inputs.to(DEVICE))
                        loss = criterion_i(outputs, labels.to(DEVICE))
                        eval_loss += loss.item()
                        emolabels_eval.extend(labels.cpu().numpy())
                        predictions_eval.extend(torch.argmax(outputs, dim=1).cpu().numpy())

                # print eval results: accuracy, and precision, recall & f1 for each emotion
                acc_eval = accuracy_score(emolabels_eval, predictions_eval)
                print(f"Validation loss: {eval_loss:.4f}, Validation accuracy: {acc_eval:.4f}")
                macro_f1_eval = f1_score(emolabels_eval, predictions_eval, average='macro')
                print(f"Macro F1-score: {macro_f1_eval:.4f}")
                for emo, emo_idx in EMOTION_LABEL.items():
                    precision, recall, f1score = get_acc_by_emo_sk(predictions_eval, emolabels_eval, emo_idx)
                    print(f"Model performance on {emo} speech (in validation): ")
                    print(f"\tPrecision: {precision:.4f}, Recall: {recall:.4f}, F1_score: {f1score:.4f}")

                # save the best checkpoint
                if acc_eval > best_acc:
                    best_acc = acc_eval
                    path_name = f"./trained_cls_model/{args.model}_{fold}_trained_{args.train}_tested_{args.test}" 
                    os.makedirs(path_name, exist_ok=True)
                    model_name = f"layer{layer}.pt"
                    best_model_path = os.path.join(path_name, model_name) 
                    torch.save(cls_model_i, best_model_path)
                    print(f"New best accuracy for layer {layer} on epoch {epoch+1}: {best_acc:.4f}. Model saved.")

                # check stop criteria
                if eval_loss <= lowest_eval_loss:
                    lowest_eval_loss = eval_loss
                    stop_criteria = 10
                else:
                    stop_criteria -= 1
                if stop_criteria == 0:
                    print(f"Validation loss does not decrease for 10 epochs. End training.")
                    break

            print(f"Model best accuracy on validation set: {best_acc:.4f}")
            
            # test phase
            print("\nTest Phase: ")
            cls_model_i_test = torch.load(best_model_path)
            cls_model_i_test.eval()
            test_loss = 0.0
            emolabels_test = []
            predictions_test = []
            
            with torch.no_grad():
                for inputs, labels in tqdm(test_cls, desc="Testing", leave=False):
                    labels = labels.squeeze(1)
                    outputs = cls_model_i_test(inputs.to(DEVICE))
                    loss = criterion_i(outputs, labels.to(DEVICE))
                    test_loss += loss.item()
                    emolabels_test.extend(labels.cpu().numpy())
                    predictions_test.extend(torch.argmax(outputs, dim=1).cpu().numpy())

            # print test results: accuracy, and precision, recall & f1 for each emotion
            acc_test = accuracy_score(emolabels_test, predictions_test)
            print(f"Test loss: {test_loss:.4f}, Test accuracy: {acc_test:.4f}")
            macro_f1_test = f1_score(emolabels_test, predictions_test, average='macro')
            print(f"Macro F1-score: {macro_f1_test:.4f}")
            fold_layer_acc.append("%.4f" % acc_test)
            for emo, emo_idx in EMOTION_LABEL.items():
                precision, recall, f1score = get_acc_by_emo_sk(predictions_test, emolabels_test, emo_idx)
                fold_emo_precision[emo].append("%.4f" % precision)
                fold_emo_recall[emo].append("%.4f" % recall)
                fold_emo_f1[emo].append("%.4f" % f1score)
                print(f"Model performance on {emo} speech (in test): ")
                print(f"\tPrecision: {precision:.4f}, Recall: {recall:.4f}, F1_score: {f1score:.4f}")

            # draw confusion matrix
            conf_matrix = confusion_matrix(emolabels_test, predictions_test)

            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='viridis', ax=ax)

            ax.set_xlabel('Model predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title(f"{args.model} {fold} trained {str(args.train).upper()} tested {str(args.test).upper()}")
            ax.xaxis.set_ticklabels(EMOTION_LABEL)
            ax.yaxis.set_ticklabels(EMOTION_LABEL)
            fig_path = f"./figs/layer_on_test_cfm/{args.model}_{fold}_trained_{args.train}_tested_{args.test}"
            os.makedirs(fig_path, exist_ok=True)
            fig_name = f"layer{layer}.png"
            fig_savepath = os.path.join(fig_path, fig_name)
            fig.savefig(fig_savepath)
        
        # output fold results
        print(f"\n{args.model}, {fold}, all layer accuracy: {fold_layer_acc}")
        print(f"{args.model}, {fold}, all emo precision: {fold_emo_precision}")
        print(f"{args.model}, {fold}, all emo recall: {fold_emo_recall}")
        print(f"{args.model}, {fold}, all emo f1score: {fold_emo_f1}")


    
