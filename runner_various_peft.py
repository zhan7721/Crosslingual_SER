import transformers
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor
import torch
from torch import nn
from datasets import load_from_disk
from utils import *
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from peft import LoraConfig, get_peft_model
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
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
BATCH_SIZE=16
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_PATH=os.getcwd()

# load models
cn_model = Wav2Vec2Model.from_pretrained(CN_MODEL_NAME, output_hidden_states=True).to(DEVICE)
de_model = Wav2Vec2Model.from_pretrained(DE_MODEL_NAME, output_hidden_states=True).to(DEVICE)
en_model = Wav2Vec2Model.from_pretrained(EN_MODEL_NAME, output_hidden_states=True).to(DEVICE)
cn_featext = Wav2Vec2FeatureExtractor.from_pretrained(CN_MODEL_NAME)
de_featext = Wav2Vec2FeatureExtractor.from_pretrained(DE_MODEL_NAME)
en_featext = Wav2Vec2FeatureExtractor.from_pretrained(EN_MODEL_NAME)

# do featext for dataset
def featext_de(example):
    audio = torch.tensor(example['audio']['array']).unsqueeze(0).tolist()
    audio_feat = de_featext(audio, return_tensors="pt", sampling_rate=16000)
    example['audio']['array'] = audio_feat['input_values'].squeeze(0).tolist()
    return example

def featext_cn(example):
    audio = torch.tensor(example['audio']['array']).unsqueeze(0).tolist()
    audio_feat = cn_featext(audio, return_tensors="pt", sampling_rate=16000)
    example['audio']['array'] = audio_feat['input_values'].squeeze(0).tolist()
    return example

def featext_en(example):
    audio = torch.tensor(example['audio']['array']).unsqueeze(0).tolist()
    audio_feat = en_featext(audio, return_tensors="pt", sampling_rate=16000)
    example['audio']['array'] = audio_feat['input_values'].squeeze(0).tolist()
    return example

# batch preprocessed dataset in a list for model input
def listbatch(batch):
    audios = [torch.tensor(item['audio']['array']) for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    return {"audios_list": audios, "labels": labels.to(DEVICE)}

# read arguments to load dataset
def load_dataset(model_name, dataset_split, dataset_name, fold_name):
    """
    dataset_split: str; train, eval or test
    dataset_name: str; de, cn or en
    """
    print(f"Loading {dataset_name} {dataset_split} data: {fold_name}...")
    if dataset_name == "de":
        dataset_path = os.path.join(DE_FOLD_DATASET_PATH, fold_name)
    elif dataset_name == "cn":
        dataset_path = os.path.join(CN_FOLD_DATASET_PATH, fold_name)
    elif dataset_name == "en": 
        dataset_path = os.path.join(EN_FOLD_DATASET_PATH, fold_name)  
    else:
        raise ValueError(f"NO dataset named {dataset_name}.")
    
    selected_ds = load_from_disk(dataset_path)

    print(f"Preprocess {dataset_name} {fold_name} data for {model_name} model")
    if model_name == "cn":   
        featext_ds = selected_ds.map(featext_cn)
        ds = DataLoader(featext_ds[dataset_split], batch_size=1, shuffle=True, collate_fn=listbatch)
    elif model_name == "de":
        featext_ds = selected_ds.map(featext_de)
        ds = DataLoader(featext_ds[dataset_split], batch_size=1, shuffle=True, collate_fn=listbatch)
    elif model_name == "en":
        featext_ds = selected_ds.map(featext_en)
        ds = DataLoader(featext_ds[dataset_split], batch_size=1, shuffle=True, collate_fn=listbatch)
    else:
        raise ValueError(f"NO model named {model_name}.")   

    return ds

# read arguments to customise PEFT model
def get_custom_peft_model(model_name, layer_index, lora_r, lora_alpha, lora_dropout, 
                          bn=True, wg=True, ws=False, everylayer=True):

    print(f"Use {model_name} model to add lora")
    if model_name == "cn":   
        custom_w2v2_config = Wav2Vec2Config.from_pretrained(CN_MODEL_SAVE_PATH)
        model_param = cn_model.state_dict()
    elif model_name == "de":
        custom_w2v2_config = Wav2Vec2Config.from_pretrained(DE_MODEL_SAVE_PATH)
        model_param = de_model.state_dict()
    elif model_name == "en":
        custom_w2v2_config = Wav2Vec2Config.from_pretrained(EN_MODEL_SAVE_PATH)
        model_param = en_model.state_dict()
    else:
        raise ValueError(f"NO model named {model_name}.")  
    
    # instantiate a custom w2v2
    custom_w2v2_config.num_hidden_layers = layer_index + 1
    init_custom_w2v2 = ModifiedWav2Vec2Model(config=custom_w2v2_config, bn=bn, wg=wg, every_layer=everylayer, best_layer=layer_index)
    init_custom_w2v2.load_state_dict(model_param, strict=False)
    init_custom_w2v2.to(DEVICE)

    if not everylayer:
        # wrap lora: for single layer
        lora_config = LoraConfig(
        r=lora_r, 
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[f'encoder.layers.{str(layer_index)}.attention.q_proj', 
                        f'encoder.layers.{str(layer_index)}.attention.k_proj', 
                        f'encoder.layers.{str(layer_index)}.attention.v_proj', 
                        f"encoder.layers.{str(layer_index)}.attention.out_proj", 
                        f"encoder.layers.{str(layer_index)}.feed_forward.intermediate_dense", 
                        f"encoder.layers.{str(layer_index)}.feed_forward.output_dense"]
        )
    else:
        # wrap lora: for every layer
        lora_config = LoraConfig(
        r=lora_r, 
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[f'encoder.layers.{str(l)}.attention.q_proj' for l in range(layer_index+1)] +  
                        [f'encoder.layers.{str(l)}.attention.k_proj' for l in range(layer_index+1)] + 
                        [f'encoder.layers.{str(l)}.attention.v_proj' for l in range(layer_index+1)] + 
                        [f'encoder.layers.{str(l)}.attention.out_proj' for l in range(layer_index+1)] + 
                        [f'encoder.layers.{str(l)}.feed_forward.intermediate_dense' for l in range(layer_index+1)] + 
                        [f"encoder.layers.{str(l)}.feed_forward.output_dense" for l in range(layer_index+1)]
        )

    modified_w2v2_lora = get_peft_model(init_custom_w2v2, lora_config)
    # Default: use normal classification head; weighted sum not used
    modified_w2v2_lora_cls = ModifiedWav2Vec2ModelforSER(modified_w2v2_lora, ws=ws, cls=init_cls_model).to(DEVICE)

    print("================== SET ALL PARAMS =====================")
    for name, param in modified_w2v2_lora_cls.named_parameters():
        if 'bottleneck' in name or 'weighted' in name or 'normal_classifier' in name:
            param.requires_grad = True
        print(f"{name}: {param.requires_grad}")

    return modified_w2v2_lora_cls


if __name__ == "__main__": 

    # create parser
    parser = argparse.ArgumentParser(description="Modify model settings")

    # Arguments
    parser.add_argument('-m', '--model', type=str, required=True, help='Which model to use: de, cn or en')   # won't use this: de script
    parser.add_argument('-t', '--train', type=str, required=True, help='Training datasets: de, cn or en')
    parser.add_argument('-e', '--eval', type=str, required=True, help='Evaluation datasets: de, cn or en')
    parser.add_argument('-T', '--test', type=str, required=True, help='Test datasets: de, cn or en')
    parser.add_argument('-l', '--layer', type=str, help='which layer to add LoRA. Give index')
    parser.add_argument('-E', '--everylayer', action="store_true", help='LoRA for every attention layer')
    parser.add_argument('-b', '--bottleneck', action="store_true", help='whether to add bottleneck')
    parser.add_argument('-g', '--weightedgate', action="store_true", help='whether to add weighted gate')
    parser.add_argument('-s', '--weightedsum', action="store_true", help='whether to add weighted sum') 
    parser.add_argument('-S', '--twostage', action="store_true", help='Change training data for the 2nd stage')
    parser.print_help()

    args = parser.parse_args()

    fold_layer_acc = []
    fold_emo_precision = {'Angry': [], 'Happy': [], 'Neutral': [], 'Sad': []}
    fold_emo_recall = {'Angry': [], 'Happy': [], 'Neutral': [], 'Sad': []}
    fold_emo_f1 = {'Angry': [], 'Happy': [], 'Neutral': [], 'Sad': []}

    for fold in FOLD_LABEL:

        print(f"\n======================= This is {fold} on {args.model} =======================")
        print(f"\nLoad dataset: ")
        # get layer representations in the dataloader
        train_output = load_dataset(args.model, "train", args.train, fold)
        eval_output = load_dataset(args.model, "eval", args.eval, fold)
        test_output = load_dataset(args.model, "test", args.test, fold)

        # get final customised model
        customed_lora_cls_model = get_custom_peft_model(args.model, int(args.layer), 
                                                        lora_r=8, lora_alpha=16, lora_dropout=0.1, 
                                                        bn=args.bottleneck, wg=args.weightedgate,
                                                        ws=args.weightedsum, everylayer=args.everylayer)

        # save cls model with lora
        torch.save(customed_lora_cls_model, f"./init_cls_model/0823_various_{args.model}_train_{args.train}_test_{args.test}_layer{args.layer}.pt")

        print("Set optimizer and criterion")
        optimizer = torch.optim.AdamW(customed_lora_cls_model.parameters(), lr=1e-4, eps=1e-8, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
            
        best_acc = 0.0
        num_epochs = 100
        best_performance = 0.0
        lowest_eval_loss = 1e10
        early_stop = False

        ### Training phase
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            # test if entering 2nd stage
            if (early_stop and args.twostage) or (args.twostage and epoch+1 == num_epochs // 2):
                if args.train == "cn":
                    print(f"\nEntering 2ND training phase: change training data from {args.train} to DE")
                    train_output = load_dataset(args.model, "train", "de", fold)
                    args.train = "cn-2nd"
                    stop_criteria = 10
                    lowest_eval_loss = 1e10
                    early_stop = False
                elif args.train == "de":
                    print(f"\nEntering 2ND training phase: change training data from {args.train} to CN")
                    train_output = load_dataset(args.model, "train", "cn", fold)
                    args.train = "de-2nd"
                    stop_criteria = 10
                    lowest_eval_loss = 1e10
                    early_stop = False
                # turn cn/de-2nd into cn/de
                elif "-2nd" in args.train:
                    print(f"\nTwo-stage training complete.")
                    args.train = args.train[:2]
                    break
                else:
                    raise ValueError("Only support transfer between CN and DE")

            # training phase
            print("\nTraining Phase:")
            customed_lora_cls_model.train() 
            train_loss = 0.0
            emolabels_train = []
            predictions_train = []

            ### Here we use cn dataset to train de model ###
            for batch in tqdm(train_output, desc="Training", leave=False, mininterval=10.0):
                inputs = batch['audios_list']
                labels = batch['labels']
                optimizer.zero_grad()
                outputs = customed_lora_cls_model(input_values=inputs)
                loss = criterion(outputs, labels.to(DEVICE))
                loss.backward()
                optimizer.step()
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
            print("\nEval Phase: ")
            customed_lora_cls_model.eval()
            eval_loss = 0.0
            emolabels_eval = []
            predictions_eval = []

            with torch.no_grad():
                for batch in tqdm(eval_output, desc="Evaluating", leave=False, mininterval=10.0):
                    inputs = batch['audios_list']
                    labels = batch['labels']
                    outputs = customed_lora_cls_model(input_values=inputs)
                    loss = criterion(outputs, labels.to(DEVICE))
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
                path_name = f"./trained_peft_model_fold/various_modules/various_{args.model}_{fold}_trained_{args.train}_tested_{args.test}_twostage_{args.twostage}_2ndfreeze_{args.freezelora}"
                os.makedirs(path_name, exist_ok=True)
                model_name = f"0823_various_layer_{args.layer}_epoch_{epoch+1}_acc_{best_acc:.4f}.pt"
                best_model_path = os.path.join(path_name, model_name)
                torch.save(customed_lora_cls_model, best_model_path)
                print(f"New best accuracy for layer {args.layer} on epoch {epoch+1}: {best_acc:.4f}. Model saved.")

            # check stop criteria
            if eval_loss <= lowest_eval_loss:
                lowest_eval_loss = eval_loss
                stop_criteria = 10
            else:
                stop_criteria -= 1
            if stop_criteria == 0:
                early_stop = True
                print(f"Validation loss does not decrease for 10 epochs. End training.")
                if not args.twostage:
                    break

        print(f"Model best accuracy on validation set: {best_acc:.4f}")

        # test phase
        print("\nTest Phase: ")
        customed_lora_cls_model_i = torch.load(best_model_path)
        customed_lora_cls_model_i.eval()
        test_loss = 0.0
        emolabels_test = []
        predictions_test = []

        with torch.no_grad():
            for batch in tqdm(test_output, desc="Testing", leave=False):
                inputs = batch['audios_list']
                labels = batch['labels']
                outputs = customed_lora_cls_model_i(input_values=inputs)
                loss = criterion(outputs, labels.to(DEVICE))
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
        ax.set_title(f"Confusion Matrix of {args.model} (various PEFT) predictions on {str(args.test).upper()} data - Layer {args.layer}")
        ax.xaxis.set_ticklabels(EMOTION_LABEL)
        ax.yaxis.set_ticklabels(EMOTION_LABEL)
        os.makedirs("./figs_fold/0823_variousPEFT_layer_test_cfm", exist_ok=True)
        fig_path = f"{args.model}_various_{fold}_trained_{args.train}_tested_{args.test}_layer_{args.layer}_twostage_{args.twostage}_2ndfreeze_{args.freezelora}.png"
        fig_savepath = os.path.join("./figs_fold/0823_variousPEFT_layer_test_cfm", fig_path)
        fig.savefig(fig_savepath)
    
    print(f"\n{args.model}, all folds layer accuracy: {fold_layer_acc}")
    print(f"{args.model}, all emo precision: {fold_emo_precision}")
    print(f"{args.model}, all emo recall: {fold_emo_recall}")
    print(f"{args.model}, all emo f1score: {fold_emo_f1}")
