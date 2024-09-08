import datasets
import torch
from torch import nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import Wav2Vec2Model
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderLayer


# torch dataset class
class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    def add_data(self, new_data, new_label):
        # new_data shape: [sample_length, 768]
        self.data = torch.cat((self.data, new_data))
        self.label = torch.cat((self.label, new_label))


# Classification head
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.dense1 = nn.Linear(768, 128)
        self.flat = nn.Flatten()
        self.dense = nn.Linear(128, 16)
        self.acti = nn.ReLU()
        self.out = nn.Linear(16, 4) #change number_of_emotion_class

    def forward(self, x):
        x = self.flat(x)
        x = self.dense1(x)
        x = self.acti(x)
        x = self.dense(x)
        res = self.acti(x)
        emotion = self.out(res)
        return emotion
    

class CustomWav2Vec2LoraClsModel(nn.Module):
    def __init__(self, wav2vec2_model, classification_head):
        super(CustomWav2Vec2LoraClsModel, self).__init__()
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wav2vec2 = wav2vec2_model
        self.classification_head = classification_head

    def forward(self, input_values):
        pooled_feat = []

        for input_v in input_values:
            input_v = input_v.unsqueeze(0).to(self.DEVICE).float()
            outputs = self.wav2vec2(input_values=input_v)           # don't need all hidden states
            last_hidden = outputs.last_hidden_state
            last_hidden_meanpool = torch.mean(last_hidden, dim=1)
            last_hidden_meanpool = last_hidden_meanpool.squeeze(0)
            pooled_feat.append(last_hidden_meanpool)
        
        # Batch the resulting hidden states
        batched_pooled_feat = torch.stack(pooled_feat)
        
        # Pass the batched hidden states to the classification head
        logits = self.classification_head(batched_pooled_feat)

        return logits


### New class for various PEFT modules ###
# utils for other ways of PEFT
class WeightGating(nn.Module):
    def __init__(self):
        super(WeightGating,self).__init__ ()
        hdim = 768
        self.gate = nn.Parameter(torch.ones(hdim), requires_grad=True)
        
    def forward(self, h):
        sigma_g = torch.sigmoid(self.gate)
        h = sigma_g * h
        return h
    
class WSNeuralNet(nn.Module): # weighted sum neuro net, input [12, 768]
    def __init__(self, hdim):
        super(WSNeuralNet, self).__init__()
        self.W = nn.Parameter(torch.ones(1, 12), requires_grad=True) # weighting parameters
        self.dense1 = nn.Linear(hdim, 128)
        self.dense = nn.Linear(128, 16)
        self.acti = nn.ReLU()
        number_of_emotion_class = 4
        self.out = nn.Linear(16, number_of_emotion_class) #change number_of_emotion_class

    def forward(self, x):
        x = torch.matmul(self.W, x.permute(1, 0, 2)) # shape [1, batch_size, 768]
        x = x.squeeze(1)
        x= torch.div(x, torch.sum(self.W, 1))
        x = self.dense1(x)
        x = self.acti(x)
        x = self.dense(x)
        res = self.acti(x)
        emotion = self.out(res)
        return emotion
    
class BottleneckAdapter(nn.Module):
    def __init__(self):
        super(BottleneckAdapter, self).__init__()
        self.down = nn.Linear(768, 48) # reduction factor = 16
        self.up = nn.Linear(48, 768)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.down(x)
        x = self.activation(x)
        x = self.up(x)
        return x


# Modify Wav2Vec2 Codebase
class ModifiedWav2Vec2EncoderLayer(Wav2Vec2EncoderLayer):
    def __init__(self, config, bn=None, wg=None):

        super().__init__(config)
        self.weighted_gate = wg
        self.bottleneck_adaptor = bn

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        if self.weighted_gate:
            hidden_states = self.weighted_gate(hidden_states)
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        if self.bottleneck_adaptor:
            ff_out = self.feed_forward(hidden_states)
            hidden_states = hidden_states + self.bottleneck_adaptor(ff_out) + ff_out
        else:
            hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class ModifiedWav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config, bn=False, wg=False, every_layer=True, best_layer=False):
        super().__init__(config)

        if not every_layer:
            if bn:
                self.bn = BottleneckAdapter()
            else:
                self.bn = bn
            if wg:
                self.wg = WeightGating()
            else:
                self.wg = wg

            enc_layer_list = [Wav2Vec2EncoderLayer(config) for _ in range(best_layer-1)]
            enc_layer_list.append(ModifiedWav2Vec2EncoderLayer(config, bn=self.bn, wg=self.wg))
            self.encoder.layers = nn.ModuleList(enc_layer_list)
        else:
            if bn and wg:
                self.encoder.layers = nn.ModuleList([
                    ModifiedWav2Vec2EncoderLayer(config, bn=BottleneckAdapter(), wg=WeightGating()) for _ in range(config.num_hidden_layers)])
            elif bn and not wg:
                self.encoder.layers = nn.ModuleList([
                    ModifiedWav2Vec2EncoderLayer(config, bn=BottleneckAdapter(), wg=False) for _ in range(config.num_hidden_layers)])
            elif not bn and wg:
                self.encoder.layers = nn.ModuleList([
                    ModifiedWav2Vec2EncoderLayer(config, bn=False, wg=WeightGating()) for _ in range(config.num_hidden_layers)])
            else:
                self.encoder.layers = nn.ModuleList([
                    ModifiedWav2Vec2EncoderLayer(config, bn=False, wg=False) for _ in range(config.num_hidden_layers)])


class ModifiedWav2Vec2ModelforSER(nn.Module):
    def __init__(self, modified_wav2vec2_model, ws=None, cls=None):
        super(ModifiedWav2Vec2ModelforSER, self).__init__()
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modified_wav2vec2 = modified_wav2vec2_model
        if ws:
            self.ws = True
            self.weighted_sum_classifier = ws
        elif cls:
            self.ws = False
            self.normal_classifier = cls     # normal cls if ws=False, WSNeuralNet if ws=True
        else:
            return ValueError("Don't know which classification head to initialise")

    def forward(self, input_values):
        pooled_feat = []

        for input_v in input_values:
            input_v = input_v.unsqueeze(0).to(self.DEVICE).float()
            outputs = self.modified_wav2vec2(input_values=input_v)
            if self.ws:
                hidden = outputs.hidden_states
                hidden = torch.stack(hidden)
                print(hidden.shape)
                hidden = hidden.squeeze()[1:]                   # [12, batch_size=1, num_frames, 768]
                hidden = torch.mean(hidden, dim=2).squeeze(1)   # [12, (batch_size=1), 768]
            else:
                hidden = outputs.last_hidden_state
                hidden = torch.mean(hidden, dim=1)
                hidden = hidden.squeeze(0)
            
            pooled_feat.append(hidden)
        
        # Batch the resulting hidden states
        batched_pooled_feat = torch.stack(pooled_feat)
        
        # Pass the batched hidden states to the classification head
        if self.ws:
            logits = self.weighted_sum_classifier(batched_pooled_feat)
        else:
            logits = self.normal_classifier(batched_pooled_feat)

        return logits


# change text labels into numbers for classifier
def map_labels(example):
    label_mapping = {'Angry': 0, 'Happy': 1, 'Neutral': 2, 'Sad': 3}
    example['label'] = label_mapping[example['label']]
    return example


def dataloader_for_cls(prev_dataloader, layer, batch_size):
    """
    Extract representations of certain layer, and then rebatch

    prev_dataloader should be:
    ['pooled_feature']: mean_pooled model output, [13, 768]
    ['label']: a label, [1]
    """
    for i, batch in enumerate(prev_dataloader):
        rep = batch['pooled_feature']
        label = batch['label']

        if i == 0:
            new_dataset = MyDataset(rep[layer].unsqueeze(0), label.unsqueeze(0))
        else: 
            new_dataset.add_data(rep[layer].unsqueeze(0), label.unsqueeze(0))

    new_dataset.label.squeeze(1)

    new_dataloader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

    return new_dataloader


def delete_error_entries(dataset_split, error_idx):
    data_list = dataset_split.to_list()
    del data_list[error_idx]
    updated_dataset = datasets.Dataset.from_list(data_list)

    return updated_dataset


def get_acc_by_emo_sk(predicted_emo_list, gs_emo_list, certain_emo):

    gs_binary = [1 if emo == certain_emo else 0 for emo in gs_emo_list]
    predicted_binary = [1 if emo == certain_emo else 0 for emo in predicted_emo_list]

    if not gs_binary:
        print(f"The model does not predict {certain_emo}. Returning 0 for all metrics.")
        return 0.0, 0.0, 0.0

    precision, recall, f1score, _ = precision_recall_fscore_support(gs_binary, predicted_binary, average='binary')

    return precision, recall, f1score