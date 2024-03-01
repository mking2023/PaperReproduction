# @Time   : 2022/2/13
# @Author : Hui Yu
# @Email  : ishyu@outlook.com
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ptsr.datasets import *
from ptsr.utils import *


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.emb_dim % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.emb_dim, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.emb_dim / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.emb_dim, self.all_head_size)
        self.key = nn.Linear(args.emb_dim, self.all_head_size)
        self.value = nn.Linear(args.emb_dim, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(args.emb_dim, args.emb_dim)
        self.LayerNorm = LayerNorm(args.emb_dim, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FilterLayer(nn.Module):
    def __init__(self, args):
        super(FilterLayer, self).__init__()
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.complex_weight = nn.Parameter(
            torch.randn(1, args.max_len // 2 + 1, args.emb_dim, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.emb_dim, eps=1e-12)

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        # sequence_emb_fft = torch.rfft(input_tensor, 2, onesided=False)  # [:, :, :, 0]
        # sequence_emb_fft = torch.fft(sequence_emb_fft.transpose(1, 2), 2)[:, :, :, 0].transpose(1, 2)
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.emb_dim, args.emb_dim * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(4 * args.emb_dim, args.emb_dim)
        self.LayerNorm = LayerNorm(args.emb_dim, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.no_filters = args.no_filters
        if self.no_filters:
            self.attention = SelfAttention(args)
        else:
            self.filterlayer = FilterLayer(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        if self.no_filters:
            hidden_states = self.attention(hidden_states, attention_mask)
        else:
            hidden_states = self.filterlayer(hidden_states)

        intermediate_output = self.intermediate(hidden_states)
        return intermediate_output


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class FMLPRecModel(nn.Module):
    
    @staticmethod
    def parse_args(parser):
        parser.add_argument('--hidden_dropout_prob', type=float, default=0.5)
        parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.5)
        parser.add_argument('--emb_dim', type=int, default=64)
        parser.add_argument('--cuda_condition', type=bool, default=True)
        parser.add_argument('--initializer_range', type=float, default=0.02)
        parser.add_argument('--num_attention_heads', type=float, default=2)
        parser.add_argument('--num_hidden_layers', type=int, default=2)
        parser.add_argument('--hidden_act', type=str, default='gelu')
        parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
        parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")
        parser.add_argument("--no_filters", action="store_true", help="if no filters, filter layers transform to self-attention")
        
    
    def __init__(self, args):
        super(FMLPRecModel, self).__init__()
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_num + 1, args.emb_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_len, args.emb_dim)
        self.LayerNorm = LayerNorm(args.emb_dim, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = Encoder(args)

        self.apply(self.init_weights)


    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb


    # same as SASRec
    def forward(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        sequence_output = item_encoded_layers[-1]

        return sequence_output


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    
    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.item_embeddings(pos_ids).squeeze()  # (B, E)
        neg_emb = self.item_embeddings(neg_ids).squeeze()  # (B, E) 

        # [batch hidden_size]
        # pos = pos_emb.view(-1, pos_emb.size(2))
        # neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out[:, -1, :]  # [batch*seq_len hidden_size]
        
        pos_logits = torch.sum(pos_emb * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg_emb * seq_emb, -1)
        # istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.mean(- torch.log(torch.sigmoid(pos_logits) + 1e-24) - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24))  # / torch.sum(istarget)
        return loss
    
    
    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits
    
    
    def train_model(self, dataset):
        user_train, user_valid, user_test, user_num, item_num = dataset
        train_data = SequentialDataset(self.args, user_train, padding_mode='left')
        valid_data = EvalDataset2(self.args, user_train, user_valid, user_test, mode='valid', padding_mode='left')
        test_data = EvalDataset2(self.args, user_train, user_valid, user_test, mode='test', padding_mode='left')
        
        train_loader = DataLoader(train_data, self.args.batchsize, shuffle=False, pin_memory=True, num_workers=4)
        valid_loader = DataLoader(valid_data, self.args.batchsize_eval, pin_memory=True)
        test_loader = DataLoader(test_data, self.args.batchsize_eval, pin_memory=True) 
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(self.args.adam_beta1, self.args.adam_beta2))
                
        best_valid_ndcg = -1.
        test_result = None
        for epoch in range(1, self.args.epoch + 1):
            epoch_loss = self.train_an_epoch(train_loader, optimizer)
            out_str = "Epoch [{}] Loss = {:.5f} lr = {:.5f}".format(epoch, epoch_loss, optimizer.param_groups[0]['lr'])
            print(out_str)
            logging.info(out_str)
            if self.args.use_wandb:
                wandb.log({"Loss": epoch_loss})
            
            if epoch % self.args.eval_interval == 0:
                valid_ndcg, valid_hr = self.evaluate_model(valid_loader, mode='Valid')
                test_ndcg, test_hr = self.evaluate_model(test_loader, mode='Test')
                
                if valid_ndcg > best_valid_ndcg:
                    best_valid_ndcg = valid_ndcg
                    test_result = (test_ndcg, test_hr)
                    torch.save(self.state_dict(), os.path.join(self.args.log_dir, f'best_model_epoch_{epoch}.pth'))
                print("\tVali NDCG10 = {:.5f}, Vali HR10 = {:.5f}".format(valid_ndcg, valid_hr))
                print("\tTest NDCG10 = {:.5f}, Test HR10 = {:.5f}".format(test_ndcg, test_hr))
                logging.info("Vali NDCG10 = {:.5f},\tVali HR10 = {:.5f}".format(valid_ndcg, valid_hr))
                logging.info("Test NDCG10 = {:.5f},\tTest HR10 = {:.5f}".format(test_ndcg, test_hr))
        
        print("Test NDCG10 = {:.5f}, Test HR10 = {:.5f}".format(test_result[0], test_result[1]))
        logging.info("Test NDCG10 = {:.5f}, Test HR10 = {:.5f}".format(test_result[0], test_result[1]))
        if self.args.use_wandb:
            wandb.log({"Test NDCG10": test_result[0], "Test HR10": test_result[1]})
        
        
    def train_an_epoch(self, train_loader, optimizer):
        epoch_loss = []
        self.train()
        for seq, pos, neg in add_process_bar(train_loader, desc='Train'):
            optimizer.zero_grad()
            seq, pos, neg = seq.cuda(), pos.cuda(), neg.cuda()
            sequence_output = self.forward(seq)
            loss = self.cross_entropy(sequence_output, pos, neg)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        return np.mean(epoch_loss)
    
    
    def evaluate_model(self, dataloader, mode='Valid'):
        with torch.no_grad():
            self.eval()
            
            NDCG10, HR10 = 0., 0.
            sample_num = 0
            for seq, pos, neg in add_process_bar(dataloader, desc=mode):
                seq, pos, neg = seq.cuda(), pos.cuda(), neg.cuda()
                sequence_output = self.forward(seq)
                sequence_output = sequence_output[:, -1, :]
                test_neg_items = torch.cat((pos, neg), dim=-1)
                scores = self.predict_sample(sequence_output, test_neg_items)
                NDCG10 += cal_ndcg_accum(scores, k=10)
                HR10 += cal_hr_accum(scores, k=10)
                sample_num += seq.shape[0]
            
            NDCG10 /= sample_num
            HR10 /= sample_num
            return NDCG10, HR10

