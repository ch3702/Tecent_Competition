from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import save_emb


class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'): # PyTorch 2.0+ has a built-in Flash Attention
            # PyTorch 2.0+ 使用内置的Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            )
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)

        return output, None


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs


class BaselineModel(torch.nn.Module):
    """
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息, key为特征ID, value为特征数量
        feat_types: 各个特征的特征类型, key为特征类型名称, value为包含的特征ID列表, 包括user和item的sparse, array, emb, continual类型
        args: 全局参数

    Attributes:
        user_num: 用户数量
        item_num: 物品数量
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first # whether to apply layer norm before or after the attention and feed-forward layers
        self.maxlen = args.maxlen
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        
        # padding_idx=0 means that the index 0 always maps to a zero vector. Probably for the padding token and/or missing values.
        # each item we've seen has an embedding
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        # each user we've seen has an embedding
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        # each position in the sequence has an embedding
        # only the first maxlen+1 positions are used, so actually this over-initializes the embedding table
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0) 
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        # for all sparse features, create an embedding table for all possible values of that feature
        self.sparse_emb = torch.nn.ModuleDict() # a dictionary of embedding tables for each sparse feature
        # for all embedding features, create a linear transformation from the embedding dimension to the hidden dimension
        self.emb_transform = torch.nn.ModuleDict() # a dictionary of linear transformations for each embedding feature
        
        # initialized to hold the layers for the transformer blocks
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types) # initialize info such as which features are in USER_SPARSE_FEAT, etc.
        
        # each sparse feature and each array feature has an embedding, so that gives us a total of 
        # hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) dimensions
        # the continuous features are numerical values, so they don't have an embedding table and are directly concatenated to the user features
        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        # each sparse feature and each array feature has an embedding, so that gives us a total of 
        # hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT)) dimensions
        # the continuous features are numerical values, so they don't have an embedding table and are directly concatenated to the item features
        # each (multi-modal) embedding feature has an (model) embedding, so that gives us a total of 
        # hidden_units * len(self.ITEM_EMB_FEAT) dimensions
        itemdim = (
            args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT))
            + len(self.ITEM_CONTINUAL_FEAT)
            + args.hidden_units * len(self.ITEM_EMB_FEAT)
        )
        
        # these two big linear layers are used to project the processed user and item features to the hidden dimension
        # this is done in the feat2emb function
        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)
        
        # the last layer norm is used to normalize the output of the transformer blocks to obtain final normalized feature representation
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # each block consists of: an attention layer norm, an attention layer, a forward layer norm, and a forward layer
        # the blocks are stacked sequentially with residual connections (see log2feats functions)
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )  # 优化：用FlashAttention替代标准Attention
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
        
        # For all sparse features, create an embedding table. This includes array features.
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        # For all embedding features, create a linear transformation from the embedding dimension to the hidden dimension.
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)

    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息 (特征数量) 按特征类型分组产生不同的字典, 方便声明稀疏特征的Embedding Table

        Args:
            feat_statistics: 特征统计信息, key为特征ID, value为特征数量
            feat_types: 各个特征的特征类型, key为特征类型名称, value为包含的特征ID列表, 包括user和item的sparse, array, emb, continual类型
        """
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度

    def feat2tensor(self, seq_feature, k):
        """
        Given the feature key k, extract k from all timesteps across all sequences in seq_feature.
        If k is a sparse or continual feature, return a tensor of shape [batch_size, maxlen].
        If k is an array feature, return a tensor of shape [batch_size, maxlen, max_array_len].

        Args:
            seq_feature: 序列特征list, 每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]. 
            k: 特征ID. This feature does not concern an embedding feature, i.e., it can only be sparse, array, or continual.

            seq_feature is a list of sequences, where each sequence is a list of feature dictionaries like this:
            seq_feature = [
                # Sequence 1 (batch item 0)
                [
                    {"feature_1": value, "feature_2": value, ...},  # Timestep 0
                    {"feature_1": value, "feature_2": value, ...},  # Timestep 1
                    {"feature_1": value, "feature_2": value, ...},  # Timestep 2
                    ...
                ],
                
                # Sequence 2 (batch item 1) 
                [
                    {"feature_1": value, "feature_2": value, ...},  # Timestep 0
                    {"feature_1": value, "feature_2": value, ...},  # Timestep 1
                    ...
                ],
                
                # ... more sequences
            ]

        Returns:
            batch_data: 特征值的tensor, 形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)

        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
            max_array_len = 0
            max_seq_len = 0

            # find the max length of sequences in the batch, as well as the max length of the arrays in the sequences
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))
            
            # batch_data collects all the values of k across all sequences in the batch, 
            # with each value being an array with possible padding for the shorter arrays
            # as well as padding for the shorter sequences
            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]

            return torch.from_numpy(batch_data).to(self.dev)
        else:
            # for sparse and continual features, extract the integer or float values, respectively, with possible padding for the shorter sequences
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data

            return torch.from_numpy(batch_data).to(self.dev)

    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        """
        Args:
            seq: 序列ID, a tensor of IDs with shape [batch_size, maxlen], where each entry is either an item ID, a user ID, or 0 (padding). 
            It looks like this:
            seq = torch.tensor([
                # Batch item 0: 
                [item_123, user_456, item_789, 0, 0],
                
                # Batch item 1:   
                [item_234, item_567, user_789, item_123, 0],
                
                # Batch item 2: 
                [item_345, 0, 0, 0, 0]
            ])
            feature_array: 特征list, 每个元素为当前时刻的特征字典. 
            feature_array has the same structure as seq_feature in the feat2tensor function, with shape [batch_size, maxlen] (so already padded).
            mask: 掩码, 1表示item, 2表示user. A tensor of 1 or 2 of shape [batch_size, maxlen]. Not used if include_user is False.
            include_user: 是否处理用户特征, 在两种情况下不打开: 1) 训练时在转换正负样本的特征时 (因为正负样本都是item); 2) 生成候选库item embedding时。Boolean.

        Returns:
            seqs_emb: 序列特征的Embedding, shape [batch_size, maxlen, hidden_units], 
            where hidden_units is because both userdnn and itemdnn are linear layers mapping from userdim and itemdim to hidden_units
        """
        seq = seq.to(self.dev)
        # pre-compute embedding
        # item_embedding and user_embedding have shape [batch_size, maxlen, hidden_units]
        # item_feat_list and user_feat_list consist of a list of tensors, each with shape [batch_size, maxlen, hidden_units]
        if include_user:
            user_mask = (mask == 2).to(self.dev) # an array of booleans
            item_mask = (mask == 1).to(self.dev) # an array of booleans
            user_embedding = self.user_emb(user_mask * seq) # grab the relevant rows of the user embedding table
            item_embedding = self.item_emb(item_mask * seq) # grab the relevant rows of the item embedding table
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq) # if include_user is False, all records of the sequence are items, so can look up directly
            item_feat_list = [item_embedding]

        # batch-process all feature types except embedding features
        # for each feature type, we call the feat2tensor function to extract the corresponding feature values for all records in all sequences
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]

        if include_user:
            all_feat_types.extend(
                [
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                    (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
                ]
            )

        # batch-process each feature type using the feat2tensor function
        # Note that, all entries in the all_feat_types are initialized with item_feat_list, so because of the same reference pointer.
        # As we go through the feat_types below and append to the feat_list in all the entries, we are actually appending to the same list.
        # As a result, at the end of the following loop, item_feat_list becomes a big list of tensors, each with shape [batch_size, maxlen, hidden_units].
        # This does not seem to be a good practice...
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                tensor_feature = self.feat2tensor(feature_array, k)

                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    # sum across the embeddings of the array features, so multiple values for an array feature are aggregated
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2)) 
                elif feat_type.endswith('continual'):
                    # for continuous features, we don't need to sum, just add a dimension
                    feat_list.append(tensor_feature.unsqueeze(2))

        for k in self.ITEM_EMB_FEAT:
            # collect all data to numpy, then batch-convert
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0]) # maxlen, already padded so the length of the first sequence is representative

            # batch_emb_data collects all the embeddings for feature k across all records in all sequences in the batch
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)
            for i, seq in enumerate(feature_array):
                for j, item in enumerate(seq):
                    if k in item:
                        batch_emb_data[i, j] = item[k]

            # batch-convert and transfer to GPU
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))

        # merge features
        # The concatenation is done across the hidden_units dimension, so all_item_emb will have shape [batch_size, maxlen, itemdim], 
        # where itemdim is defined in the __init__ function.
        # The same is true for all_user_emb.
        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = torch.relu(self.itemdnn(all_item_emb))
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = torch.relu(self.userdnn(all_user_emb))
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature):
        """
        Args:
            log_seqs: 序列ID, shape [batch_size, maxlen], where each entry is either an item ID or a user ID.
            mask: token类型掩码, 1表示item token, 2表示user token. A tensor of 1 or 2 of shape [batch_size, maxlen].
            seq_feature: 序列特征list, 每个元素为当前时刻的特征字典. 
            seq_feature has the same structure as feature_array in the feat2emb function, with shape [batch_size, maxlen] (so already padded).

        Returns:
            seqs_emb: 序列的Embedding, 形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        # below, seq is actually seqs_emb, the output of the feat2emb function, with shape [batch_size, maxlen, hidden_units]
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        seqs *= self.item_emb.embedding_dim**0.5 # scale the embeddings by the square root of the embedding dimension
        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone() 
        poss *= log_seqs != 0 # mask out the padding positions to obtain the position embeddings
        seqs += self.pos_emb(poss) # add the position embeddings to the sequence embeddings
        seqs = self.emb_dropout(seqs) # apply dropout to the sequence embeddings

        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix) # lower triangular matrix with ones on and below the diagonal, causal mask
        attention_mask_pad = (mask != 0).to(self.dev) # mask out the padding positions
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1) # combine causal mask and padding mask

        for i in range(len(self.attention_layers)): # same as num_blocks in the __init__ function
            if self.norm_first: # apply layer norm before the attention and feed-forward layers
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else: # apply layer norm after the attention and feed-forward layers
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        # apply layer norm to the output of the transformer blocks to obtain final normalized feature representation,
        # shape [batch_size, maxlen, hidden_units]
        log_feats = self.last_layernorm(seqs)

        # return the output of the transformer blocks
        return log_feats

    def forward(
        self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature
    ):
        """
        训练时调用, 计算正负样本的logits

        Args:
            user_item: 用户序列ID, shape [batch_size, maxlen], where each entry is either an item ID or a user ID.
            pos_seqs: 正样本序列ID, shape [batch_size, maxlen], where each entry is an item ID.
            neg_seqs: 负样本序列ID, shape [batch_size, maxlen], where each entry is an item ID.
            mask: token类型掩码, 1表示item token, 2表示user token. A tensor of 1 or 2 of shape [batch_size, maxlen].
            next_mask: 下一个token类型掩码, 1表示item token, 2表示user token. A tensor of 1 or 2 of shape [batch_size, maxlen].
            next_action_type: 下一个token动作类型, 0表示曝光, 1表示点击. A tensor of 0 or 1 of shape [batch_size, maxlen].
            seq_feature: 序列特征list, 每个元素为当前时刻的特征字典
            pos_feature: 正样本特征list, 每个元素为当前时刻的特征字典
            neg_feature: 负样本特征list, 每个元素为当前时刻的特征字典

        Returns:
            pos_logits: 正样本logits, 形状为 [batch_size, maxlen]
            neg_logits: 负样本logits, 形状为 [batch_size, maxlen]
        """
        log_feats = self.log2feats(user_item, mask, seq_feature) # shape [batch_size, maxlen, hidden_units]
        loss_mask = (next_mask == 1).to(self.dev) # don't compute loss if the next token is a user token, shape [batch_size, maxlen]

        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False) # shape [batch_size, maxlen, hidden_units]
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False) # shape [batch_size, maxlen, hidden_units]

        pos_logits = (log_feats * pos_embs).sum(dim=-1) # dot product of the sequence embeddings and the positive item embeddings, shape [batch_size, maxlen]
        neg_logits = (log_feats * neg_embs).sum(dim=-1) # dot product of the sequence embeddings and the negative item embeddings, shape [batch_size, maxlen]
        pos_logits = pos_logits * loss_mask # don't compute logits if the next token is a user token, shape [batch_size, maxlen]
        neg_logits = neg_logits * loss_mask # don't compute logits if the next token is a user token, shape [batch_size, maxlen]

        return pos_logits, neg_logits

    def predict(self, log_seqs, seq_feature, mask):
        """
        计算用户序列的表征
        Args:
            log_seqs: 用户序列ID, shape [batch_size, maxlen], where each entry is either an item ID or a user ID.
            seq_feature: 序列特征list, 每个元素为当前时刻的特征字典, shape [batch_size, maxlen].
            mask: token类型掩码, 1表示item token, 2表示user token. A tensor of 1 or 2 of shape [batch_size, maxlen].

        Returns:
            final_feat: 用户序列的表征, 形状为 [batch_size, hidden_units], only contain the representation of the last token in the sequence
        """
        log_feats = self.log2feats(log_seqs, mask, seq_feature)

        final_feat = log_feats[:, -1, :]

        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding, 用于检索

        Args:
            item_ids: 候选item ID (re-id形式)
            retrieval_ids: 候选item ID (检索ID, 从0开始编号, 检索脚本使用)
            feat_dict: 训练集所有item特征字典, key为特征ID, value为特征值
            save_path: 保存路径
            batch_size: 批次大小
        """
        all_embs = []

        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids)) # batch window, from start_idx to end_idx - 1
            
            # batch_feat is a list of dictionaries, each containing the feature values for a single item
            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])

            batch_feat = np.array(batch_feat, dtype=object)
            
            # embed the batch of items using the current model. maxlen = 1 because we are only embedding a single item at a time ([batch_feat]).
            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0) # shape [batch_size, hidden_units]

            # append the embeddings to the list
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1) # shape [num_items, 1]
        final_embs = np.concatenate(all_embs, axis=0) # shape [num_items, hidden_units]
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))
