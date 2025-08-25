# 数据集说明

### seq.jsonl
数据样例：
```json
[
    [480, 53961, null, {"112": 19, "117": 103, "118": 125, "119": 87, "120": 126, "100": 2, "101": 40, "102": 8559, "122": 5998, "114": 16, "116": 1, "121": 52176, "111": 5630}, 0, 1746077791], 
    [480, 50654, null, {"112": 15, "117": 285, "118": 737, "119": 1500, "120": 1071, "100": 6, "101": 22, "102": 10420, "122": 2269, "114": 16, "115": 43, "116": 13, "121": 12734, "111": 6737}, 0, 1746094091], 
    [480, 23149, null, {"112": 15, "117": 84, "118": 774, "119": 1668, "120": 348, "100": 6, "101": 32, "102": 6372, "122": 2980, "114": 16, "116": 15, "121": 30438, "111": 34195}, 0, 1746225104],
    ...
]
```
1. 每一行为一个用户的行为序列，按时间排序
2. 用户序列中每一个record的数据格式为：
```
[user_id, item_id, user_feature, item_feature, action_type, timestamp]
```
*注：
    i. 每一条record为记录user profile和item interaction其中的一个。若当前record为user profile，则item_id、item_feature、action_type为null；若当前record为item interaction，则user_feature为null。
    ii. user_feature/item_feature的值以dict组织，其中key表示feature_id，value表示对应的feature_value取值。
3. user_id, item_id, feature_value全部从1进行了重新编号（re-id），方便在模型中进行embedding lookup。原始值到re-id的映射参考下方的indexer.pkl文件。

### indexer.pkl
记录了原始wuid, creative_id和feature_value从1开始编号后的值（以下记为re-id）
```
indexer['u'] # 记录了wuid到re-id,key为wuid,value为re-id
indexer['i'] # 记录了creative_id到re-id,key为creative_id,value为re-id
indexer['f'] # 记录了各feature_id中feature_value到re-id，例如：⬇️
indexer['f']['112'] # 记录了feature_id 112中的feature_value到re-id的映射
```

### item_feat_dict.json
1. 记录了训练集中所有item的feature，方便训练时负采样。
2. key为item re-id，value为item的feature

### seq_offsets.pkl
记录了seq.jsonl中每一行的起始文件指针offset，方便读取数据时快速定位数据位置，提升I/O效率。


# Observations on Data
* `indexer.pkl` does not contain additional feature information. In particular, the "feature_value" are just indices, probably categorical index. From `indexer['f']`, we can see that there are 22 feature keys in total. 14 of them are for items, and 8 of them are for users. They are not overlapping. Items also have 6 additional multi-modal embedding features (see below).
* All the re-ids start from 1.
* According to `item_feat_dict.json`, there are 58734 items in the training set. However, there might be new items; see `__getitem__` function in `MyTestDataset` class in `dataset.py`.
* Not all items have all the feature keys, though most items have at least 13 keys. Items have at least 2 keys.
* Lengths of the sequences follow a "negative" exponential-like distribution from 101 to 11, with a mean of 90 and a median of 95.
* Not all sequences have interactions. The number of interactions range from 0 to 45, with a mean of 9 and median of 8.
* All sequences have exactly **one** record that contains user features. This record can be anywhere in the sequence, so at test time we may not know the user feature. # tianyi: in their model.py, how did they split the user feature and item iterations? # yuhang: when processing embeddings, they put the user token at the front of a seuqnce if its's available; the same goes for testing. When calculating loss for training, they ignore a token if the next item is a user token.
* Not all users have all the feature keys. The number of user feature keys range from 1 to 8, with a mean of 6 and median of 5.
* The `creative_emb` contains multi-modal feature values. It's too large to be included in the `seq.jsonl` file. Instead, it is added to the data in the `dataset.py` file through `fill_missing_feat` function when `__getitem__` is called (not during initialization). There are 6 embedding feature keys: '81'-'86'. Not all items have embeddings (e.g., out of 58734 items, for feature key '82' only 58006 items have feature values). See also `load_mm_emb` function in `dataset.py` for more details. `mm_emb_dict` attribute in the `MyDataset` class in `dataset.py` is a dictionary of dictionaries. For a fixed feature key (e.g., '82'), `mm_emb_dict['82']` is a dictionary that maps from item ID (the keys used here are `creative_id`, so the `indexer_i_rev` attribute in the `MyDataset` class is constructed from `indexer.pkl` to map from `re-id` to `creative_id`) to embedding values. 
* For test dataset, there is a seprate file called `predict_seq.jsonl` that we do not get to see. Not sure if the records there have `action_type` or `timestamp`; see `__getitem__` function in `MyTestDataset` class in `dataset.py`.
* In the baseline model, the action type and the timestamp up to the current record do not affect the embedding. That's why `__getitem__` in both `MyDataset` and `MyTestDataset` ignore those two fields.
* In the `dataset.py` file, they currently treat **the next item that the user is exposed to** as a positive example, not an item that the user will actually interacts with. 
