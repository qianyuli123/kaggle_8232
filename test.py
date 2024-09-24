# test.py
import pandas as pd
import torch
from encode import Encoder
from dataload import DataLoader

def run_test(model_path='model.pth'):
    # 读取数据
    train_df, mapping_df = DataLoader('dataset').get_result_df(), DataLoader('dataset').get_mapping_df()
    
    # 初始化 Encoder
    encoder = Encoder(train_df, mapping_df, device='cuda')
    
    # 加载模型
    encoder.load_model(model_path)
    
    # 获取测试数据
    test_df = DataLoader('dataset').get_result_test_df()
    print(test_df.keys())

    # 构建结果字典
    results = {}

    for x in range(len(test_df)):
        row1 = test_df.iloc[x]
        feature_vector = encoder.encode_features(row1)

        similarity = []
        for i in range(len(mapping_df)):
            row = mapping_df.iloc[i]
            label_vector = encoder.encode_labels(row)
            loss = encoder.similarity_loss(feature_vector, label_vector)
            similarity.append((loss.item(), mapping_df.iloc[i]['MisconceptionId'], mapping_df.iloc[i]['MisconceptionName']))

        similarity.sort(reverse=False)
        # 提取前 25 个相似度最高的结果
        top_25 = [str(item[1]) for item in similarity[:25]]
        
        # 构建结果字典
        key = f"{row1['QuestionId']}_{row1['Answer']}"
        results[key] = ' '.join(top_25)
    
    # 将结果保存为 CSV 文件
    df_results = pd.DataFrame(list(results.items()), columns=['QuestionId_Answer', 'MisconceptionId'])
    df_results.to_csv('results.csv', index=False)

if __name__ == '__main__':
    run_test()