from encode import Encoder
from dataload import DataLoader

# 测试用例
if __name__ == '__main__':
    # 示例数据

    '''data = {
        'ConstructName': ['Construct1', 'Construct2'],
        'SubjectName': ['Subject1', 'Subject2'],
        'QuestionText': ['What is the capital of France?', 'What is the capital of Germany?'],
        'AnswerAText': ['Paris', 'Berlin'],
        'MisconceptionId': [1, 2]
    }
    mapping_data = {
        'MisconceptionId': [1, 2],
        'MisconceptionName': ['Capital of France', 'Capital of Germany']
    }
    
    train_df = pd.DataFrame(data)
    mapping_df = pd.DataFrame(mapping_data)'''

    train_df, mapping_df = DataLoader('dataset').get_result_df(), DataLoader('dataset').get_mapping_df()
    
    encoder = Encoder(train_df, mapping_df, device='cuda')
    
    # 训练模型
    encoder.train(epochs=10, learning_rate=0.001)
    
    # 保存模型
    encoder.save_model('model.pth')
    