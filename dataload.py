import pandas as pd

class DataLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.train_path = f'{base_path}/train.csv'
        self.test_path = f'{base_path}/test.csv'
        self.mapping_path = f'{base_path}/misconception_mapping.csv'
        self.sub_path = f'{base_path}/sample_submission.csv'
        
        # 读取 train.csv test.csv和 misconception_mapping.csv 文件
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        self.mapping_df = pd.read_csv(self.mapping_path)
        
        # 构建从 MisconceptionId 到 MisconceptionName 的映射表
        self.misconception_map = dict(zip(self.mapping_df['MisconceptionId'], self.mapping_df['MisconceptionName']))
        
        # 初始化一个空列表来存储train结果
        self.result = []
        
        # 创建一个空列表来存储test结果
        self.result_test = []
        
        # 处理训练数据
        self._process_train_data()

        # 处理测试数据
        self._process_test_data()

    def get_train_df(self):
        return self.train_df
    
    def get_mapping_df(self):
        return self.mapping_df
    
    def get_test_df(self):
        return self.test_df
    
    def _process_train_data(self):
        # 遍历 train.csv 中的每一行
        for index, row in self.train_df.iterrows():
            # 对于每个答案选项
            for answer in ['A', 'B', 'C', 'D']:
                # 获取其它信息跟答案文本和对应的误解标识
                QuestionId = row['QuestionId']
                ConstructId = row['ConstructId']
                ConstructName = row['ConstructName']
                SubjectId = row['SubjectId']
                SubjectName = row['SubjectName']
                CorrectAnswer = row['CorrectAnswer']
                QuestionText = row['QuestionText']
                Answer = answer
                answer_text = row[f'Answer{answer}Text']
                misconception_id = row[f'Misconception{answer}Id']
                
                # 如果误解标识存在，则查找对应的误解名称
                if not pd.isna(misconception_id):
                    misconception_name = self.misconception_map[misconception_id]
                    self.result.append([QuestionId, ConstructId, ConstructName, SubjectId, SubjectName,
                                        CorrectAnswer, QuestionText, Answer, answer_text, misconception_id])

    def _process_test_data(self):
        # 遍历 test.csv 中的每一行
        for index, row in self.test_df.iterrows():
            # 对于每个答案选项
            for answer in ['A', 'B', 'C', 'D']:
                # 获取其它信息跟答案文本
                QuestionId = row['QuestionId']
                ConstructId = row['ConstructId']
                ConstructName = row['ConstructName']
                SubjectId = row['SubjectId']
                SubjectName = row['SubjectName']
                CorrectAnswer = row['CorrectAnswer']
                QuestionText = row['QuestionText']
                Answer = answer
                answer_text = row[f'Answer{answer}Text']
                
                self.result_test.append([QuestionId, ConstructId, ConstructName, SubjectId, SubjectName,
                                        CorrectAnswer, QuestionText, Answer, answer_text])

    def get_result_df(self):
        # 创建 DataFrame 并选择所需的列
        result_df = pd.DataFrame(self.result, columns=['QuestionId', 'ConstructId', 'ConstructName',
                                                       'SubjectId', 'SubjectName', 'CorrectAnswer',
                                                       'QuestionText', 'Answer', 'AnswerText', 'MisconceptionId'])
        return result_df

    def get_result_test_df(self):
        # 创建 DataFrame 并选择所需的列
        result_test_df = pd.DataFrame(self.result_test, columns=['QuestionId', 'ConstructId', 'ConstructName',
                                                       'SubjectId', 'SubjectName', 'CorrectAnswer',
                                                       'QuestionText', 'Answer', 'AnswerText'])
        return result_test_df
    def save_result(self, output_path='mapped_answers.csv'):
        result_df = self.get_result_df()
        result_df.to_csv(output_path, index=False)

# 使用示例
# base_path = 'dataset'
# data_loader = DataLoader(base_path)

# 获取处理后的结果 DataFrame
# result_df = data_loader.get_result_df()
# print(result_df.head())

# 保存结果到新的 CSV 文件
# data_loader.save_result('mapped_answers.csv')