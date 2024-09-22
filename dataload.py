import pandas as pd

base_path = 'dataset'
train_path = f'{base_path}/train.csv'
test_path = f'{base_path}/test.csv'
mapping_path = f'{base_path}/misconception_mapping.csv'
sub_path = f'{base_path}/sample_submission.csv'

# 读取 train.csv 和 misconception_mapping.csv 文件
train_df = pd.read_csv(train_path)
mapping_df = pd.read_csv(mapping_path)
print(train_df.loc[0]['ConstructName'])

# 构建从 MisconceptionId 到 MisconceptionName 的映射表
misconception_map = dict(zip(mapping_df['MisconceptionId'], mapping_df['MisconceptionName']))

# 初始化一个空列表来存储结果
result = []

# 遍历 train.csv 中的每一行
for index, row in train_df.iterrows():
    # 对于每个答案选项
    for answer in ['A', 'B', 'C', 'D']:
        # 获取其它信息跟答案文本和对应的误解标识
        QuestionId = row['QuestionId']
        ConstructId	= row['ConstructId']
        ConstructName = row['ConstructName']
        SubjectId = row['SubjectId']
        SubjectName = row['SubjectName']
        CorrectAnswer = row['CorrectAnswer']
        QuestionText = row['QuestionText']
        answer_text = row[f'Answer{answer}Text']
        misconception_id = row[f'Misconception{answer}Id']
        
        # 如果误解标识存在，则查找对应的误解名称
        if not pd.isna(misconception_id):
            misconception_name = misconception_map[misconception_id]
            result.append([QuestionId,ConstructId,ConstructName,SubjectId,SubjectName,CorrectAnswer,QuestionText,answer_text, misconception_name])
            # result.append([index + 1, f'Answer{answer}Text', answer_text, misconception_name])

# 创建 DataFrame 并选择所需的列
result_df = pd.DataFrame(result, columns=['QuestionId','ConstructId','ConstructName','SubjectId','SubjectName','CorrectAnswer','QuestionText','AnswerText','MisconceptionName'])

# 保存结果到新的 CSV 文件
result_df.to_csv('mapped_answers.csv', index=True)