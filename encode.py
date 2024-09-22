import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data = """
,QuestionId,ConstructId,ConstructName,SubjectId,SubjectName,CorrectAnswer,QuestionText,AnswerText,MisconceptionName
0,0,856,Use the order of operations to carry out calculations involving powers,33,BIDMAS,A,"\[
3 \times 2+4-5
\]
Where do the brackets need to go to make the answer equal \( 13 \) ?",Does not need brackets,"Confuses the order of operations, believes addition comes before multiplication "
1,1,1612,Simplify an algebraic fraction by factorising the numerator,1077,Simplifying Algebraic Fractions,D,"Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \)",\( m+1 \),"Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term"
"""

# 将字符串转换为 DataFrame
string_data = StringIO(data)
df = pd.read_csv(string_data, skipinitialspace=True)

# 提取 QuestionText 和 MisconceptionName
question_texts = df['QuestionText'].tolist()
misconception_names = df['MisconceptionName'].tolist()


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(question_texts)




# 计算余弦相似度
cosine_similarities = cosine_similarity(tfidf_matrix)

# 定义一个函数来找到最相似的 MisconceptionName
def predict_misconception(question_text, cosine_similarities, question_texts, misconception_names):
    # 将新问题向量化
    new_question_tfidf = vectorizer.transform([question_text])
    
    # 计算新问题与其他问题的相似度
    new_cosine_similarities = cosine_similarity(new_question_tfidf, tfidf_matrix)[0]
    
    # 找到最相似的问题索引
    most_similar_index = new_cosine_similarities.argmax()
    
    # 返回最相似的问题的 MisconceptionName
    return misconception_names[most_similar_index]

# 示例预测
new_question = "What is the value of \( 2 + 3 \times 4 \)?"
predicted_misconception = predict_misconception(new_question, cosine_similarities, question_texts, misconception_names)
print("Predicted Misconception:", predicted_misconception)