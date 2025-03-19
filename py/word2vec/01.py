import jieba
import jieba.analyse

text = "我去吃晚饭，我去学习，我看电视，然后我去吃晚饭"

seg_list = jieba.lcut(text, cut_all=False)
print("分词结果: " + "/".join(seg_list))


keywords = jieba.analyse.extract_tags(text, topK=5, withWeight=True)
# keywords = jieba.analyse.textrank(text, topK=5, withWeight=True)
print(keywords)
for word, weight in keywords:
    print(f"{word}: {weight}")

