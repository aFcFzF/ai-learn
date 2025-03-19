import jieba 

words = []

with open('my.txt', 'r', encoding="utf-8") as file:
    while True:
        line = file.readline()
        if not line:
            break

        split_words = jieba.lcut(line)
        words.append(split_words)

print(words)
