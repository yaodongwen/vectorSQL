import random

def generate_random_vector(dim=384):
    """生成0-1之间的随机浮点数向量"""
    return [random.random() for _ in range(dim)]

# 生成并打印向量
vector = generate_random_vector()
print("向量长度:", len(vector))
print(vector)  
