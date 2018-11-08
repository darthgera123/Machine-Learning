import math
import matplotlib.pyplot as plt 
import data

# Importing Data
db = data.data
print(db)

""" Now in order to recommend a book to someone, we need to find similarity between 2 people
In order to do that, we have 2 ways - Euclidean method and Pearson method """

""" In the first way, what we do is find the euclidean distance between the points of similarity """
x = []
y = []
z = []
for name in db:
    try:
        x.append(db[name]['Programming language theory'])
        y.append(db[name]['Systems programming'])
        z.append(name)
    except KeyError:
        pass
""" for i,types in enumerate(z):
    x1 = x[i]
    y1 = y[i]
    plt.scatter(y1,x1,marker='x',color='red')
    plt.text(y1,x1,types,fontsize=9)

plt.ylabel('Programming language theory')
plt.xlabel('Systems programming')
plt.show()
 """
def euclidean(p1,p2):
    common_ranked = [itm for itm in db[p1] if itm in db[p2]]
    ranked = [(db[p1][course],db[p2][course]) for course in common_ranked]
    score = [pow(rank[0]-rank[1],2) for rank in ranked]
    return (1/(1+math.sqrt(sum(score))))

similarity = euclidean('Michael Stonebraker', 'Alan Perlis')
print(similarity)
""" Doesnt work best because it can be skewed with the ratings and doesnt really form a correlation """

def pcc(p1,p2):
    common_ranked = common_ranked = [itm for itm in db[p1] if itm in db[p2]]
    n = len(common_ranked)
    s1 = sum([db[p1][item] for item in common_ranked])
    s2 = sum([db[p2][item] for item in common_ranked])
    ss1 = sum([pow(db[p1][item],2) for item in common_ranked])
    ss2 = sum([pow(db[p2][item],2) for item in common_ranked])
    ps  = sum([db[p1][item] * db[p2][item] for item in common_ranked])
    num = n*ps - (s1*s2)
    den = math.sqrt((n * ss1 - math.pow(s1, 2)) * (n * ss2 - math.pow(s2, 2)))
    return (num/den) if den != 0 else 0
similarity = pcc('Michael Stonebraker', 'Alan Perlis')
print(similarity)