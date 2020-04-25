## 코드 기반 collaborative filtering 이해, 장단점

[Build Recommendation Engine with Collaboration Filtering](https://realpython.com/build-recommendation-engine-collaborative-filtering/) 을 요약한 내용입니다.

### 기본 개념 정립

**Collaborative Filtering**

- 개념: filter out ITEMS 유저가 좋아할만한, on the basis of 유사한 유저들의 reactions (행동패턴)

    즉, 사람 베이스 추천

    - search a large group of people and find smaller set of users with similar tastes
    - looking at the items they like & create a ranked list of suggestions
- output: give suggestions to a user on the basis of the LIKES and DISLIKES of similar users
- 예시: amazon, youtube, netflix 이 part of their sophisticated recommendation system 으로 사용중

### 프로세스를 먼저 이해하자

1. 필요한 데이터셋 형태는
    - a set of users → u
    - a set of items → i
    - make into u*i MATRIX
        - filled with 'rating' or some reactions  (ex. 아래 예시에선 5,4,1 등등)

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/84469a4a-2aa8-4c03-b1c5-888cadd8127d/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/84469a4a-2aa8-4c03-b1c5-888cadd8127d/Untitled.png)

        - 너무 빈 칸이 많으면, 즉 reaction이 적은 데이터셋은 sparse하다고 하고, 그 반대는 dense하다고 칭함.
2. Similar users or items 를 찾는다
    - 이 similarity를 따지는 Metric이 상당히 많다. 답이 없다.
    - 유의할 점은 similarity를 따질때 ratings(reactions) 즉 행동양상으로 따지지, 유저 demographic은 영향 없어야 Pure collaborative 이다.
    1. Memory Based

        절차: find users similar to U who rated the item I

        calculate rating R based on the ratings of user found in the previous step

        1) 거리, 거리 측정 metric은 알다시피 다양. 벡터간 거리 계산부터 시작

        2) 유의! tough raters, average raters 의 individual rating preferences 때문에 거리 측정시에 왜곡된 결과 가능

        3) 따라서 need to remove biases of their rating preferences. 

        어떻게 하냐면, normalize each user's ratings 유저별로. 

        ex. 각 rating에 average rating 빼주는 방식→make avg into 0

        그럼 이제 adjusted vectors

        4) fill up the missing values in the vectors

        can be 'average' of each user's rating

3. Predict the rating of the items that are not yet rated by a user
    - 이 predicting rating도 다양한 approach가 있다.
    - 요 prediction의 성능 평가 방식도 다양한 approach 존재.
        1. RMSE
        2. MAE
    1. 상위 2번에서 determined a list of users similar to a user U. need to get R that U would give to a certain item I
    2. Ru = a user's rating R for an item I. u to n is the users that are similar to user U

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b7d78f6a-54c0-4b66-91dd-f2e2ea66965d/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b7d78f6a-54c0-4b66-91dd-f2e2ea66965d/Untitled.png)

    3. 여기에 가중치 추가(weighted average approach). Su = similarity factor of the users each n

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/027a4e78-9d93-4af9-95ec-e521e0dfe8a9/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/027a4e78-9d93-4af9-95ec-e521e0dfe8a9/Untitled.png)

4. 위 방식들은 User-based Collaborative Filtering. 공식이나 프로세스상으로는 거의 유사하나 그냥 Item ~ User exchange 된 것
    - User-based: For a user U, find similar users based on rating vectors → to fill empty rating for an item I → pick out N users
    - Item-based: for an item I, find similar items based on rating vectors → to fill empty rating for a user U → pick out N items

        Amazon에 의해 디벨롭; 유저 수가 아이템 수보다 많을 때 item based filtering is faster and stable. ratings matrix가 정보가 많이 없는, sparse한 상태에서도 user-based 보다 성능 좋다고 함

        문제점은, browsing or entertainment related item such as Movies, where the recommendations it gives out seem very obvious to the target users → 이 경우 matrix factorization technique 또는 hybrid recommenders ( content-based filtering과 섞은 것) 이 효과적이라고 함.

        컬러테일러의 경우 순수 item-based collaborative filtering은 쓰지 않아야 겠다!!

5. Dimensionality reduction of sparse or condense matrix

    BY: matrix factorization or autoencoders

    - matrix factorization: m*n 을 m*p 와 p*n 두개로 각각 부수기

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/07fdd761-d182-43ea-a8c4-de01187f9c9c/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/07fdd761-d182-43ea-a8c4-de01187f9c9c/Untitled.png)

        위 예시 사진의 경우, rating 4 is reduced or factorized into 

        1. User vector (2, -1)
        2. item vector (2.5, 1)

        이 아이들을 latent factors 라고 하며, indication of hidden characteristics about the users of the items

        저 벡터들을 (호러 무비, 로맨스 무비) 라고 할 때, (호러 무비 rating 2, 로맨스 rating -1), (그 무비가 호러일 확률 2.5, 로맨스일 확률 1) ⇒ 이렇게 combine되어서 (2*2.5) + (-1*1) = 4. 

        해석: 이 무비가 호러일 확률이 더 크나, 경미하게 로맨스일 확률이 있어서, 레이팅이 5에서 4로 떨어지는 것으로 예측됨

        → 이 예시에서는 factor 가 두 가지지만, 실제로는 무지 많다.

    - matrix factorization을 위한 알고리즘
        1. [SVD (singular value decomposition)](https://en.wikipedia.org/wiki/Singular_value_decomposition) 
        2. PCA
        3. NMF
        4. [Autoencoders](https://en.wikipedia.org/wiki/Autoencoder) 

            if Neural Network 쓸 때

### 코드로 이해하자

1. 시작하는 자료형 형태는 무난하게 데이터프레임

```python
import pandas as pd
from surprise import Dataset
from surprise import Reader

# This is the same data that was plotted for similarity earlier
# with one new user "E" who has rated only movie 1
ratings_dict = {
    "item": [1, 2, 1, 2, 1, 2, 1, 2, 1],
    "user": ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],
    "rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3],
} 

df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(1, 5))

# Loads Pandas dataframe
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
# Loads the builtin Movielens-100k data
movielens = Dataset.load_builtin('ml-100k')
```

2. 먼저 similarity 계산 알고리즘이 centered KNN을 할 수 있다고 함. memory based approaches 일 경우에 

```python
from surprise import KNNWithMeans

# To use item-based cosine similarity
sim_options = {
	    "name": "cosine", # 거리 메트릭은 데이터 형태에 맞게 고르는 것으로!
    "user_based": False,  # Compute  similarities between items
}
algo = KNNWithMeans(sim_options=sim_options)
```

그리고 cross-validated 사용할 것

```python
from surprise import KNNWithMeans
from surprise import Dataset
from surprise.model_selection import GridSearchCV

data = Dataset.load_builtin("ml-100k")
sim_options = {
    "name": ["msd", "cosine"],
    "min_support": [3, 4, 5],
    "user_based": [False, True],
}

param_grid = {"sim_options": sim_options}

gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=3)
# 위 라인에서 'KNNWithMeans'만 matrix factorization method로 import 후에 바꿔주면 됨 아주 간단

gs.fit(data)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])
```

요렇게 트레이닝

3. matrix factorization

[https://surprise.readthedocs.io/en/stable/matrix_factorization.html](https://surprise.readthedocs.io/en/stable/matrix_factorization.html)

모듈 안에 다양하게 있다. 이름이 촌스럽게 왜 surprise람?

### 콜라보 필터링은 언제 쓰여?

- 수퍼마켓 인벤토리. 다양한 카테고리의 아이템에 대한 reaction이 존재할 때

    예를 들어, 서점이나 영화의 개인의 취향 bias가 끼는 곳에서는 사용하면 성능 떨어짐

    이런 경우에는 content-based or hybrid approaches 가 좋다

- 추천시스템이 not overspecialize in a user's profile, recommend items that are completely different from what they have seen before.

### 콜라보 필터링 문제점?

1. cold start

    until anyone rates them , they don't get recommended

2. data sparsity의 영향이 크다
3. if dataset is large or growing, item-based is faster. 스케일링이 어렵다고 함(dimension reduction을 의미하는 듯)
4. long tail items ignored

    주로 already popular 한 아이템이 추천된다고 함

### 참고 라이브러리

Libraries:

- **[LightFM](https://github.com/lyst/lightfm):** a hybrid recommendation algorithm in Python
- **[Python-recsys](https://github.com/ocelma/python-recsys):** a Python library for implementing a recommender system
