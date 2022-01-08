users = [
    {"id": 0, "name": "Hero"},
    {"id": 1, "name": "Dunn"},
    {"id": 2, "name": "Sue"},
    {"id": 3, "name": "Chi"},
    {"id": 4, "name": "Thor"},
    {"id": 5, "name": "Clive"},
    {"id": 6, "name": "Hicks"},
    {"id": 7, "name": "Devin"},
    {"id": 8, "name": "Kate"},
    {"id": 9, "name": "Klein"},
]

friendship_pairs = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 8),
    (8, 9),
]


friendships = {user["id"]: [] for user in users}

for i, j in friendship_pairs:
    friendships[i].append(j)
    friendships[j].append(i)


def number_of_friends(user):
    user_id = user["id"]
    friends_ids = friendships[user_id]
    return len(friends_ids)


def foaf_ids_bad(user):
    foafs = set()
    user_id = user["id"]
    for friend_id in friendships[user_id]:
        for foaf in friendships[friend_id]:
            foafs.add(foaf)
    return foafs


from collections import Counter


def friends_of_friends(user):
    user_id = user["id"]
    foafs = []
    for friend_id in friendships[user_id]:
        for foaf_id in friendships[friend_id]:
            foafs.append(foaf_id)
    return Counter(foafs)


interests = [
    (0, "Hadoop"),
    (0, "Big Data"),
    (0, "HBase"),
    (0, "Java"),
    (0, "Spark"),
    (0, "Storm"),
    (0, "Cassandra"),
    (1, "NoSQL"),
    (1, "MongoDB"),
    (1, "Cassandra"),
    (1, "HBase"),
    (1, "Postgres"),
    (2, "Python"),
    (2, "scikit-learn"),
    (2, "scipy"),
    (2, "numpy"),
    (2, "statsmodels"),
    (2, "pandas"),
    (3, "R"),
    (3, "Python"),
    (3, "statistics"),
    (3, "regression"),
    (3, "probability"),
    (4, "machine learning"),
    (4, "regression"),
    (4, "decision trees"),
    (4, "libsvm"),
    (5, "Python"),
    (5, "R"),
    (5, "Java"),
    (5, "C++"),
    (5, "Haskell"),
    (5, "programming languages"),
    (6, "statistics"),
    (6, "probability"),
    (6, "mathematics"),
    (6, "theory"),
    (7, "machine learning"),
    (7, "scikit-learn"),
    (7, "Mahout"),
    (7, "neural networks"),
    (8, "neural networks"),
    (8, "deep learning"),
    (8, "Big Data"),
    (8, "artificial intelligence"),
    (9, "Hadoop"),
    (9, "Java"),
    (9, "MapReduce"),
    (9, "Big Data"),
]

from typing import List


def data_scientists_who_like(target_interest: str) -> List[int]:
    return [
        user_id
        for user_id, user_interest in interests
        if user_interest == target_interest
    ]


from collections import defaultdict

# Keys are interests, values are lists of user_ids with that interest
user_ids_by_interest = defaultdict(list)
for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)

# Keys are user_ids, values are lists of interests for that user_id.
interests_by_user_id = defaultdict(list)
for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)


def most_common_interests_with(user):
    user_id = user["id"]
    return Counter(
        [
            user_id
            for interest in interests_by_user_id[user_id]
            for user_id in user_ids_by_interest[interest]
        ]
    )


salaries_and_tenures = [
    (83000, 8.7),
    (88000, 8.1),
    (48000, 0.7),
    (76000, 6),
    (69000, 6.5),
    (76000, 7.5),
    (60000, 2.5),
    (83000, 10),
    (48000, 1.9),
    (63000, 4.2),
]

# Keys are years, values are lists of the salaries for each tenure.
salary_by_tenure = defaultdict(list)
for salary, tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)

avg_salary_by_tenure = {}
for tenure, salaries in salary_by_tenure.items():
    avg_salary_by_tenure[tenure] = sum(salaries) / len(salaries)

def tenure_bucket(tenure_in_years):
    if tenure_in_years <= 2:
        return "less than two"
    elif tenure_in_years <5:
        return "between two and five"
    else:
        return "five or more"

salary_by_tenure_bucket = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    key = tenure_bucket(tenure)
    salary_by_tenure_bucket[key].append(salary)

avg_salary_by_tenure_bucket = {
    tenure_bucket: sum(salaries) / len(salaries)
    for tenure_bucket, salaries in salary_by_tenure_bucket.items()
}


if __name__ == "__main__":
    total_connections = sum(number_of_friends(user) for user in users)
    print(f"Number of connections: {total_connections}")

    num_users = len(users)
    avg_connections = total_connections / num_users
    print(f"Average connections: {avg_connections}")

    num_friends_by_id = [
        (id, len(connections)) for id, connections in friendships.items()
    ]
    sorted_connnections_by_user = sorted(
        num_friends_by_id, key=lambda x: x[1], reverse=True
    )
    print(f"user ids sorted by connection: {sorted_connnections_by_user}")

    print(f"Friend of a friend (bad) :{foaf_ids_bad(user=users[0])}")
    print(f"Friend of a friend (Counter) :{friends_of_friends(user=users[0])}")

    target_interest = "Python"
    print(
        f"Data scientists who like `{target_interest}`: {data_scientists_who_like(target_interest)}"
    )

    target_user = users[0]
    print(
        f"Users who have most common with user_id `{target_user['id']}`: {most_common_interests_with(target_user)}"
    )

    print(f"Salary by tenure bucket: {salary_by_tenure_bucket}")
    print(f"Avg salary by tenure bucket: {avg_salary_by_tenure_bucket}")
