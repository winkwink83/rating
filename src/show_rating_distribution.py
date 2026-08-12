from collections import Counter

from dataset import load_ratings_data


def show_rating_distribution():
    ratings_data = load_ratings_data()

    if not ratings_data:
        print("Brak ocen w datasetcie.")
        return

    ratings = []

    for item in ratings_data:
        rating = int(round(item["rating"]))

        ratings.append(rating)

    counter = Counter(ratings)

    total = len(ratings)

    print()
    print("RATING DISTRIBUTION")
    print("===================")
    print()

    for rating in range(1, 11):
        count = counter.get(rating, 0)

        percentage = (count / total) * 100

        bar = "█" * count

        print(
            f"{rating:>2}: "
            f"{count:>4} "
            f"({percentage:>5.1f}%) "
            f"{bar}"
        )

    print()
    print(f"Total ratings: {total}")


if __name__ == "__main__":
    show_rating_distribution()