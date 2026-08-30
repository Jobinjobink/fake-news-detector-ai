from train import build_pipeline


def test_pipeline_learns_a_small_separable_sample():
    texts = [
        "official agency publishes verified monthly employment report",
        "court releases complete written judgment after public hearing",
        "shocking secret miracle cure doctors desperately hide from you",
        "you will not believe this celebrity conspiracy exposed tonight",
    ] * 3
    labels = [1, 1, 0, 0] * 3
    model = build_pipeline().fit(texts, labels)
    assert model.predict(["official agency releases verified public report"])[0] == 1
    assert model.predict_proba(["secret miracle conspiracy exposed"])[0].shape == (2,)
