from wm.eval.context import build_ruler_examples


def test_ruler_generators_are_deterministic():
    a = build_ruler_examples("ruler_sniah", 2, seed=3, length_words=300)
    b = build_ruler_examples("ruler_sniah", 2, seed=3, length_words=300)
    assert [x.answer for x in a] == [x.answer for x in b]
    assert all(example.answer in example.context for example in a)


def test_all_ruler_tasks_have_answers():
    for task in ["ruler_sniah", "ruler_multi_needle", "ruler_variable_tracking", "ruler_aggregation"]:
        example = build_ruler_examples(task, 1, seed=1, length_words=200)[0]
        assert example.answer
        assert example.context
