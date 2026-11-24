"""Two guided comparisons on paired evaluation records."""


def math_by_level(questions):
    """Group math questions by difficulty.

    Each question has a level and correct_draws for original, teacher, sft and opd.
    Return a dict keyed by the levels present. Each value contains questions,
    pass_at_1 (a dict of fractions for all four arms), sft_wins, opd_wins and ties.
    Count a question in sft_wins or opd_wins when that method has more correct draws.
    """
    draws_by_level = {}
    for question in questions:
        draws_by_level.setdefault(question["level"], []).append(question["correct_draws"])

    arms = ("original", "teacher", "sft", "opd")
    results = {}
    for level, draws in draws_by_level.items():
        total_draws = 4 * len(draws)
        paired_differences = [row["sft"] - row["opd"] for row in draws]
        results[level] = {
            "questions": len(draws),
            "pass_at_1": {
                arm: sum(row[arm] for row in draws) / total_draws for arm in arms
            },
            "sft_wins": sum(diff > 0 for diff in paired_differences),
            "opd_wins": sum(diff < 0 for diff in paired_differences),
            "ties": sum(diff == 0 for diff in paired_differences),
        }
    return results


def ifeval_transitions(questions):
    """Compare each adapter with the original student at the pass@4 threshold.

    Return a dict keyed by sft and opd. For each adapter, count retained
    (pass to pass), lost (pass to fail), gained (fail to pass) and
    still_failing (fail to fail). A question passes if at least one draw succeeds.
    """
    transition_names = {
        (True, True): "retained",
        (True, False): "lost",
        (False, True): "gained",
        (False, False): "still_failing",
    }
    results = {arm: dict.fromkeys(transition_names.values(), 0) for arm in ("sft", "opd")}
    for question in questions:
        draws = question["correct_draws"]
        original_passed = draws["original"] > 0
        for arm, counts in results.items():
            transition = transition_names[original_passed, draws[arm] > 0]
            counts[transition] += 1
    return results
