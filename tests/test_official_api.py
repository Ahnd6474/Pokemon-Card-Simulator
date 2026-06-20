from types import SimpleNamespace
import unittest

from pokemon_card_simulator import (
    BeamSearchConfig,
    PointDistribution,
    beam_search_point_distribution,
    ensure_cg_api,
    final_point_from_observation,
    iter_selection_choices,
    load_official_attacks,
    load_official_cards,
    normalize_prior,
)


class OfficialApiTests(unittest.TestCase):
    def test_loads_official_card_and_attack_data(self) -> None:
        cards = load_official_cards()
        attacks = load_official_attacks()

        self.assertEqual(len(cards), 1267)
        self.assertEqual(len(attacks), 1556)
        self.assertEqual(cards[0].cardId, 1)
        self.assertGreater(max(card.cardId for card in cards), 1200)

    def test_can_import_official_api_module(self) -> None:
        api = ensure_cg_api()

        self.assertTrue(hasattr(api, "search_begin"))
        self.assertTrue(hasattr(api, "search_step"))
        self.assertTrue(hasattr(api, "all_card_data"))

    def test_iter_selection_choices_uses_min_and_max_count(self) -> None:
        select = SimpleNamespace(minCount=1, maxCount=2, option=[object(), object(), object()])

        choices = iter_selection_choices(select)

        self.assertEqual(choices, ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2)))

    def test_iter_selection_choices_can_limit_count(self) -> None:
        select = SimpleNamespace(minCount=0, maxCount=2, option=[object(), object(), object()])

        choices = iter_selection_choices(select, limit=3)

        self.assertEqual(choices, ((), (0,), (1,)))

    def test_final_point_uses_both_players(self) -> None:
        observation = SimpleNamespace(
            current=SimpleNamespace(
                players=[
                    SimpleNamespace(prize=[object(), object(), object(), object()]),
                    SimpleNamespace(prize=[object(), object()]),
                ]
            )
        )

        self.assertEqual(final_point_from_observation(observation, player_id=0, starting_prize_count=6), (2, 4))
        self.assertEqual(final_point_from_observation(observation, player_id=1, starting_prize_count=6), (4, 2))

    def test_point_distribution_expected_point(self) -> None:
        distribution = PointDistribution({(1, 0): 0.25, (2, 1): 0.75}, 1.0, 2)

        self.assertEqual(distribution.expected_point(), (1.75, 0.75))

    def test_normalize_prior_falls_back_to_uniform(self) -> None:
        self.assertEqual(normalize_prior((0.0, -1.0), 2), (0.5, 0.5))

    def test_beam_config_defaults_to_competition_prizes(self) -> None:
        config = BeamSearchConfig()

        self.assertEqual(config.starting_prize_count, 6)

    def test_beam_distribution_counts_no_choice_leaf_once(self) -> None:
        observation = SimpleNamespace(
            select=None,
            current=SimpleNamespace(
                yourIndex=0,
                result=-1,
                players=[
                    SimpleNamespace(prize=[object()] * 6),
                    SimpleNamespace(prize=[object()] * 6),
                ],
            ),
        )
        root = SimpleNamespace(observation=observation, searchId=1)

        distribution = beam_search_point_distribution(
            root,
            config=BeamSearchConfig(max_depth=3),
        )

        self.assertEqual(distribution.retained_probability, 1.0)
        self.assertEqual(distribution.leaf_count, 1)
        self.assertEqual(distribution.probabilities, {(0, 0): 1.0})


if __name__ == "__main__":
    unittest.main()
