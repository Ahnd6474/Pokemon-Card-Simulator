from types import SimpleNamespace
import unittest

from pokemon_card_simulator import (
    BeamSearchConfig,
    GameOutcomeSearchConfig,
    PointDistribution,
    STATE_FEATURE_NAMES,
    TurnSequenceSearchConfig,
    beam_search_game_outcome_distribution,
    beam_search_point_distribution,
    beam_search_turn_sequence_distribution,
    encode_game_state,
    ensure_cg_api,
    final_point_from_observation,
    is_turn_boundary,
    iter_selection_choices,
    load_official_attacks,
    load_official_cards,
    normalize_prior,
    outcome_point_from_observation,
    terminal_result_reason,
)
import pokemon_card_simulator.official as official


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

    def test_encode_game_state_returns_bounded_numeric_vector(self) -> None:
        pokemon = SimpleNamespace(
            hp=40,
            maxHp=70,
            energies=[1, 2],
            energyCards=[],
            tools=[object()],
            appearThisTurn=True,
        )
        player = SimpleNamespace(
            active=[pokemon],
            bench=[],
            benchMax=5,
            prize=[object()] * 5,
            deckCount=40,
            handCount=7,
            discard=[object()] * 3,
            asleep=False,
            burned=False,
            confused=False,
            paralyzed=False,
            poisoned=False,
        )
        opponent = SimpleNamespace(
            active=[],
            bench=[],
            benchMax=5,
            prize=[object()] * 6,
            deckCount=42,
            handCount=5,
            discard=[],
            asleep=False,
            burned=False,
            confused=False,
            paralyzed=False,
            poisoned=False,
        )
        observation = SimpleNamespace(
            select=SimpleNamespace(type=0, context=0, minCount=1, maxCount=1, option=[object(), object()]),
            current=SimpleNamespace(
                turn=3,
                yourIndex=0,
                result=-1,
                supporterPlayed=True,
                stadiumPlayed=False,
                energyAttached=True,
                retreated=False,
                turnActionCount=2,
                players=[player, opponent],
            ),
        )

        vector = encode_game_state(observation, player_id=0)

        self.assertEqual(len(vector), len(STATE_FEATURE_NAMES))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in vector))
        self.assertGreater(vector[STATE_FEATURE_NAMES.index("self_active_damage_norm")], 0.0)

    def test_outcome_point_maps_non_prize_wins_to_max_score(self) -> None:
        observation = SimpleNamespace(
            logs=[SimpleNamespace(result=0, reason=2)],
            current=SimpleNamespace(
                result=0,
                players=[
                    SimpleNamespace(prize=[object()] * 4),
                    SimpleNamespace(prize=[object()] * 6),
                ],
            ),
        )

        self.assertEqual(terminal_result_reason(observation), 2)
        self.assertEqual(outcome_point_from_observation(observation, player_id=0), (6, 0))
        self.assertEqual(outcome_point_from_observation(observation, player_id=1), (0, 6))

    def test_outcome_point_keeps_prize_win_score(self) -> None:
        observation = SimpleNamespace(
            logs=[SimpleNamespace(result=0, reason=1)],
            current=SimpleNamespace(
                result=0,
                players=[
                    SimpleNamespace(prize=[]),
                    SimpleNamespace(prize=[object()] * 2),
                ],
            ),
        )

        self.assertEqual(outcome_point_from_observation(observation, player_id=0), (6, 4))

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

    def test_turn_boundary_detects_next_player_or_turn(self) -> None:
        root = SimpleNamespace(turn=3, yourIndex=0, result=-1)

        self.assertFalse(is_turn_boundary(root, SimpleNamespace(turn=3, yourIndex=0, result=-1)))
        self.assertTrue(is_turn_boundary(root, SimpleNamespace(turn=4, yourIndex=1, result=-1)))
        self.assertTrue(is_turn_boundary(root, SimpleNamespace(turn=3, yourIndex=1, result=-1)))
        self.assertTrue(is_turn_boundary(root, SimpleNamespace(turn=3, yourIndex=0, result=0)))

    def test_turn_sequence_distribution_keeps_selection_sequences(self) -> None:
        root_state = SimpleNamespace(
            turn=1,
            yourIndex=0,
            result=-1,
            players=[
                SimpleNamespace(prize=[object()] * 6),
                SimpleNamespace(prize=[object()] * 6),
            ],
        )
        root_observation = SimpleNamespace(
            select=SimpleNamespace(minCount=1, maxCount=1, option=[object(), object()]),
            current=root_state,
        )
        root = SimpleNamespace(observation=root_observation, searchId=10)

        class FakeApi:
            @staticmethod
            def search_step(_search_id, select):
                if select == [0]:
                    prizes = ([object()] * 5, [object()] * 6)
                else:
                    prizes = ([object()] * 6, [object()] * 5)
                current = SimpleNamespace(
                    turn=2,
                    yourIndex=1,
                    result=-1,
                    players=[
                        SimpleNamespace(prize=prizes[0]),
                        SimpleNamespace(prize=prizes[1]),
                    ],
                )
                return SimpleNamespace(
                    observation=SimpleNamespace(select=None, current=current),
                    searchId=20 + select[0],
                )

        original_ensure_cg_api = official.ensure_cg_api
        official.ensure_cg_api = lambda _cg_root=None: FakeApi
        try:
            distribution = beam_search_turn_sequence_distribution(
                root,
                config=TurnSequenceSearchConfig(max_sequence_steps=3, beam_width=4),
            )
        finally:
            official.ensure_cg_api = original_ensure_cg_api

        self.assertEqual(distribution.retained_probability, 1.0)
        self.assertEqual(distribution.truncated_count, 0)
        self.assertEqual(distribution.point_probabilities, {(0, 1): 0.5, (1, 0): 0.5})
        self.assertEqual(
            {leaf.sequence for leaf in distribution.sequence_leaves},
            {((0,),), ((1,),)},
        )

    def test_game_outcome_distribution_rolls_across_turns_to_terminal(self) -> None:
        root_state = SimpleNamespace(
            turn=1,
            yourIndex=0,
            result=-1,
            players=[
                SimpleNamespace(prize=[object()] * 6),
                SimpleNamespace(prize=[object()] * 6),
            ],
        )
        root_observation = SimpleNamespace(
            logs=[],
            select=SimpleNamespace(
                type=0,
                context=0,
                minCount=1,
                maxCount=1,
                option=[SimpleNamespace(type=7), SimpleNamespace(type=14)],
            ),
            current=root_state,
        )
        root = SimpleNamespace(observation=root_observation, searchId=10)

        class FakeApi:
            @staticmethod
            def search_step(search_id, select):
                if search_id == 10 and select == [0]:
                    current = SimpleNamespace(
                        turn=2,
                        yourIndex=1,
                        result=-1,
                        players=[
                            SimpleNamespace(prize=[object()] * 6),
                            SimpleNamespace(prize=[object()] * 6),
                        ],
                    )
                    return SimpleNamespace(
                        observation=SimpleNamespace(
                            logs=[],
                            select=SimpleNamespace(
                                type=0,
                                context=0,
                                minCount=1,
                                maxCount=1,
                                option=[SimpleNamespace(type=14)],
                            ),
                            current=current,
                        ),
                        searchId=20,
                    )
                if search_id == 10 and select == [1]:
                    current = SimpleNamespace(
                        turn=1,
                        yourIndex=0,
                        result=1,
                        players=[
                            SimpleNamespace(prize=[object()] * 6),
                            SimpleNamespace(prize=[object()] * 6),
                        ],
                    )
                    return SimpleNamespace(
                        observation=SimpleNamespace(
                            logs=[SimpleNamespace(result=1, reason=4)],
                            select=None,
                            current=current,
                        ),
                        searchId=21,
                    )
                current = SimpleNamespace(
                    turn=2,
                    yourIndex=1,
                    result=0,
                    players=[
                        SimpleNamespace(prize=[object()] * 6),
                        SimpleNamespace(prize=[object()] * 6),
                    ],
                )
                return SimpleNamespace(
                    observation=SimpleNamespace(
                        logs=[SimpleNamespace(result=0, reason=2)],
                        select=None,
                        current=current,
                    ),
                    searchId=30,
                )

        original_ensure_cg_api = official.ensure_cg_api
        official.ensure_cg_api = lambda _cg_root=None: FakeApi
        try:
            distribution = beam_search_game_outcome_distribution(
                root,
                config=GameOutcomeSearchConfig(
                    beam_width=8,
                    max_turns=4,
                    max_total_steps=8,
                ),
                player_id=0,
            )
        finally:
            official.ensure_cg_api = original_ensure_cg_api

        self.assertEqual(distribution.total_case_count, 2)
        self.assertEqual(distribution.terminal_count, 2)
        self.assertEqual(distribution.truncated_count, 0)
        self.assertEqual(distribution.point_case_counts, {(0, 6): 1, (6, 0): 1})
        self.assertEqual(distribution.point_probabilities, {(0, 6): 0.5, (6, 0): 0.5})

    def test_game_outcome_choice_filter_excludes_cases_before_counting(self) -> None:
        root_state = SimpleNamespace(
            turn=1,
            yourIndex=0,
            result=-1,
            players=[
                SimpleNamespace(prize=[object()] * 6),
                SimpleNamespace(prize=[object()] * 6),
            ],
        )
        root = SimpleNamespace(
            observation=SimpleNamespace(
                logs=[],
                select=SimpleNamespace(
                    type=0,
                    context=0,
                    minCount=1,
                    maxCount=1,
                    option=[SimpleNamespace(type=7), SimpleNamespace(type=14)],
                ),
                current=root_state,
            ),
            searchId=10,
        )

        class FakeApi:
            @staticmethod
            def search_step(_search_id, select):
                current = SimpleNamespace(
                    turn=1,
                    yourIndex=0,
                    result=select[0],
                    players=[
                        SimpleNamespace(prize=[object()] * 6),
                        SimpleNamespace(prize=[object()] * 6),
                    ],
                )
                return SimpleNamespace(
                    observation=SimpleNamespace(
                        logs=[SimpleNamespace(result=select[0], reason=4)],
                        select=None,
                        current=current,
                    ),
                    searchId=20 + select[0],
                )

        original_ensure_cg_api = official.ensure_cg_api
        official.ensure_cg_api = lambda _cg_root=None: FakeApi
        try:
            distribution = beam_search_game_outcome_distribution(
                root,
                config=GameOutcomeSearchConfig(beam_width=8, max_turns=4, max_total_steps=8),
                player_id=0,
                choice_filter=lambda _state, choice: choice == (0,),
            )
        finally:
            official.ensure_cg_api = original_ensure_cg_api

        self.assertEqual(distribution.total_case_count, 1)
        self.assertEqual(distribution.point_case_counts, {(6, 0): 1})
        self.assertEqual(distribution.point_probabilities, {(6, 0): 1.0})

    def test_game_outcome_sequence_filter_excludes_cases_before_counting(self) -> None:
        root_state = SimpleNamespace(
            turn=1,
            yourIndex=0,
            result=-1,
            players=[
                SimpleNamespace(prize=[object()] * 6),
                SimpleNamespace(prize=[object()] * 6),
            ],
        )
        root = SimpleNamespace(
            observation=SimpleNamespace(
                logs=[],
                select=SimpleNamespace(
                    type=0,
                    context=0,
                    minCount=1,
                    maxCount=1,
                    option=[SimpleNamespace(type=7), SimpleNamespace(type=14)],
                ),
                current=root_state,
            ),
            searchId=10,
        )

        class FakeApi:
            @staticmethod
            def search_step(_search_id, select):
                current = SimpleNamespace(
                    turn=1,
                    yourIndex=0,
                    result=select[0],
                    players=[
                        SimpleNamespace(prize=[object()] * 6),
                        SimpleNamespace(prize=[object()] * 6),
                    ],
                )
                return SimpleNamespace(
                    observation=SimpleNamespace(
                        logs=[SimpleNamespace(result=select[0], reason=4)],
                        select=None,
                        current=current,
                    ),
                    searchId=20 + select[0],
                )

        original_ensure_cg_api = official.ensure_cg_api
        official.ensure_cg_api = lambda _cg_root=None: FakeApi
        try:
            distribution = beam_search_game_outcome_distribution(
                root,
                config=GameOutcomeSearchConfig(beam_width=8, max_turns=4, max_total_steps=8),
                player_id=0,
                sequence_choice_filter=lambda _state, _choice, sequence, _history: sequence == ((0,),),
            )
        finally:
            official.ensure_cg_api = original_ensure_cg_api

        self.assertEqual(distribution.total_case_count, 1)
        self.assertEqual(distribution.point_case_counts, {(6, 0): 1})

    def test_game_outcome_node_ranker_controls_beam_pruning(self) -> None:
        root_state = SimpleNamespace(
            turn=1,
            yourIndex=0,
            result=-1,
            players=[
                SimpleNamespace(prize=[object()] * 6),
                SimpleNamespace(prize=[object()] * 6),
            ],
        )
        root = SimpleNamespace(
            observation=SimpleNamespace(
                logs=[],
                select=SimpleNamespace(
                    type=0,
                    context=0,
                    minCount=1,
                    maxCount=1,
                    option=[SimpleNamespace(type=7), SimpleNamespace(type=14)],
                ),
                current=root_state,
            ),
            searchId=10,
        )

        class FakeApi:
            @staticmethod
            def search_step(_search_id, select):
                if select == [1]:
                    return SimpleNamespace(
                        observation=SimpleNamespace(
                            logs=[SimpleNamespace(result=0, reason=4)],
                            select=None,
                            current=SimpleNamespace(
                                turn=1,
                                yourIndex=0,
                                result=0,
                                players=[
                                    SimpleNamespace(prize=[object()] * 6),
                                    SimpleNamespace(prize=[object()] * 6),
                                ],
                            ),
                        ),
                        searchId=21,
                    )
                return SimpleNamespace(
                    observation=SimpleNamespace(
                        logs=[],
                        select=None,
                        current=SimpleNamespace(
                            turn=1,
                            yourIndex=0,
                            result=-1,
                            players=[
                                SimpleNamespace(prize=[object()] * 6),
                                SimpleNamespace(prize=[object()] * 6),
                            ],
                        ),
                    ),
                    searchId=20,
                )

        original_ensure_cg_api = official.ensure_cg_api
        official.ensure_cg_api = lambda _cg_root=None: FakeApi
        try:
            distribution = beam_search_game_outcome_distribution(
                root,
                config=GameOutcomeSearchConfig(
                    beam_width=1,
                    max_turns=4,
                    max_total_steps=4,
                    release_pruned_states=False,
                ),
                player_id=0,
                node_ranker=lambda node: 1.0 if node.search_state.observation.current.result >= 0 else 0.0,
            )
        finally:
            official.ensure_cg_api = original_ensure_cg_api

        self.assertEqual(distribution.terminal_count, 1)
        self.assertEqual(distribution.truncated_count, 0)
        self.assertEqual(distribution.terminal_case_count, 1)
        self.assertEqual(distribution.terminal_depth_counts, {1: 1})

    def test_game_outcome_absolute_turn_cap_truncates_long_games(self) -> None:
        root = SimpleNamespace(
            observation=SimpleNamespace(
                logs=[],
                select=SimpleNamespace(
                    type=0,
                    context=0,
                    minCount=1,
                    maxCount=1,
                    option=[SimpleNamespace(type=14)],
                ),
                current=SimpleNamespace(
                    turn=17,
                    yourIndex=0,
                    result=-1,
                    players=[
                        SimpleNamespace(prize=[object()] * 6),
                        SimpleNamespace(prize=[object()] * 6),
                    ],
                ),
            ),
            searchId=10,
        )

        distribution = beam_search_game_outcome_distribution(
            root,
            config=GameOutcomeSearchConfig(max_absolute_turn=16),
            player_id=0,
        )

        self.assertEqual(distribution.total_case_count, 1)
        self.assertEqual(distribution.terminal_count, 0)
        self.assertEqual(distribution.truncated_count, 1)


if __name__ == "__main__":
    unittest.main()
