use flybrain_game_core::{AxisInput, HandAction, RoundStatus, Simulation, Vec2};

const REPLAY: &str = include_str!("../../../tests/replays/player-movement-v1.csv");
const ACTION_REPLAY: &str = include_str!("../../../tests/replays/action-round-v1.csv");

#[test]
fn movement_replay_matches_golden_checkpoints() {
    let mut simulation = Simulation::new();

    for (index, line) in REPLAY.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let values: Vec<i32> = line
            .split(',')
            .map(|value| value.parse().expect("fixture values must be integers"))
            .collect();
        assert_eq!(values.len(), 8, "invalid fixture row {}", index + 1);

        let ticks = u32::try_from(values[2]).expect("tick count must be nonnegative");
        let horizontal = i8::try_from(values[0]).expect("horizontal input must fit in i8");
        let vertical = i8::try_from(values[1]).expect("vertical input must fit in i8");
        for _ in 0..ticks {
            simulation.step(AxisInput::new(horizontal, vertical));
        }

        let player = simulation.player();
        assert_eq!(
            player.tick,
            u32::try_from(values[3]).expect("expected tick must be nonnegative"),
            "tick at row {}",
            index + 1
        );
        assert_eq!(
            player.position,
            Vec2::new(values[4], values[5]),
            "position at row {}",
            index + 1
        );
        assert_eq!(
            player.velocity,
            Vec2::new(values[6], values[7]),
            "velocity at row {}",
            index + 1
        );
    }
}

#[test]
fn action_replay_matches_every_gameplay_tick() {
    let mut simulation = Simulation::new();
    let mut current_episode = None;
    let mut phases_seen = [[false; 5]; 2];
    let mut replay_ticks = 0;

    for (index, line) in ACTION_REPLAY.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let values: Vec<i32> = line
            .split(',')
            .map(|value| value.parse().expect("fixture values must be integers"))
            .collect();
        assert_eq!(values.len(), 26, "invalid fixture row {}", index + 1);

        let episode = values[0];
        if current_episode != Some(episode) {
            if let Some(previous_episode) = current_episode {
                assert_eq!(episode, previous_episode + 1, "episode sequence");
                assert_eq!(simulation.round().status, RoundStatus::Survived);
                simulation.restart();
            }
            current_episode = Some(episode);
        }

        simulation.step_axis_with_action(
            AxisInput::new(
                i8::try_from(values[2]).expect("horizontal input must fit in i8"),
                i8::try_from(values[3]).expect("vertical input must fit in i8"),
            ),
            HandAction::new(Vec2::new(values[4], values[5]), values[6] == 1),
        );
        replay_ticks += 1;

        let player = simulation.player();
        let round = simulation.round();
        assert_eq!(player.tick, as_u32(values[1]), "tick at row {}", index + 1);
        assert_eq!(
            player.position,
            Vec2::new(values[7], values[8]),
            "player position at row {}",
            index + 1
        );
        assert_eq!(
            player.velocity,
            Vec2::new(values[9], values[10]),
            "player velocity at row {}",
            index + 1
        );
        assert_eq!(
            round.status.code(),
            as_u8(values[11]),
            "round status at row {}",
            index + 1
        );
        assert_eq!(
            round.elapsed_ticks,
            as_u32(values[12]),
            "round elapsed ticks at row {}",
            index + 1
        );
        assert_eq!(
            simulation.attack_active(),
            values[13] == 1,
            "attack active at row {}",
            index + 1
        );

        for (hand_index, offset) in [(0, 14), (1, 20)] {
            let expected_phase =
                assert_hand_state(&simulation, hand_index, &values, offset, index + 1);
            if episode == 0 {
                phases_seen[hand_index][usize::from(expected_phase)] = true;
            }
        }
    }

    assert_eq!(replay_ticks, 1_868);
    assert_eq!(current_episode, Some(1));
    assert_eq!(simulation.round().status, RoundStatus::Hit);
    assert!(
        phases_seen.iter().all(|hand| hand.iter().all(|seen| *seen)),
        "survival episode must exercise every phase on both hands"
    );
}

fn assert_hand_state(
    simulation: &Simulation,
    hand_index: usize,
    values: &[i32],
    offset: usize,
    row: usize,
) -> u8 {
    let hand = simulation
        .hand(hand_index)
        .expect("fixture hand must exist");
    let expected_phase = as_u8(values[offset]);
    assert_eq!(
        hand.phase.code(),
        expected_phase,
        "hand {hand_index} phase at row {row}"
    );
    assert_eq!(
        hand.phase_tick,
        as_u16(values[offset + 1]),
        "hand {hand_index} phase tick at row {row}"
    );
    assert_eq!(
        hand.cooldown_remaining,
        as_u16(values[offset + 2]),
        "hand {hand_index} cooldown at row {row}"
    );
    assert_eq!(
        hand.palm_position,
        Vec2::new(values[offset + 3], values[offset + 4]),
        "hand {hand_index} palm at row {row}"
    );
    assert_eq!(
        simulation.hand_has_active_contact(hand_index),
        values[offset + 5] == 1,
        "hand {hand_index} contact at row {row}"
    );
    expected_phase
}

fn as_u8(value: i32) -> u8 {
    u8::try_from(value).expect("fixture value must fit in u8")
}

fn as_u16(value: i32) -> u16 {
    u16::try_from(value).expect("fixture value must fit in u16")
}

fn as_u32(value: i32) -> u32 {
    u32::try_from(value).expect("fixture value must fit in u32")
}
