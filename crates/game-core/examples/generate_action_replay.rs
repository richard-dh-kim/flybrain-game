use std::io::{self, Write};

use flybrain_game_core::{
    AxisInput, HandAction, ROUND_TICKS, RoundStatus, Simulation, TABLE_TOP_PIXELS, UNITS_PER_PIXEL,
    Vec2, WORLD_WIDTH_PIXELS,
};

const STRIKE_INTERVAL_TICKS: u32 = 72;

fn main() -> io::Result<()> {
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    writeln!(output, "# FlyBrain deterministic action replay v1")?;
    writeln!(
        output,
        "# episode,expected_tick,input_x,input_y,target_x,target_y,strike,player_x,player_y,velocity_x,velocity_y,round_status,elapsed_ticks,attack_active,left_phase,left_phase_tick,left_cooldown,left_palm_x,left_palm_y,left_contact,right_phase,right_phase_tick,right_cooldown,right_palm_x,right_palm_y,right_contact"
    )?;

    write_survival_episode(&mut output)?;
    write_hit_episode(&mut output)?;
    output.flush()
}

fn write_survival_episode(output: &mut impl Write) -> io::Result<()> {
    let directions = [
        AxisInput::new(1, 1),
        AxisInput::new(-1, 1),
        AxisInput::new(-1, -1),
        AxisInput::new(1, -1),
    ];
    let mut simulation = Simulation::new();

    while simulation.round().status == RoundStatus::Active && simulation.player().tick < ROUND_TICKS
    {
        let tick = simulation.player().tick;
        let direction_index = usize::try_from((tick / STRIKE_INTERVAL_TICKS) % 4)
            .expect("direction index must fit in usize");
        let input = directions[direction_index];
        let player = simulation.player();
        let target = Vec2::new(
            if player.position.x < (WORLD_WIDTH_PIXELS * UNITS_PER_PIXEL) / 2 {
                (WORLD_WIDTH_PIXELS - 60) * UNITS_PER_PIXEL
            } else {
                60 * UNITS_PER_PIXEL
            },
            if player.position.y < (TABLE_TOP_PIXELS * UNITS_PER_PIXEL) / 2 {
                (TABLE_TOP_PIXELS - 60) * UNITS_PER_PIXEL
            } else {
                60 * UNITS_PER_PIXEL
            },
        );
        let action = HandAction::new(
            target,
            tick >= 45 && tick.is_multiple_of(STRIKE_INTERVAL_TICKS),
        );
        simulation.step_axis_with_action(input, action);
        write_tick(output, 0, input, action, &simulation)?;
    }

    assert_eq!(simulation.round().status, RoundStatus::Survived);
    Ok(())
}

fn write_hit_episode(output: &mut impl Write) -> io::Result<()> {
    let mut simulation = Simulation::new();
    let input = AxisInput::default();

    while simulation.round().status == RoundStatus::Active && simulation.player().tick < 200 {
        let player = simulation.player();
        let action = HandAction::new(player.position, player.tick == 45);
        simulation.step_axis_with_action(input, action);
        write_tick(output, 1, input, action, &simulation)?;
    }

    assert_eq!(simulation.round().status, RoundStatus::Hit);
    Ok(())
}

fn write_tick(
    output: &mut impl Write,
    episode: u8,
    input: AxisInput,
    action: HandAction,
    simulation: &Simulation,
) -> io::Result<()> {
    let player = simulation.player();
    let round = simulation.round();
    let left = simulation.hand(0).expect("left hand must exist");
    let right = simulation.hand(1).expect("right hand must exist");
    writeln!(
        output,
        "{episode},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        player.tick,
        input.x,
        input.y,
        action.target.x,
        action.target.y,
        u8::from(action.strike),
        player.position.x,
        player.position.y,
        player.velocity.x,
        player.velocity.y,
        round.status.code(),
        round.elapsed_ticks,
        u8::from(simulation.attack_active()),
        left.phase.code(),
        left.phase_tick,
        left.cooldown_remaining,
        left.palm_position.x,
        left.palm_position.y,
        u8::from(simulation.hand_has_active_contact(0)),
        right.phase.code(),
        right.phase_tick,
        right.cooldown_remaining,
        right.palm_position.x,
        right.palm_position.y,
        u8::from(simulation.hand_has_active_contact(1)),
    )
}
