#![forbid(unsafe_code)]

use flybrain_game_core::{
    ACTION_SCHEMA_VERSION, AxisInput, FLIGHT_MAXIMUM, FLIGHT_MINIMUM, HAND_ACTIVE_CONTACT_TICKS,
    HAND_COOLDOWN_TICKS, HAND_IMPACT_HOLD_TICKS, HAND_RADIUS_UNITS, HAND_RECOVER_TICKS,
    HAND_STRIKE_TICKS, HAND_WIND_UP_TICKS, HandAction, OBSERVATION_SCHEMA_VERSION,
    PLAYER_RADIUS_UNITS, ROUND_TICKS, Simulation as CoreSimulation, UNITS_PER_PIXEL, Vec2,
};
use pyo3::prelude::*;

type HandObservationTuple = (i32, i32, i32, i32, u8, u16, u16, bool);
type ObservationTuple = (
    u16,
    u32,
    (i32, i32),
    (i32, i32),
    (i32, i32),
    HandObservationTuple,
    HandObservationTuple,
    bool,
    u8,
    u32,
    u32,
);

#[pyclass(module = "flybrain_game")]
struct Simulation {
    inner: CoreSimulation,
}

#[pymethods]
impl Simulation {
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreSimulation::new(),
        }
    }

    #[staticmethod]
    fn with_player_position(horizontal_units: i32, vertical_units: i32) -> Self {
        Self {
            inner: CoreSimulation::with_player_position(Vec2::new(
                horizontal_units,
                vertical_units,
            )),
        }
    }

    fn step(&mut self, horizontal: i8, vertical: i8) {
        self.inner.step(AxisInput::new(horizontal, vertical));
    }

    fn step_toward(&mut self, target_horizontal_units: i32, target_vertical_units: i32) {
        self.inner
            .step_toward(Vec2::new(target_horizontal_units, target_vertical_units));
    }

    fn step_toward_with_action(
        &mut self,
        destination_horizontal_units: i32,
        destination_vertical_units: i32,
        target_horizontal_units: i32,
        target_vertical_units: i32,
        strike: bool,
    ) {
        self.inner.step_toward_with_action(
            Vec2::new(destination_horizontal_units, destination_vertical_units),
            HandAction::new(
                Vec2::new(target_horizontal_units, target_vertical_units),
                strike,
            ),
        );
    }

    fn step_with_action(
        &mut self,
        horizontal: i8,
        vertical: i8,
        target_horizontal_units: i32,
        target_vertical_units: i32,
        strike: bool,
    ) {
        self.inner.step_axis_with_action(
            AxisInput::new(horizontal, vertical),
            HandAction::new(
                Vec2::new(target_horizontal_units, target_vertical_units),
                strike,
            ),
        );
    }

    fn restart(&mut self) {
        self.inner.restart();
    }

    #[getter]
    fn tick(&self) -> u32 {
        self.inner.player().tick
    }

    #[getter]
    fn player_x(&self) -> f64 {
        f64::from(self.inner.player().position.x) / f64::from(UNITS_PER_PIXEL)
    }

    #[getter]
    fn player_y(&self) -> f64 {
        f64::from(self.inner.player().position.y) / f64::from(UNITS_PER_PIXEL)
    }

    #[getter]
    fn player_x_units(&self) -> i32 {
        self.inner.player().position.x
    }

    #[getter]
    fn player_y_units(&self) -> i32 {
        self.inner.player().position.y
    }

    #[getter]
    fn player_velocity_x_units(&self) -> i32 {
        self.inner.player().velocity.x
    }

    #[getter]
    fn player_velocity_y_units(&self) -> i32 {
        self.inner.player().velocity.y
    }

    #[getter]
    fn player_acceleration_x_units(&self) -> i32 {
        self.inner.player().acceleration.x
    }

    #[getter]
    fn player_acceleration_y_units(&self) -> i32 {
        self.inner.player().acceleration.y
    }

    #[getter]
    fn round_status(&self) -> u8 {
        self.inner.round().status.code()
    }

    #[getter]
    fn round_elapsed_ticks(&self) -> u32 {
        self.inner.round().elapsed_ticks
    }

    #[getter]
    fn round_remaining_ticks(&self) -> u32 {
        ROUND_TICKS.saturating_sub(self.inner.round().elapsed_ticks)
    }

    #[getter]
    fn attack_active(&self) -> bool {
        self.inner.attack_active()
    }

    fn hand_state(&self, index: usize) -> Option<(u8, u16, u16, i32, i32)> {
        self.inner.hand(index).map(|hand| {
            (
                hand.phase.code(),
                hand.phase_tick,
                hand.cooldown_remaining,
                hand.palm_position.x,
                hand.palm_position.y,
            )
        })
    }

    fn hand_has_active_contact(&self, index: usize) -> bool {
        self.inner.hand_has_active_contact(index)
    }

    fn observation_v1(&self) -> ObservationTuple {
        let observation = self.inner.observation_v1();
        let hand_tuple = |index: usize| {
            let hand = observation.hands[index];
            (
                hand.position.x,
                hand.position.y,
                hand.velocity.x,
                hand.velocity.y,
                hand.phase.code(),
                hand.phase_tick,
                hand.cooldown_remaining,
                hand.active_contact,
            )
        };
        (
            observation.schema_version,
            observation.tick,
            (observation.player_position.x, observation.player_position.y),
            (observation.player_velocity.x, observation.player_velocity.y),
            (
                observation.player_acceleration.x,
                observation.player_acceleration.y,
            ),
            hand_tuple(0),
            hand_tuple(1),
            observation.attack_active,
            observation.round_status.code(),
            observation.round_elapsed_ticks,
            observation.round_remaining_ticks,
        )
    }
}

#[pymodule]
fn flybrain_game(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Simulation>()?;
    module.add("UNITS_PER_PIXEL", UNITS_PER_PIXEL)?;
    module.add("OBSERVATION_SCHEMA_VERSION", OBSERVATION_SCHEMA_VERSION)?;
    module.add("ACTION_SCHEMA_VERSION", ACTION_SCHEMA_VERSION)?;
    module.add("FLIGHT_MIN_X_UNITS", FLIGHT_MINIMUM.x)?;
    module.add("FLIGHT_MIN_Y_UNITS", FLIGHT_MINIMUM.y)?;
    module.add("FLIGHT_MAX_X_UNITS", FLIGHT_MAXIMUM.x)?;
    module.add("FLIGHT_MAX_Y_UNITS", FLIGHT_MAXIMUM.y)?;
    module.add("PLAYER_RADIUS_UNITS", PLAYER_RADIUS_UNITS)?;
    module.add("HAND_RADIUS_UNITS", HAND_RADIUS_UNITS)?;
    module.add("HAND_WIND_UP_TICKS", HAND_WIND_UP_TICKS)?;
    module.add("HAND_STRIKE_TICKS", HAND_STRIKE_TICKS)?;
    module.add("HAND_ACTIVE_CONTACT_TICKS", HAND_ACTIVE_CONTACT_TICKS)?;
    module.add("HAND_IMPACT_HOLD_TICKS", HAND_IMPACT_HOLD_TICKS)?;
    module.add("HAND_RECOVER_TICKS", HAND_RECOVER_TICKS)?;
    module.add("HAND_COOLDOWN_TICKS", HAND_COOLDOWN_TICKS)?;
    Ok(())
}
