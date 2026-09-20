#![forbid(unsafe_code)]

use flybrain_game_core::{
    ACTION_SCHEMA_VERSION, AxisInput, HandAction, OBSERVATION_SCHEMA_VERSION, ROUND_TICKS,
    Simulation as CoreSimulation, UNITS_PER_PIXEL, Vec2,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[must_use]
pub fn observation_schema_version() -> u16 {
    OBSERVATION_SCHEMA_VERSION
}

#[wasm_bindgen]
#[must_use]
pub fn action_schema_version() -> u16 {
    ACTION_SCHEMA_VERSION
}

#[wasm_bindgen]
pub struct Simulation {
    inner: CoreSimulation,
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: CoreSimulation::new(),
        }
    }

    pub fn step(&mut self, horizontal: i8, vertical: i8) {
        self.inner.step(AxisInput::new(horizontal, vertical));
    }

    pub fn step_toward(&mut self, target_horizontal_units: i32, target_vertical_units: i32) {
        self.inner
            .step_toward(Vec2::new(target_horizontal_units, target_vertical_units));
    }

    pub fn step_with_action(
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

    pub fn step_toward_with_action(
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

    pub fn restart(&mut self) {
        self.inner.restart();
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn tick(&self) -> u32 {
        self.inner.player().tick
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn player_x(&self) -> f64 {
        f64::from(self.inner.player().position.x) / f64::from(UNITS_PER_PIXEL)
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn player_y(&self) -> f64 {
        f64::from(self.inner.player().position.y) / f64::from(UNITS_PER_PIXEL)
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn player_velocity_x(&self) -> f64 {
        f64::from(self.inner.player().velocity.x) / f64::from(UNITS_PER_PIXEL)
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn player_velocity_y(&self) -> f64 {
        f64::from(self.inner.player().velocity.y) / f64::from(UNITS_PER_PIXEL)
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn player_x_units(&self) -> i32 {
        self.inner.player().position.x
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn player_y_units(&self) -> i32 {
        self.inner.player().position.y
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn player_velocity_x_units(&self) -> i32 {
        self.inner.player().velocity.x
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn player_velocity_y_units(&self) -> i32 {
        self.inner.player().velocity.y
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn player_acceleration_x_units(&self) -> i32 {
        self.inner.player().acceleration.x
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn player_acceleration_y_units(&self) -> i32 {
        self.inner.player().acceleration.y
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn round_status(&self) -> u8 {
        self.inner.round().status.code()
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn round_elapsed_ticks(&self) -> u32 {
        self.inner.round().elapsed_ticks
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn round_remaining_ticks(&self) -> u32 {
        ROUND_TICKS.saturating_sub(self.inner.round().elapsed_ticks)
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn attack_active(&self) -> bool {
        self.inner.attack_active()
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn player_radius(&self) -> f64 {
        f64::from(self.inner.player_radius()) / f64::from(UNITS_PER_PIXEL)
    }

    #[must_use]
    pub fn hand_phase(&self, index: u8) -> u8 {
        self.inner
            .hand(usize::from(index))
            .map_or(0, |hand| hand.phase.code())
    }

    #[must_use]
    pub fn hand_phase_tick(&self, index: u8) -> u16 {
        self.inner
            .hand(usize::from(index))
            .map_or(0, |hand| hand.phase_tick)
    }

    #[must_use]
    pub fn hand_cooldown(&self, index: u8) -> u16 {
        self.inner
            .hand(usize::from(index))
            .map_or(0, |hand| hand.cooldown_remaining)
    }

    #[must_use]
    pub fn hand_has_active_contact(&self, index: u8) -> bool {
        self.inner.hand_has_active_contact(usize::from(index))
    }

    #[must_use]
    pub fn hand_palm_radius(&self, index: u8) -> f64 {
        self.inner
            .hand_config(usize::from(index))
            .map_or(0.0, |config| {
                f64::from(config.palm_radius) / f64::from(UNITS_PER_PIXEL)
            })
    }

    #[must_use]
    pub fn hand_palm_x_units(&self, index: u8) -> i32 {
        self.inner
            .hand(usize::from(index))
            .map_or(0, |hand| hand.palm_position.x)
    }

    #[must_use]
    pub fn hand_palm_y_units(&self, index: u8) -> i32 {
        self.inner
            .hand(usize::from(index))
            .map_or(0, |hand| hand.palm_position.y)
    }

    #[must_use]
    pub fn hand_palm_velocity_x_units(&self, index: u8) -> i32 {
        self.inner
            .hand(usize::from(index))
            .map_or(0, |hand| hand.palm_velocity.x)
    }

    #[must_use]
    pub fn hand_palm_velocity_y_units(&self, index: u8) -> i32 {
        self.inner
            .hand(usize::from(index))
            .map_or(0, |hand| hand.palm_velocity.y)
    }

    #[must_use]
    pub fn hand_palm_x(&self, index: u8) -> f64 {
        self.inner.hand(usize::from(index)).map_or(0.0, |hand| {
            f64::from(hand.palm_position.x) / f64::from(UNITS_PER_PIXEL)
        })
    }

    #[must_use]
    pub fn hand_palm_y(&self, index: u8) -> f64 {
        self.inner.hand(usize::from(index)).map_or(0.0, |hand| {
            f64::from(hand.palm_position.y) / f64::from(UNITS_PER_PIXEL)
        })
    }
}

impl Default for Simulation {
    fn default() -> Self {
        Self::new()
    }
}
