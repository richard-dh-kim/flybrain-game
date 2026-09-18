#![forbid(unsafe_code)]

use flybrain_game_core::{AxisInput, PlayerConfig, PlayerState, UNITS_PER_PIXEL};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Simulation {
    config: PlayerConfig,
    player: PlayerState,
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        let config = PlayerConfig::default();
        let player = PlayerState::centered(&config);
        Self { config, player }
    }

    pub fn step(&mut self, horizontal: i8, vertical: i8) {
        self.player
            .step(AxisInput::new(horizontal, vertical), &self.config);
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn tick(&self) -> u32 {
        self.player.tick
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn player_x(&self) -> f64 {
        f64::from(self.player.position.x) / f64::from(UNITS_PER_PIXEL)
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn player_y(&self) -> f64 {
        f64::from(self.player.position.y) / f64::from(UNITS_PER_PIXEL)
    }
}

impl Default for Simulation {
    fn default() -> Self {
        Self::new()
    }
}
