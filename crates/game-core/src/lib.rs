#![forbid(unsafe_code)]

/// Fixed-point scale used for world positions and per-tick velocities.
/// One rendered pixel is exactly 1,024 simulation units.
pub const UNITS_PER_PIXEL: i32 = 1_024;
pub const TICKS_PER_SECOND: u32 = 60;
pub const WORLD_WIDTH_PIXELS: i32 = 960;
pub const WORLD_HEIGHT_PIXELS: i32 = 540;
pub const TABLE_TOP_PIXELS: i32 = 438;
pub const ROUND_TICKS: u32 = 30 * TICKS_PER_SECOND;
pub const OBSERVATION_SCHEMA_VERSION: u16 = 1;
pub const ACTION_SCHEMA_VERSION: u16 = 1;
pub const PLAYER_RADIUS_UNITS: i32 = 12 * UNITS_PER_PIXEL;
pub const HAND_RADIUS_UNITS: i32 = 30 * UNITS_PER_PIXEL;
pub const FLIGHT_MINIMUM: Vec2 = Vec2::new(32 * UNITS_PER_PIXEL, 32 * UNITS_PER_PIXEL);
pub const FLIGHT_MAXIMUM: Vec2 = Vec2::new(
    (WORLD_WIDTH_PIXELS - 32) * UNITS_PER_PIXEL,
    (TABLE_TOP_PIXELS - 32) * UNITS_PER_PIXEL,
);
pub const HAND_WIND_UP_TICKS: u16 = 12;
pub const HAND_STRIKE_TICKS: u16 = 14;
pub const HAND_ACTIVE_CONTACT_TICKS: u16 = 5;
pub const HAND_IMPACT_HOLD_TICKS: u16 = 5;
pub const HAND_RECOVER_TICKS: u16 = 36;
pub const HAND_COOLDOWN_TICKS: u16 = 48;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Vec2 {
    pub x: i32,
    pub y: i32,
}

impl Vec2 {
    #[must_use]
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AxisInput {
    /// Horizontal intent, clamped to -1, 0, or 1.
    pub x: i8,
    /// Vertical intent, clamped to -1, 0, or 1.
    pub y: i8,
}

impl AxisInput {
    #[must_use]
    pub const fn new(x: i8, y: i8) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlayerConfig {
    pub world_size: Vec2,
    pub minimum_position: Vec2,
    pub maximum_position: Vec2,
    pub radius: i32,
    pub acceleration_per_tick: i32,
    pub deceleration_per_tick: i32,
    pub max_speed_per_tick: i32,
}

impl Default for PlayerConfig {
    fn default() -> Self {
        Self {
            world_size: Vec2::new(
                WORLD_WIDTH_PIXELS * UNITS_PER_PIXEL,
                WORLD_HEIGHT_PIXELS * UNITS_PER_PIXEL,
            ),
            minimum_position: FLIGHT_MINIMUM,
            maximum_position: FLIGHT_MAXIMUM,
            radius: PLAYER_RADIUS_UNITS,
            acceleration_per_tick: 768,
            deceleration_per_tick: 512,
            max_speed_per_tick: 5 * UNITS_PER_PIXEL,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlayerState {
    pub position: Vec2,
    pub velocity: Vec2,
    pub acceleration: Vec2,
    pub tick: u32,
}

impl PlayerState {
    #[must_use]
    pub fn centered(config: &PlayerConfig) -> Self {
        Self {
            position: Vec2::new(config.world_size.x / 2, config.world_size.y / 2),
            velocity: Vec2::default(),
            acceleration: Vec2::default(),
            tick: 0,
        }
    }

    pub fn step(&mut self, input: AxisInput, config: &PlayerConfig) {
        let raw_desired_velocity = Vec2::new(
            i32::from(input.x.clamp(-1, 1)).saturating_mul(config.max_speed_per_tick),
            i32::from(input.y.clamp(-1, 1)).saturating_mul(config.max_speed_per_tick),
        );
        let desired_velocity = clamp_magnitude(raw_desired_velocity, config.max_speed_per_tick);
        self.step_toward_velocity(desired_velocity, config);
    }

    pub fn step_toward(&mut self, destination: Vec2, config: &PlayerConfig) {
        let offset = Vec2::new(
            destination.x.saturating_sub(self.position.x),
            destination.y.saturating_sub(self.position.y),
        );
        let desired_velocity = clamp_magnitude(offset, config.max_speed_per_tick);
        self.step_toward_velocity(desired_velocity, config);
    }

    fn step_toward_velocity(&mut self, desired_velocity: Vec2, config: &PlayerConfig) {
        let previous_velocity = self.velocity;
        let maximum_change = if desired_velocity == Vec2::default() {
            config.deceleration_per_tick
        } else {
            config.acceleration_per_tick
        };
        self.velocity = move_toward(self.velocity, desired_velocity, maximum_change);
        self.velocity = clamp_magnitude(self.velocity, config.max_speed_per_tick);

        self.position.x = self.position.x.saturating_add(self.velocity.x);
        self.position.y = self.position.y.saturating_add(self.velocity.y);

        clamp_axis(
            &mut self.position.x,
            &mut self.velocity.x,
            config.minimum_position.x,
            config.maximum_position.x,
        );
        clamp_axis(
            &mut self.position.y,
            &mut self.velocity.y,
            config.minimum_position.y,
            config.maximum_position.y,
        );

        self.acceleration = Vec2::new(
            self.velocity.x.saturating_sub(previous_velocity.x),
            self.velocity.y.saturating_sub(previous_velocity.y),
        );

        self.tick = self.tick.wrapping_add(1);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HandAction {
    pub target: Vec2,
    pub strike: bool,
}

impl HandAction {
    #[must_use]
    pub const fn new(target: Vec2, strike: bool) -> Self {
        Self { target, strike }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum HandPhase {
    #[default]
    Track = 0,
    WindUp = 1,
    Strike = 2,
    ImpactHold = 3,
    Recover = 4,
}

impl HandPhase {
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::Track => 0,
            Self::WindUp => 1,
            Self::Strike => 2,
            Self::ImpactHold => 3,
            Self::Recover => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandConfig {
    pub rest_position: Vec2,
    pub palm_radius: i32,
    pub side: i8,
    pub wind_up_ticks: u16,
    pub strike_ticks: u16,
    pub impact_hold_ticks: u16,
    pub recover_ticks: u16,
    pub cooldown_ticks: u16,
}

impl HandConfig {
    #[must_use]
    pub const fn left() -> Self {
        Self {
            rest_position: Vec2::new(210 * UNITS_PER_PIXEL, 344 * UNITS_PER_PIXEL),
            palm_radius: HAND_RADIUS_UNITS,
            side: -1,
            wind_up_ticks: HAND_WIND_UP_TICKS,
            strike_ticks: HAND_STRIKE_TICKS,
            impact_hold_ticks: HAND_IMPACT_HOLD_TICKS,
            recover_ticks: HAND_RECOVER_TICKS,
            cooldown_ticks: HAND_COOLDOWN_TICKS,
        }
    }

    #[must_use]
    pub const fn right() -> Self {
        Self {
            rest_position: Vec2::new(750 * UNITS_PER_PIXEL, 344 * UNITS_PER_PIXEL),
            side: 1,
            ..Self::left()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandState {
    pub phase: HandPhase,
    pub phase_tick: u16,
    pub cooldown_remaining: u16,
    pub palm_position: Vec2,
    pub palm_velocity: Vec2,
    phase_start: Vec2,
    phase_end: Vec2,
    strike_target: Vec2,
}

impl HandState {
    #[must_use]
    pub const fn resting(config: HandConfig) -> Self {
        Self {
            phase: HandPhase::Track,
            phase_tick: 0,
            cooldown_remaining: 0,
            palm_position: config.rest_position,
            palm_velocity: Vec2::new(0, 0),
            phase_start: config.rest_position,
            phase_end: config.rest_position,
            strike_target: config.rest_position,
        }
    }

    #[must_use]
    pub const fn can_strike(self) -> bool {
        matches!(self.phase, HandPhase::Track) && self.cooldown_remaining == 0
    }

    fn begin_strike(&mut self, config: HandConfig, target: Vec2) {
        self.phase = HandPhase::WindUp;
        self.phase_tick = 0;
        self.phase_start = self.palm_position;
        self.phase_end = clamp_hand_position(Vec2::new(
            target
                .x
                .saturating_add(hand_side(config).saturating_mul(160 * UNITS_PER_PIXEL)),
            target.y,
        ));
        self.strike_target = clamp_hand_position(target);
    }

    fn step(&mut self, config: HandConfig, tracking_target: Vec2) {
        let previous_position = self.palm_position;
        self.cooldown_remaining = self.cooldown_remaining.saturating_sub(1);

        match self.phase {
            HandPhase::Track => {
                self.palm_position = move_toward(
                    self.palm_position,
                    tracking_position(config, tracking_target),
                    5 * UNITS_PER_PIXEL,
                );
                self.phase_tick = 0;
            }
            HandPhase::WindUp => {
                if self.advance_motion(config.wind_up_ticks) {
                    self.start_phase(HandPhase::Strike, self.strike_target);
                }
            }
            HandPhase::Strike => {
                if self.advance_motion(config.strike_ticks) {
                    self.phase = HandPhase::ImpactHold;
                    self.phase_tick = 0;
                    self.cooldown_remaining = config.cooldown_ticks;
                }
            }
            HandPhase::ImpactHold => {
                self.phase_tick = self.phase_tick.saturating_add(1);
                if self.phase_tick >= config.impact_hold_ticks {
                    self.start_phase(
                        HandPhase::Recover,
                        tracking_position(config, tracking_target),
                    );
                }
            }
            HandPhase::Recover => {
                if self.advance_motion(config.recover_ticks) {
                    self.phase = HandPhase::Track;
                    self.phase_tick = 0;
                }
            }
        }

        self.palm_velocity = Vec2::new(
            self.palm_position.x.saturating_sub(previous_position.x),
            self.palm_position.y.saturating_sub(previous_position.y),
        );
    }

    fn advance_motion(&mut self, duration: u16) -> bool {
        self.phase_tick = self.phase_tick.saturating_add(1);
        self.palm_position = lerp(
            self.phase_start,
            self.phase_end,
            self.phase_tick,
            duration.max(1),
        );
        self.phase_tick >= duration
    }

    fn start_phase(&mut self, phase: HandPhase, end: Vec2) {
        self.phase = phase;
        self.phase_tick = 0;
        self.phase_start = self.palm_position;
        self.phase_end = end;
    }

    #[must_use]
    pub fn has_active_contact(self, config: HandConfig) -> bool {
        matches!(self.phase, HandPhase::Strike)
            && self.phase_tick
                >= config
                    .strike_ticks
                    .saturating_sub(HAND_ACTIVE_CONTACT_TICKS)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandObservationV1 {
    pub position: Vec2,
    pub velocity: Vec2,
    pub phase: HandPhase,
    pub phase_tick: u16,
    pub cooldown_remaining: u16,
    pub active_contact: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum RoundStatus {
    #[default]
    Active = 0,
    Hit = 1,
    Survived = 2,
}

impl RoundStatus {
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::Active => 0,
            Self::Hit => 1,
            Self::Survived => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundState {
    pub status: RoundStatus,
    pub elapsed_ticks: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservationV1 {
    pub schema_version: u16,
    pub tick: u32,
    pub player_position: Vec2,
    pub player_velocity: Vec2,
    pub player_acceleration: Vec2,
    pub hands: [HandObservationV1; 2],
    pub attack_active: bool,
    pub round_status: RoundStatus,
    pub round_elapsed_ticks: u32,
    pub round_remaining_ticks: u32,
}

impl Default for RoundState {
    fn default() -> Self {
        Self {
            status: RoundStatus::Active,
            elapsed_ticks: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Simulation {
    player_config: PlayerConfig,
    player: PlayerState,
    hand_configs: [HandConfig; 2],
    hands: [HandState; 2],
    attack_active: bool,
    round: RoundState,
}

impl Simulation {
    #[must_use]
    pub fn new() -> Self {
        let player_config = PlayerConfig::default();
        let player = PlayerState::centered(&player_config);
        let hand_configs = [HandConfig::left(), HandConfig::right()];
        let hands = [
            HandState::resting(hand_configs[0]),
            HandState::resting(hand_configs[1]),
        ];
        Self {
            player_config,
            player,
            hand_configs,
            hands,
            attack_active: false,
            round: RoundState::default(),
        }
    }

    #[must_use]
    pub fn with_player_position(position: Vec2) -> Self {
        let mut simulation = Self::new();
        simulation.player.position = Vec2::new(
            position.x.clamp(
                simulation.player_config.minimum_position.x,
                simulation.player_config.maximum_position.x,
            ),
            position.y.clamp(
                simulation.player_config.minimum_position.y,
                simulation.player_config.maximum_position.y,
            ),
        );
        simulation
    }

    pub fn restart(&mut self) {
        *self = Self::new();
    }

    pub fn step(&mut self, input: AxisInput) {
        self.step_axis_with_action(input, HandAction::default());
    }

    pub fn step_toward(&mut self, destination: Vec2) {
        self.step_toward_with_action(destination, HandAction::default());
    }

    pub fn step_axis_with_action(&mut self, input: AxisInput, action: HandAction) {
        if self.round.status != RoundStatus::Active {
            return;
        }
        self.player.step(input, &self.player_config);
        self.step_hands(action);
        self.finish_tick();
    }

    pub fn step_toward_with_action(&mut self, destination: Vec2, action: HandAction) {
        if self.round.status != RoundStatus::Active {
            return;
        }
        self.player.step_toward(destination, &self.player_config);
        self.step_hands(action);
        self.finish_tick();
    }

    fn step_hands(&mut self, action: HandAction) {
        if !self.attack_active && action.strike && self.hands.iter().all(|hand| hand.can_strike()) {
            self.attack_active = true;
            for (hand, config) in self.hands.iter_mut().zip(self.hand_configs) {
                hand.begin_strike(config, action.target);
            }
        }

        for (hand, config) in self.hands.iter_mut().zip(self.hand_configs) {
            hand.step(config, action.target);
        }

        if self.attack_active && self.hands.iter().all(|hand| hand.phase == HandPhase::Track) {
            self.attack_active = false;
        }
    }

    fn finish_tick(&mut self) {
        let hit = self
            .hands
            .iter()
            .zip(self.hand_configs)
            .any(|(hand, config)| {
                hand.has_active_contact(config)
                    && circles_overlap(
                        hand.palm_position,
                        config.palm_radius,
                        self.player.position,
                        self.player_config.radius,
                    )
            });
        if hit {
            self.round.status = RoundStatus::Hit;
        }

        self.round.elapsed_ticks = self.round.elapsed_ticks.saturating_add(1);
        if self.round.elapsed_ticks >= ROUND_TICKS && self.round.status == RoundStatus::Active {
            self.round.status = RoundStatus::Survived;
        }
    }

    #[must_use]
    pub const fn player(&self) -> PlayerState {
        self.player
    }

    #[must_use]
    pub const fn round(&self) -> RoundState {
        self.round
    }

    #[must_use]
    pub const fn attack_active(&self) -> bool {
        self.attack_active
    }

    #[must_use]
    pub const fn player_radius(&self) -> i32 {
        self.player_config.radius
    }

    #[must_use]
    pub fn hand_config(&self, index: usize) -> Option<HandConfig> {
        self.hand_configs.get(index).copied()
    }

    #[must_use]
    pub fn hand(&self, index: usize) -> Option<HandState> {
        self.hands.get(index).copied()
    }

    #[must_use]
    pub fn hand_has_active_contact(&self, index: usize) -> bool {
        self.hands
            .get(index)
            .zip(self.hand_configs.get(index))
            .is_some_and(|(hand, config)| hand.has_active_contact(*config))
    }

    #[must_use]
    pub fn observation_v1(&self) -> ObservationV1 {
        let hand_observation = |index: usize| {
            let hand = self.hands[index];
            HandObservationV1 {
                position: hand.palm_position,
                velocity: hand.palm_velocity,
                phase: hand.phase,
                phase_tick: hand.phase_tick,
                cooldown_remaining: hand.cooldown_remaining,
                active_contact: hand.has_active_contact(self.hand_configs[index]),
            }
        };
        ObservationV1 {
            schema_version: OBSERVATION_SCHEMA_VERSION,
            tick: self.player.tick,
            player_position: self.player.position,
            player_velocity: self.player.velocity,
            player_acceleration: self.player.acceleration,
            hands: [hand_observation(0), hand_observation(1)],
            attack_active: self.attack_active,
            round_status: self.round.status,
            round_elapsed_ticks: self.round.elapsed_ticks,
            round_remaining_ticks: ROUND_TICKS.saturating_sub(self.round.elapsed_ticks),
        }
    }
}

impl Default for Simulation {
    fn default() -> Self {
        Self::new()
    }
}

fn clamp_axis(position: &mut i32, velocity: &mut i32, minimum: i32, maximum: i32) {
    let clamped = (*position).clamp(minimum, maximum);
    if clamped != *position {
        *position = clamped;
        *velocity = 0;
    }
}

fn move_toward(current: Vec2, target: Vec2, maximum_change: i32) -> Vec2 {
    let difference = Vec2::new(
        target.x.saturating_sub(current.x),
        target.y.saturating_sub(current.y),
    );
    let change = clamp_magnitude(difference, maximum_change);
    Vec2::new(
        current.x.saturating_add(change.x),
        current.y.saturating_add(change.y),
    )
}

fn clamp_magnitude(vector: Vec2, maximum: i32) -> Vec2 {
    if maximum <= 0 {
        return Vec2::default();
    }

    let x = i64::from(vector.x);
    let y = i64::from(vector.y);
    let length_squared = u64::try_from(x * x + y * y).unwrap_or(u64::MAX);
    let maximum_u64 = u64::try_from(maximum).unwrap_or_default();
    let maximum_squared = maximum_u64.saturating_mul(maximum_u64);

    if length_squared <= maximum_squared {
        return vector;
    }

    let floor_length = integer_sqrt(length_squared).max(1);
    let length = if floor_length.saturating_mul(floor_length) < length_squared {
        floor_length + 1
    } else {
        floor_length
    };
    let scale = i64::try_from(maximum_u64).unwrap_or(i64::MAX);
    let divisor = i64::try_from(length).unwrap_or(i64::MAX);

    Vec2::new(
        i32::try_from(x.saturating_mul(scale) / divisor)
            .unwrap_or_else(|_| i32::try_from(x.signum()).unwrap_or_default() * maximum),
        i32::try_from(y.saturating_mul(scale) / divisor)
            .unwrap_or_else(|_| i32::try_from(y.signum()).unwrap_or_default() * maximum),
    )
}

fn hand_side(config: HandConfig) -> i32 {
    i32::from(config.side)
}

fn tracking_position(config: HandConfig, target: Vec2) -> Vec2 {
    clamp_hand_position(Vec2::new(
        target
            .x
            .saturating_add(hand_side(config).saturating_mul(110 * UNITS_PER_PIXEL)),
        target.y,
    ))
}

fn clamp_hand_position(position: Vec2) -> Vec2 {
    Vec2::new(
        position.x.clamp(
            30 * UNITS_PER_PIXEL,
            (WORLD_WIDTH_PIXELS - 30) * UNITS_PER_PIXEL,
        ),
        position.y.clamp(
            30 * UNITS_PER_PIXEL,
            (TABLE_TOP_PIXELS - 30) * UNITS_PER_PIXEL,
        ),
    )
}

fn lerp(start: Vec2, end: Vec2, numerator: u16, denominator: u16) -> Vec2 {
    let numerator = i64::from(numerator.min(denominator));
    let denominator = i64::from(denominator.max(1));
    let x = i64::from(start.x) + (i64::from(end.x) - i64::from(start.x)) * numerator / denominator;
    let y = i64::from(start.y) + (i64::from(end.y) - i64::from(start.y)) * numerator / denominator;
    Vec2::new(saturating_i32(x), saturating_i32(y))
}

fn circles_overlap(first: Vec2, first_radius: i32, second: Vec2, second_radius: i32) -> bool {
    let combined_radius = i64::from(first_radius.saturating_add(second_radius));
    distance_squared(first, second) <= combined_radius * combined_radius
}

fn distance_squared(first: Vec2, second: Vec2) -> i64 {
    let x = i64::from(first.x) - i64::from(second.x);
    let y = i64::from(first.y) - i64::from(second.y);
    x * x + y * y
}

fn saturating_i32(value: i64) -> i32 {
    i32::try_from(value).unwrap_or_else(|_| {
        if value.is_negative() {
            i32::MIN
        } else {
            i32::MAX
        }
    })
}

fn integer_sqrt(value: u64) -> u64 {
    if value < 2 {
        return value;
    }

    let mut estimate = value;
    let mut next = u64::midpoint(estimate, value / estimate);
    while next < estimate {
        estimate = next;
        next = u64::midpoint(estimate, value / estimate);
    }
    estimate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_input_sequences_produce_identical_state() {
        let config = PlayerConfig::default();
        let mut first = PlayerState::centered(&config);
        let mut second = first;
        let sequence = [
            AxisInput::new(1, 0),
            AxisInput::new(1, -1),
            AxisInput::new(0, -1),
            AxisInput::new(-1, 1),
        ];

        for tick in 0..1_000 {
            let input = sequence[tick % sequence.len()];
            first.step(input, &config);
            second.step(input, &config);
        }

        assert_eq!(first, second);
        assert_eq!(first.tick, 1_000);
    }

    #[test]
    fn velocity_never_exceeds_the_configured_cap() {
        let config = PlayerConfig::default();
        let mut player = PlayerState::centered(&config);

        for _ in 0..1_000 {
            player.step(AxisInput::new(1, 1), &config);
            let speed_squared = distance_squared(player.velocity, Vec2::default());
            assert!(speed_squared <= i64::from(config.max_speed_per_tick).pow(2));
        }
    }

    #[test]
    fn axis_diagonals_use_the_same_speed_cap_as_cardinal_input() {
        let config = PlayerConfig::default();
        let mut cardinal = PlayerState::centered(&config);
        let mut diagonal = cardinal;

        for _ in 0..30 {
            cardinal.step(AxisInput::new(1, 0), &config);
            diagonal.step(AxisInput::new(1, 1), &config);
        }

        assert_eq!(cardinal.velocity.x, config.max_speed_per_tick);
        assert!(
            distance_squared(diagonal.velocity, Vec2::default())
                <= i64::from(config.max_speed_per_tick).pow(2)
        );
    }

    #[test]
    fn player_cannot_leave_flight_area_or_continue_below_the_table() {
        let config = PlayerConfig::default();
        let mut player = PlayerState::centered(&config);

        for _ in 0..10_000 {
            player.step(AxisInput::new(1, 1), &config);
        }

        assert_eq!(player.position, config.maximum_position);
        assert_eq!(player.velocity, Vec2::default());
    }

    #[test]
    fn pointer_destination_uses_acceleration_and_stops_at_the_target() {
        let config = PlayerConfig::default();
        let mut player = PlayerState::centered(&config);
        let destination = Vec2::new(
            player.position.x + 80 * UNITS_PER_PIXEL,
            player.position.y - 40 * UNITS_PER_PIXEL,
        );

        player.step_toward(destination, &config);
        assert!(player.velocity.x > 0);
        assert!(player.velocity.y < 0);
        assert_ne!(player.velocity, destination);

        for _ in 0..240 {
            player.step_toward(destination, &config);
        }

        assert_eq!(player.position, destination);
        assert_eq!(player.velocity, Vec2::default());
    }

    #[test]
    fn player_decelerates_to_rest_without_input() {
        let config = PlayerConfig::default();
        let mut player = PlayerState::centered(&config);

        for _ in 0..20 {
            player.step(AxisInput::new(1, 0), &config);
        }
        assert!(player.velocity.x > 0);

        for _ in 0..20 {
            player.step(AxisInput::default(), &config);
        }

        assert_eq!(player.velocity, Vec2::default());
    }

    #[test]
    fn both_hands_attack_together_and_contact_ends_the_round() {
        let mut simulation = Simulation::new();
        let target = simulation.player.position;

        simulation.step_axis_with_action(AxisInput::default(), HandAction::new(target, true));
        assert!(simulation.attack_active);
        assert_eq!(simulation.hands[0].phase, HandPhase::WindUp);
        assert_eq!(simulation.hands[1].phase, HandPhase::WindUp);

        for _ in 0..40 {
            simulation.step(AxisInput::default());
            if simulation.round.status == RoundStatus::Hit {
                break;
            }
        }

        assert_eq!(simulation.round.status, RoundStatus::Hit);
    }

    #[test]
    fn hands_follow_opposite_sides_of_the_policy_target() {
        let mut simulation = Simulation::new();
        let target = Vec2::new(700 * UNITS_PER_PIXEL, 180 * UNITS_PER_PIXEL);

        for _ in 0..120 {
            simulation.step_axis_with_action(AxisInput::default(), HandAction::new(target, false));
        }

        assert_eq!(
            simulation.hands[0].palm_position,
            tracking_position(simulation.hand_configs[0], target)
        );
        assert_eq!(
            simulation.hands[1].palm_position,
            tracking_position(simulation.hand_configs[1], target)
        );
        assert!(simulation.hands[0].palm_position.x < target.x);
        assert!(simulation.hands[1].palm_position.x > target.x);
    }

    #[test]
    fn observation_v1_reports_exact_motion_and_round_state() {
        let mut simulation = Simulation::new();
        simulation.step(AxisInput::new(1, 0));

        let observation = simulation.observation_v1();
        assert_eq!(observation.schema_version, OBSERVATION_SCHEMA_VERSION);
        assert_eq!(observation.tick, 1);
        assert_eq!(observation.player_position, simulation.player.position);
        assert_eq!(observation.player_velocity, simulation.player.velocity);
        assert_eq!(observation.player_acceleration, Vec2::new(768, 0));
        assert_eq!(
            observation.hands[0].velocity,
            simulation.hands[0].palm_velocity
        );
        assert_eq!(
            observation.hands[1].velocity,
            simulation.hands[1].palm_velocity
        );
        assert_eq!(observation.round_status, RoundStatus::Active);
        assert_eq!(observation.round_elapsed_ticks, 1);
        assert_eq!(observation.round_remaining_ticks, ROUND_TICKS - 1);
    }

    #[test]
    fn scenario_start_position_is_clamped_to_the_flight_area() {
        let simulation = Simulation::with_player_position(Vec2::new(i32::MIN, i32::MAX));
        assert_eq!(
            simulation.player.position,
            Vec2::new(FLIGHT_MINIMUM.x, FLIGHT_MAXIMUM.y)
        );
        assert_eq!(simulation.player.velocity, Vec2::default());
        assert_eq!(simulation.player.tick, 0);
    }
}
