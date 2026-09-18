#![forbid(unsafe_code)]

/// Fixed-point scale used for world positions and per-tick velocities.
/// One rendered pixel is exactly 1,024 simulation units.
pub const UNITS_PER_PIXEL: i32 = 1_024;

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
    pub radius: i32,
    pub acceleration_per_tick: i32,
    pub deceleration_per_tick: i32,
    pub max_speed_per_tick: i32,
}

impl Default for PlayerConfig {
    fn default() -> Self {
        Self {
            world_size: Vec2::new(960 * UNITS_PER_PIXEL, 540 * UNITS_PER_PIXEL),
            radius: 12 * UNITS_PER_PIXEL,
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
    pub tick: u32,
}

impl PlayerState {
    #[must_use]
    pub fn centered(config: &PlayerConfig) -> Self {
        Self {
            position: Vec2::new(config.world_size.x / 2, config.world_size.y / 2),
            velocity: Vec2::default(),
            tick: 0,
        }
    }

    pub fn step(&mut self, input: AxisInput, config: &PlayerConfig) {
        let axis_x = i32::from(input.x.clamp(-1, 1));
        let axis_y = i32::from(input.y.clamp(-1, 1));

        self.velocity.x = accelerate_axis(
            self.velocity.x,
            axis_x,
            config.acceleration_per_tick,
            config.deceleration_per_tick,
        );
        self.velocity.y = accelerate_axis(
            self.velocity.y,
            axis_y,
            config.acceleration_per_tick,
            config.deceleration_per_tick,
        );
        self.velocity = clamp_magnitude(self.velocity, config.max_speed_per_tick);

        self.position.x = self.position.x.saturating_add(self.velocity.x);
        self.position.y = self.position.y.saturating_add(self.velocity.y);

        clamp_axis(
            &mut self.position.x,
            &mut self.velocity.x,
            config.radius,
            config.world_size.x - config.radius,
        );
        clamp_axis(
            &mut self.position.y,
            &mut self.velocity.y,
            config.radius,
            config.world_size.y - config.radius,
        );

        self.tick = self.tick.wrapping_add(1);
    }
}

fn accelerate_axis(velocity: i32, intent: i32, acceleration: i32, deceleration: i32) -> i32 {
    if intent == 0 {
        if velocity > 0 {
            velocity.saturating_sub(deceleration).max(0)
        } else {
            velocity.saturating_add(deceleration).min(0)
        }
    } else {
        velocity.saturating_add(intent.saturating_mul(acceleration))
    }
}

fn clamp_axis(position: &mut i32, velocity: &mut i32, minimum: i32, maximum: i32) {
    let clamped = (*position).clamp(minimum, maximum);
    if clamped != *position {
        *position = clamped;
        *velocity = 0;
    }
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
            .unwrap_or_else(|_| x.signum() as i32 * maximum),
        i32::try_from(y.saturating_mul(scale) / divisor)
            .unwrap_or_else(|_| y.signum() as i32 * maximum),
    )
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
            let speed_squared =
                i64::from(player.velocity.x).pow(2) + i64::from(player.velocity.y).pow(2);
            assert!(speed_squared <= i64::from(config.max_speed_per_tick).pow(2));
        }
    }

    #[test]
    fn player_cannot_leave_world_bounds() {
        let config = PlayerConfig::default();
        let mut player = PlayerState::centered(&config);

        for _ in 0..10_000 {
            player.step(AxisInput::new(-1, -1), &config);
        }

        assert_eq!(player.position, Vec2::new(config.radius, config.radius));
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
}
