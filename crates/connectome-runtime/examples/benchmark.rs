use flybrain_connectome_runtime::native::load_package;
use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::time::Instant;

const FEATURE_COUNT: usize = 33;
const STRIKE_THRESHOLD: f32 = 0.5;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let directory = arguments
        .next()
        .ok_or("usage: benchmark PACKAGE_DIRECTORY [TICKS]")?;
    let ticks = arguments
        .next()
        .map_or(Ok(60), |value| value.parse::<usize>())?;
    let state_output = arguments.next();
    if arguments.next().is_some() || ticks == 0 {
        return Err("usage: benchmark PACKAGE_DIRECTORY [TICKS] [STATE_OUTPUT]".into());
    }

    let load_started = Instant::now();
    let mut model = load_package(Path::new(&directory), FEATURE_COUNT, STRIKE_THRESHOLD)?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
    let mut generator = XorShift32::new(20_260_920);
    let mut features = vec![0.0_f32; FEATURE_COUNT];
    let mut durations = Vec::with_capacity(ticks);
    let mut final_decision = None;
    for _ in 0..ticks {
        for feature in &mut features {
            *feature = generator.next_unit() * 2.0 - 1.0;
        }
        let started = Instant::now();
        final_decision = Some(model.step(&features)?);
        durations.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    durations.sort_by(f64::total_cmp);
    let median_ms = percentile(&durations, 50, 100);
    let p95_ms = percentile(&durations, 95, 100);
    let maximum_ms = durations[durations.len() - 1];
    let decision = final_decision.ok_or("no inference result")?;
    let nonzero_state = model.state().iter().filter(|&&value| value != 0.0).count();
    if let Some(path) = state_output {
        let mut bytes = Vec::with_capacity(model.state().len() * 4);
        for &value in model.state() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(path, bytes)?;
    }

    println!(
        concat!(
            "{{\n",
            "  \"status\": \"measured\",\n",
            "  \"backend\": \"rust-native-single-thread\",\n",
            "  \"weight_encoding\": \"{}\",\n",
            "  \"neurons\": {},\n",
            "  \"edges\": {},\n",
            "  \"ticks\": {},\n",
            "  \"load_ms\": {:.3},\n",
            "  \"inference_median_ms\": {:.3},\n",
            "  \"inference_p95_ms\": {:.3},\n",
            "  \"inference_maximum_ms\": {:.3},\n",
            "  \"within_60_hz_budget\": {},\n",
            "  \"final_target\": [{:.9}, {:.9}],\n",
            "  \"final_strike_logit\": {:.9},\n",
            "  \"final_state_nonzero\": {}\n",
            "}}"
        ),
        model.weight_encoding(),
        model.neuron_count(),
        model.edge_count(),
        ticks,
        load_ms,
        median_ms,
        p95_ms,
        maximum_ms,
        p95_ms <= 1_000.0 / 60.0,
        decision.target[0],
        decision.target[1],
        decision.strike_logit,
        nonzero_state,
    );
    Ok(())
}

fn percentile(sorted: &[f64], numerator: usize, denominator: usize) -> f64 {
    let scaled = (sorted.len() - 1) * numerator;
    let index = scaled.div_ceil(denominator);
    sorted[index]
}

struct XorShift32(u32);

impl XorShift32 {
    const fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn next_unit(&mut self) -> f32 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        self.0 = value;
        let bytes = value.to_le_bytes();
        f32::from(u16::from_le_bytes([bytes[2], bytes[3]])) / 65_535.0
    }
}
