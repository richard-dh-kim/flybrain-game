#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt::{Display, Formatter};

pub const PACKED_FORMAT_V1: u16 = 1;
pub const PACKED_FORMAT_V2: u16 = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelError(String);

impl ModelError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for ModelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ModelError {}

#[derive(Debug)]
pub enum EdgeWeights {
    Float32(Vec<f32>),
    RowwiseU16 { values: Vec<u16>, scales: Vec<f32> },
}

impl EdgeWeights {
    fn len(&self) -> usize {
        match self {
            Self::Float32(values) => values.len(),
            Self::RowwiseU16 { values, .. } => values.len(),
        }
    }

    fn finite(&self) -> bool {
        match self {
            Self::Float32(values) => values.iter().all(|value| value.is_finite()),
            Self::RowwiseU16 { scales, .. } => scales
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
        }
    }

    fn encoding(&self) -> &'static str {
        match self {
            Self::Float32(_) => "float32",
            Self::RowwiseU16 { .. } => "rowwise-u16",
        }
    }
}

#[derive(Debug)]
pub struct PackedArrays {
    pub row_offsets: Vec<u32>,
    pub column_indices: Vec<u32>,
    pub edge_weights: EdgeWeights,
    pub leak: Vec<f32>,
    pub sensory_indices: Vec<u32>,
    pub sensory_feature_ids: Vec<u8>,
    pub sensory_signs: Vec<i8>,
    pub motor_indices: Vec<u32>,
    pub readout_weight: Vec<f32>,
    pub readout_bias: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decision {
    pub target: [f32; 2],
    pub strike_logit: f32,
    pub strike_probability: f32,
    pub strike: bool,
}

#[derive(Debug)]
pub struct ConnectomeModel {
    arrays: PackedArrays,
    feature_count: usize,
    strike_threshold: f32,
    sensory_feature_by_neuron: Vec<u8>,
    sensory_sign_by_neuron: Vec<i8>,
    state: Vec<f32>,
    next_state: Vec<f32>,
}

impl ConnectomeModel {
    /// Validate and initialize an inference-only packed model.
    ///
    /// # Errors
    ///
    /// Returns an error when arrays disagree with the declared dimensions,
    /// contain invalid indices or values, or describe an invalid interface.
    pub fn new(
        arrays: PackedArrays,
        feature_count: usize,
        strike_threshold: f32,
    ) -> Result<Self, ModelError> {
        if arrays.row_offsets.is_empty() {
            return Err(ModelError::new("row offsets cannot be empty"));
        }
        let neuron_count = arrays.row_offsets.len() - 1;
        let edge_count = arrays.column_indices.len();
        if arrays.row_offsets[0] != 0 {
            return Err(ModelError::new("row offsets must start at zero"));
        }
        if arrays.row_offsets.windows(2).any(|pair| pair[1] < pair[0]) {
            return Err(ModelError::new("row offsets must be monotonic"));
        }
        if usize::try_from(arrays.row_offsets[neuron_count]).ok() != Some(edge_count) {
            return Err(ModelError::new("row offsets must end at the edge count"));
        }
        if arrays.edge_weights.len() != edge_count {
            return Err(ModelError::new("edge index and weight counts differ"));
        }
        if arrays.leak.len() != neuron_count {
            return Err(ModelError::new("leak count differs from neuron count"));
        }
        if let EdgeWeights::RowwiseU16 { scales, .. } = &arrays.edge_weights
            && scales.len() != neuron_count
        {
            return Err(ModelError::new(
                "row-wise weight scale count differs from neuron count",
            ));
        }
        if arrays
            .column_indices
            .iter()
            .any(|&index| usize::try_from(index).map_or(true, |value| value >= neuron_count))
        {
            return Err(ModelError::new("column index exceeds neuron count"));
        }
        if arrays.sensory_indices.len() != arrays.sensory_feature_ids.len()
            || arrays.sensory_indices.len() != arrays.sensory_signs.len()
        {
            return Err(ModelError::new("sensory array lengths differ"));
        }
        if arrays
            .motor_indices
            .iter()
            .any(|&index| usize::try_from(index).map_or(true, |value| value >= neuron_count))
        {
            return Err(ModelError::new("motor index exceeds neuron count"));
        }
        if arrays.readout_weight.len() != 3 * arrays.motor_indices.len() {
            return Err(ModelError::new(
                "readout weights do not match the motor population",
            ));
        }
        if !strike_threshold.is_finite() || !(0.0..=1.0).contains(&strike_threshold) {
            return Err(ModelError::new("strike threshold must be within [0, 1]"));
        }
        if arrays
            .leak
            .iter()
            .chain(arrays.readout_weight.iter())
            .chain(arrays.readout_bias.iter())
            .any(|value| !value.is_finite())
            || !arrays.edge_weights.finite()
        {
            return Err(ModelError::new("model arrays contain a non-finite value"));
        }

        let mut sensory_feature_by_neuron = vec![u8::MAX; neuron_count];
        let mut sensory_sign_by_neuron = vec![0; neuron_count];
        for ((&neuron, &feature), &sign) in arrays
            .sensory_indices
            .iter()
            .zip(&arrays.sensory_feature_ids)
            .zip(&arrays.sensory_signs)
        {
            let neuron = usize::try_from(neuron)
                .map_err(|_| ModelError::new("sensory index cannot fit usize"))?;
            if neuron >= neuron_count {
                return Err(ModelError::new("sensory index exceeds neuron count"));
            }
            if usize::from(feature) >= feature_count {
                return Err(ModelError::new("sensory feature exceeds feature count"));
            }
            if !matches!(sign, -1 | 1) {
                return Err(ModelError::new("sensory sign must be -1 or 1"));
            }
            if sensory_feature_by_neuron[neuron] != u8::MAX {
                return Err(ModelError::new("sensory neuron is assigned twice"));
            }
            sensory_feature_by_neuron[neuron] = feature;
            sensory_sign_by_neuron[neuron] = sign;
        }

        Ok(Self {
            arrays,
            feature_count,
            strike_threshold,
            sensory_feature_by_neuron,
            sensory_sign_by_neuron,
            state: vec![0.0; neuron_count],
            next_state: vec![0.0; neuron_count],
        })
    }

    #[must_use]
    pub fn neuron_count(&self) -> usize {
        self.state.len()
    }

    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.arrays.edge_weights.len()
    }

    #[must_use]
    pub fn weight_encoding(&self) -> &'static str {
        self.arrays.edge_weights.encoding()
    }

    #[must_use]
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    #[must_use]
    pub fn state(&self) -> &[f32] {
        &self.state
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.next_state.fill(0.0);
    }

    /// Advance the carried recurrent state by one game tick.
    ///
    /// # Errors
    ///
    /// Returns an error when the feature count is wrong, an input is not
    /// finite, or a packed index cannot be represented by this target.
    pub fn step(&mut self, features: &[f32]) -> Result<Decision, ModelError> {
        if features.len() != self.feature_count {
            return Err(ModelError::new(format!(
                "expected {} features, received {}",
                self.feature_count,
                features.len()
            )));
        }
        if features.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::new("features contain a non-finite value"));
        }

        for row in 0..self.state.len() {
            let start = usize::try_from(self.arrays.row_offsets[row])
                .map_err(|_| ModelError::new("row offset cannot fit usize"))?;
            let end = usize::try_from(self.arrays.row_offsets[row + 1])
                .map_err(|_| ModelError::new("row offset cannot fit usize"))?;
            let mut signal = match &self.arrays.edge_weights {
                EdgeWeights::Float32(weights) => {
                    let mut sum = 0.0_f32;
                    for (&weight, &column) in weights[start..end]
                        .iter()
                        .zip(&self.arrays.column_indices[start..end])
                    {
                        let source = usize::try_from(column)
                            .map_err(|_| ModelError::new("column index cannot fit usize"))?;
                        sum += weight * self.state[source];
                    }
                    sum
                }
                EdgeWeights::RowwiseU16 { values, scales } => {
                    let mut integer_weighted_sum = 0.0_f32;
                    for (&weight, &column) in values[start..end]
                        .iter()
                        .zip(&self.arrays.column_indices[start..end])
                    {
                        let source = usize::try_from(column)
                            .map_err(|_| ModelError::new("column index cannot fit usize"))?;
                        integer_weighted_sum += f32::from(weight) * self.state[source];
                    }
                    integer_weighted_sum * scales[row]
                }
            };
            let feature = self.sensory_feature_by_neuron[row];
            if feature != u8::MAX {
                signal +=
                    features[usize::from(feature)] * f32::from(self.sensory_sign_by_neuron[row]);
            }
            let leak = self.arrays.leak[row];
            self.next_state[row] = (1.0 - leak) * self.state[row] + leak * signal.tanh();
        }
        std::mem::swap(&mut self.state, &mut self.next_state);

        let mut raw = self.arrays.readout_bias;
        let motor_count = self.arrays.motor_indices.len();
        for (motor_offset, &neuron) in self.arrays.motor_indices.iter().enumerate() {
            let activity = self.state[usize::try_from(neuron)
                .map_err(|_| ModelError::new("motor index cannot fit usize"))?];
            for (output, value) in raw.iter_mut().enumerate() {
                *value +=
                    self.arrays.readout_weight[output * motor_count + motor_offset] * activity;
            }
        }
        let strike_probability = sigmoid(raw[2]);
        Ok(Decision {
            target: [sigmoid(raw[0]), sigmoid(raw[1])],
            strike_logit: raw[2],
            strike_probability,
            strike: strike_probability >= self.strike_threshold,
        })
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(not(target_arch = "wasm32"))]
pub mod native {
    use super::{ConnectomeModel, EdgeWeights, ModelError, PackedArrays};
    use std::fs;
    use std::path::Path;

    /// Load the fixed raw-array names used by packed formats v1 and v2.
    ///
    /// The caller must verify the manifest and per-file checksums before this
    /// loader is used. The constructor still validates all shapes and indices.
    ///
    /// # Errors
    ///
    /// Returns an error when a file cannot be read or the model layout is
    /// invalid.
    pub fn load_package(
        directory: &Path,
        feature_count: usize,
        strike_threshold: f32,
    ) -> Result<ConnectomeModel, ModelError> {
        let row_offsets = read_u32(&directory.join("row_offsets.u32.bin"))?;
        let column_indices = read_u32(&directory.join("column_indices.u32.bin"))?;
        let float32_weights = directory.join("edge_weights.f32.bin");
        let u16_weights = directory.join("edge_weights.u16.bin");
        let edge_weights = if float32_weights.is_file() {
            EdgeWeights::Float32(read_f32(&float32_weights)?)
        } else if u16_weights.is_file() {
            EdgeWeights::RowwiseU16 {
                values: read_u16(&u16_weights)?,
                scales: read_f32(&directory.join("edge_scales.f32.bin"))?,
            }
        } else {
            return Err(ModelError::new("package has no recognized edge weights"));
        };
        let leak = read_f32(&directory.join("leak.f32.bin"))?;
        let sensory_indices = read_u32(&directory.join("sensory_indices.u32.bin"))?;
        let sensory_feature_ids = read_bytes(&directory.join("sensory_feature_ids.u8.bin"))?;
        let sensory_signs = read_bytes(&directory.join("sensory_signs.i8.bin"))?
            .into_iter()
            .map(|value| i8::from_ne_bytes([value]))
            .collect();
        let motor_indices = read_u32(&directory.join("motor_indices.u32.bin"))?;
        let readout_weight = read_f32(&directory.join("readout_weight.f32.bin"))?;
        let readout_bias_values = read_f32(&directory.join("readout_bias.f32.bin"))?;
        let readout_bias: [f32; 3] = readout_bias_values
            .try_into()
            .map_err(|_| ModelError::new("readout bias must have three values"))?;
        ConnectomeModel::new(
            PackedArrays {
                row_offsets,
                column_indices,
                edge_weights,
                leak,
                sensory_indices,
                sensory_feature_ids,
                sensory_signs,
                motor_indices,
                readout_weight,
                readout_bias,
            },
            feature_count,
            strike_threshold,
        )
    }

    fn read_bytes(path: &Path) -> Result<Vec<u8>, ModelError> {
        fs::read(path).map_err(|error| ModelError::new(format!("{}: {error}", path.display())))
    }

    fn read_u32(path: &Path) -> Result<Vec<u32>, ModelError> {
        let bytes = read_bytes(path)?;
        if bytes.len() % 4 != 0 {
            return Err(ModelError::new(format!(
                "{} does not contain whole u32 values",
                path.display()
            )));
        }
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(u32::from_le_bytes)
            .collect())
    }

    fn read_u16(path: &Path) -> Result<Vec<u16>, ModelError> {
        let bytes = read_bytes(path)?;
        if bytes.len() % 2 != 0 {
            return Err(ModelError::new(format!(
                "{} does not contain whole u16 values",
                path.display()
            )));
        }
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .copied()
            .map(u16::from_le_bytes)
            .collect())
    }

    fn read_f32(path: &Path) -> Result<Vec<f32>, ModelError> {
        read_u32(path).map(|values| values.into_iter().map(f32::from_bits).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::{ConnectomeModel, EdgeWeights, PackedArrays};

    fn tiny_model() -> ConnectomeModel {
        ConnectomeModel::new(
            PackedArrays {
                row_offsets: vec![0, 0, 1, 2, 2],
                column_indices: vec![0, 1],
                edge_weights: EdgeWeights::Float32(vec![0.4, 0.7]),
                leak: vec![0.5, 0.4, 0.3, 0.2],
                sensory_indices: vec![0],
                sensory_feature_ids: vec![1],
                sensory_signs: vec![-1],
                motor_indices: vec![2, 3],
                readout_weight: vec![1.0, 0.0, 0.0, 1.0, 0.5, -0.25],
                readout_bias: [0.1, -0.2, 0.3],
            },
            3,
            0.5,
        )
        .unwrap()
    }

    #[test]
    fn step_matches_direct_recurrence() {
        let mut model = tiny_model();
        model.state = vec![0.8, -0.3, 0.2, -0.1];
        let decision = model.step(&[0.2, 0.6, -0.4]).unwrap();

        let signal = [-0.6_f32, 0.4 * 0.8, 0.7 * -0.3, 0.0];
        let previous = [0.8_f32, -0.3, 0.2, -0.1];
        let leak = [0.5_f32, 0.4, 0.3, 0.2];
        let expected = std::array::from_fn::<_, 4, _>(|index| {
            (1.0 - leak[index]) * previous[index] + leak[index] * signal[index].tanh()
        });
        for (actual, expected) in model.state().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-7);
        }
        let raw = [
            expected[2] + 0.1,
            expected[3] - 0.2,
            0.5 * expected[2] - 0.25 * expected[3] + 0.3,
        ];
        assert!((decision.target[0] - 1.0 / (1.0 + (-raw[0]).exp())).abs() < 1e-7);
        assert!((decision.target[1] - 1.0 / (1.0 + (-raw[1]).exp())).abs() < 1e-7);
        assert!((decision.strike_logit - raw[2]).abs() < 1e-7);
    }

    #[test]
    fn rejects_out_of_range_topology() {
        let mut arrays = tiny_model().arrays;
        arrays.column_indices[0] = 4;
        let error = ConnectomeModel::new(arrays, 3, 0.5).unwrap_err();
        assert_eq!(error.to_string(), "column index exceeds neuron count");
    }

    #[test]
    fn reset_clears_recurrent_state() {
        let mut model = tiny_model();
        model.step(&[0.2, 0.6, -0.4]).unwrap();
        assert!(model.state().iter().any(|&value| value != 0.0));
        model.reset();
        assert!(model.state().iter().all(|&value| value == 0.0));
    }

    #[test]
    fn rowwise_u16_matches_representable_float_weights() {
        let mut exact = tiny_model();
        let mut arrays = tiny_model().arrays;
        arrays.edge_weights = EdgeWeights::RowwiseU16 {
            values: vec![4, 7],
            scales: vec![0.0, 0.1, 0.1, 0.0],
        };
        let mut quantized = ConnectomeModel::new(arrays, 3, 0.5).unwrap();
        exact.state = vec![0.8, -0.3, 0.2, -0.1];
        quantized.state.clone_from(&exact.state);

        let exact_decision = exact.step(&[0.2, 0.6, -0.4]).unwrap();
        let quantized_decision = quantized.step(&[0.2, 0.6, -0.4]).unwrap();

        for (actual, expected) in quantized.state().iter().zip(exact.state()) {
            assert!((actual - expected).abs() < 1e-7);
        }
        assert!((quantized_decision.target[0] - exact_decision.target[0]).abs() < 1e-7);
        assert!((quantized_decision.target[1] - exact_decision.target[1]).abs() < 1e-7);
        assert!((quantized_decision.strike_logit - exact_decision.strike_logit).abs() < 1e-7);
    }
}
