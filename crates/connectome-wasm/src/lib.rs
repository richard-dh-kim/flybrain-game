#![forbid(unsafe_code)]

use flybrain_connectome_runtime::{ConnectomeModel, EdgeWeights, PackedArrays};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct QuantizedConnectome {
    model: ConnectomeModel,
}

#[wasm_bindgen]
impl QuantizedConnectome {
    /// Construct a row-wise u16 packed controller from verified typed arrays.
    ///
    /// # Errors
    ///
    /// Returns a JavaScript error when the packed model layout is invalid.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        row_offsets: Vec<u32>,
        column_indices: Vec<u32>,
        edge_weights: Vec<u16>,
        edge_scales: Vec<f32>,
        leak: Vec<f32>,
        sensory_indices: Vec<u32>,
        sensory_feature_ids: Vec<u8>,
        sensory_signs: Vec<i8>,
        motor_indices: Vec<u32>,
        readout_weight: Vec<f32>,
        readout_bias: Vec<f32>,
        feature_count: usize,
        strike_threshold: f32,
    ) -> Result<Self, JsValue> {
        let readout_bias = readout_bias
            .try_into()
            .map_err(|_| JsValue::from_str("readout bias must contain three values"))?;
        let model = ConnectomeModel::new(
            PackedArrays {
                row_offsets,
                column_indices,
                edge_weights: EdgeWeights::RowwiseU16 {
                    values: edge_weights,
                    scales: edge_scales,
                },
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
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok(Self { model })
    }

    /// Advance one recurrent tick and return target x/y, strike logit,
    /// probability, and Boolean decision encoded as 0 or 1.
    ///
    /// # Errors
    ///
    /// Returns a JavaScript error when the feature vector is invalid.
    pub fn step(&mut self, features: &[f32]) -> Result<Vec<f32>, JsValue> {
        let decision = self
            .model
            .step(features)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok(vec![
            decision.target[0],
            decision.target[1],
            decision.strike_logit,
            decision.strike_probability,
            f32::from(u8::from(decision.strike)),
        ])
    }

    pub fn reset(&mut self) {
        self.model.reset();
    }

    #[must_use]
    pub fn state_copy(&self) -> Vec<f32> {
        self.model.state().to_vec()
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn neuron_count(&self) -> usize {
        self.model.neuron_count()
    }

    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.model.edge_count()
    }
}
