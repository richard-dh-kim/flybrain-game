export interface GruModelData {
  inputSize: number;
  hiddenSize: number;
  weightInputHidden: number[][];
  weightHiddenHidden: number[][];
  biasInputHidden: number[];
  biasHiddenHidden: number[];
  weightOutput: number[][];
  biasOutput: number[];
}

export interface GruStep {
  output: readonly number[];
  activity: ArrayLike<number>;
}

export class GruRuntime {
  private hidden: Float64Array;

  constructor(private readonly model: GruModelData) {
    this.hidden = new Float64Array(model.hiddenSize);
  }

  reset(): void {
    this.hidden.fill(0);
  }

  activity(): ArrayLike<number> {
    return this.hidden;
  }

  step(features: ArrayLike<number>): GruStep {
    if (features.length !== this.model.inputSize) {
      throw new Error(`GRU feature mismatch: ${features.length} != ${this.model.inputSize}`);
    }
    const inputProjection = matrixVector(this.model.weightInputHidden, features);
    const hiddenProjection = matrixVector(this.model.weightHiddenHidden, this.hidden);
    const hiddenSize = this.model.hiddenSize;
    const nextHidden = new Float64Array(hiddenSize);

    for (let index = 0; index < hiddenSize; index += 1) {
      const reset = sigmoid(
        value(inputProjection, index)
          + value(this.model.biasInputHidden, index)
          + value(hiddenProjection, index)
          + value(this.model.biasHiddenHidden, index),
      );
      const update = sigmoid(
        value(inputProjection, hiddenSize + index)
          + value(this.model.biasInputHidden, hiddenSize + index)
          + value(hiddenProjection, hiddenSize + index)
          + value(this.model.biasHiddenHidden, hiddenSize + index),
      );
      const candidate = Math.tanh(
        value(inputProjection, 2 * hiddenSize + index)
          + value(this.model.biasInputHidden, 2 * hiddenSize + index)
          + reset * (
            value(hiddenProjection, 2 * hiddenSize + index)
              + value(this.model.biasHiddenHidden, 2 * hiddenSize + index)
          ),
      );
      nextHidden[index] = (1 - update) * candidate + update * value(this.hidden, index);
    }
    this.hidden = nextHidden;

    return {
      output: matrixVector(this.model.weightOutput, this.hidden).map(
        (entry, index) => entry + value(this.model.biasOutput, index),
      ),
      activity: this.hidden,
    };
  }
}

export function sigmoid(input: number): number {
  return 1 / (1 + Math.exp(-input));
}

function matrixVector(matrix: number[][], vector: ArrayLike<number>): number[] {
  return matrix.map((row) => {
    let total = 0;
    for (let index = 0; index < row.length; index += 1) {
      total += value(row, index) * value(vector, index);
    }
    return total;
  });
}

function value(values: ArrayLike<number>, index: number): number {
  const result = values[index];
  if (result === undefined) {
    throw new Error(`missing model value at index ${index}`);
  }
  return result;
}
