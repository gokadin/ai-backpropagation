package main

import (
	"fmt"
	"github.com/gokadin/ann-core/layer"
	"math"
	"math/rand"
	"time"
)

const learningRate = 0.01

func main() {
	rand.Seed(time.Now().UTC().UnixNano())

	network := buildNetwork()

	inputs := buildInputs()
	expectedOutputs := buildExpectedOutputs()

	learn(network, inputs, expectedOutputs)
}

func buildNetwork() *layerCollection {
	inputLayer := layer.NewLayer(2, layer.FunctionIdentity)
	hiddenLayer := layer.NewLayer(2, layer.FunctionSigmoid)
	outputLayer := layer.NewLayer(1, layer.FunctionIdentity)
	inputLayer.ConnectTo(hiddenLayer)
	hiddenLayer.ConnectTo(outputLayer)

	network := newLayerCollection()
	network.layers = append(network.layers, inputLayer)
	network.layers = append(network.layers, hiddenLayer)
	network.layers = append(network.layers, outputLayer)

	return network
}

func buildInputs() [][]float64 {
	inputs := make([][]float64, 4)

	inputs[0] = []float64{1.0, 0.0}
	inputs[1] = []float64{1.0, 1.0}
	inputs[2] = []float64{0.0, 1.0}
	inputs[3] = []float64{0.0, 0.0}

	return inputs
}

func buildExpectedOutputs() [][]float64 {
	outputs := make([][]float64, 4)

	outputs[0] = []float64{1.0}
	outputs[1] = []float64{0.0}
	outputs[2] = []float64{1.0}
	outputs[3] = []float64{0.0}

	return outputs
}

func learn(network *layerCollection, inputs [][]float64, expectedOutputs [][]float64) {
	learn := true
	for learn {
		for inputIndex, input := range inputs {
			forwardPass(network, input)
			backpropagate(network, expectedOutputs[inputIndex])
		}

		err := calculateError(network.outputLayer())
		fmt.Println("error:", err)
		if err < 0.01 {
			learn = false
			fmt.Println("Network finished learning.")
		}

		// 4. update weights
		updateWeights(network)
	}
}

func forwardPass(network *layerCollection, input []float64) {
	network.inputLayer().ResetValues()
	network.inputLayer().SetValues(input)
	network.inputLayer().Activate()
}

func backpropagate(network *layerCollection, expectedOutput []float64) {
	// 1. accumulate deltas on output layer
	accumulateOutputDeltas(network.outputLayer(), expectedOutput)

	// 2. accumulate deltas on hidden layers
	accumulateHiddenDeltas(network)

	// 3. accumulate gradients
	accumulateGradients(network)
}

func accumulateOutputDeltas(outputLayer *layer.Layer, expectedOutput []float64) {
	for nodeIndex, outputNode := range outputLayer.Nodes() {
		outputNode.AddDelta(outputNode.Value() - expectedOutput[nodeIndex])
	}
}

func accumulateHiddenDeltas(network *layerCollection) {
	// going backwards from the last hidden layer to the first hidden layer
	for i := len(network.layers) - 2; i > 0; i-- {
		for _, node := range network.layers[i].Nodes() {
			sumPreviousDeltasAndWeights := 0.0
			for _, connection := range node.Connections() {
				sumPreviousDeltasAndWeights += connection.NextNode().Delta() * connection.Weight()
			}
			node.AddDelta(sumPreviousDeltasAndWeights * network.layers[i].ActivationDerivative()(node.Value()))
		}
	}
}

func accumulateGradients(network *layerCollection) {
	// going backwards from the last hidden layer to the input layer
	for i := len(network.layers) - 2; i >= 0; i-- {
		for _, node := range network.layers[i].Nodes() {
			for _, connection := range node.Connections() {
				connection.AddGradient(connection.NextNode().Delta() * node.Value())
			}
		}
	}
}

func updateWeights(network *layerCollection) {
	for i := 0; i < len(network.layers)-1; i++ {
		for _, node := range network.layers[i].Nodes() {
			for _, connection := range node.Connections() {
				connection.UpdateWeight(learningRate)
				connection.ResetGradient()
			}
			node.ResetDelta()
		}
	}
}

func calculateError(outputLayer *layer.Layer) float64 {
	err := 0.0
	for _, node := range outputLayer.Nodes() {
		err += math.Pow(node.Delta(), 2)
	}
	return err / 2
}
