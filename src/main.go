package main

import (
	"github.com/gokadin/ann-core/layer"
	"math/rand"
	"time"
)

const learningRate = 0.01

func main() {
	rand.Seed(time.Now().UTC().UnixNano())

	network := buildNetwork()

	inputs := buildInputs()
	expectedOutputs := buildExpectedOutputs()
	_ = expectedOutputs

	learn(network, inputs, expectedOutputs)
}

func buildNetwork() *network {
	inputLayer := layer.NewLayer(2, layer.Identity)
	hiddenLayer := layer.NewLayer(2, layer.Sigmoid)
	outputLayer := layer.NewLayer(1, layer.Identity)
	inputLayer.ConnectTo(hiddenLayer)
	hiddenLayer.ConnectTo(outputLayer)

	network := newNetwork()
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

func learn(network *network, inputs [][]float64, expectedOutputs [][]float64) {
	learn := true
	for learn {
		for inputIndex, input := range inputs {
			forwardPass(network, input)
            backpropagate(network, expectedOutputs[inputIndex])
		}
	}
}

func forwardPass(network *network, input []float64) {
	network.inputLayer().ResetValues()
	network.inputLayer().SetValues(input)
	network.inputLayer().Activate()
}

func backpropagate(network *network, expectedOutput []float64) {
	// set delta on output layer

	for nodeIndex, outputNode := range network.outputLayer().Nodes() {
        delta := outputNode.Value() - expectedOutput[nodeIndex]
        outputNode.AddDelta(delta)
	}

	// set delta on hidden layers

	// going backwards from the last hidden layer to the first hidden layer
	for i := len(network.layers) - 2; i > 0; i-- {
        previousLayer := network.layers[i + 1]
	}
}
