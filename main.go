package main

import (
	"github.com/gokadin/ai-backpropagation/algorithm"
	"github.com/gokadin/ai-backpropagation/layer"
	"math/rand"
	"time"
)

const learningRate = 0.01

func main() {
	rand.Seed(time.Now().UTC().UnixNano())

	network := buildNetwork()

	inputs := buildInputs()
	expectedOutputs := buildExpectedOutputs()

	algorithm.Learn(network, inputs, expectedOutputs, learningRate)

	algorithm.Test(network, inputs)
}

func buildNetwork() *layer.Collection {
	inputLayer := layer.NewLayer(2, layer.FunctionIdentity)
	hiddenLayer := layer.NewLayer(2, layer.FunctionSigmoid)
	outputLayer := layer.NewOutputLayer(1, layer.FunctionIdentity)
	inputLayer.ConnectTo(hiddenLayer)
	hiddenLayer.ConnectTo(outputLayer)

	network := layer.NewCollection()
	network.Layers = append(network.Layers, inputLayer)
	network.Layers = append(network.Layers, hiddenLayer)
	network.Layers = append(network.Layers, outputLayer)

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

