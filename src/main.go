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

	test(network, inputs)
}

func buildNetwork() *layerCollection {
	inputLayer := layer.NewLayer(2, layer.FunctionIdentity)
	hiddenLayer := layer.NewLayer(2, layer.FunctionSigmoid)
	outputLayer := layer.NewOutputLayer(1, layer.FunctionIdentity)
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
	counter := 0
	for {
		counter++
		err := 0.0
		for inputIndex, input := range inputs {
			forwardPass(network, input)
			backpropagate(network, expectedOutputs[inputIndex])
			err += math.Pow(accumulateError(network.outputLayer(), expectedOutputs[inputIndex]), 2)
		}

		err /= 2 * float64(len(network.outputLayer().Nodes()))
		fmt.Println("error:", err)
		if err < 0.0001 {
			fmt.Println("network finished learning after", counter, "iterations")
            break
		}

		updateWeights(network)
	}
}

func forwardPass(network *layerCollection, input []float64) {
	network.inputLayer().ResetInputs()
	network.inputLayer().SetInputs(input)
	network.inputLayer().Activate()
}

func backpropagate(network *layerCollection, expectedOutput []float64) {
	calculateDeltas(network, expectedOutput)
	accumulateGradients(network)
}

func calculateDeltas(network *layerCollection, expectedOutput []float64) {
	calculateOutputDeltas(network.outputLayer(), expectedOutput)
	calculateHiddenDeltas(network)
}

func calculateOutputDeltas(outputLayer *layer.Layer, expectedOutput []float64) {
	for nodeIndex, outputNode := range outputLayer.Nodes() {
		outputNode.SetDelta(outputNode.Output() - expectedOutput[nodeIndex])
	}
}

func calculateHiddenDeltas(network *layerCollection) {
	// going backwards from the last hidden layer to the first hidden layer
	for i := len(network.layers) - 2; i > 0; i-- {
		for _, node := range network.layers[i].Nodes() {
			sumPreviousDeltasAndWeights := 0.0
			for _, connection := range node.Connections() {
				sumPreviousDeltasAndWeights += connection.NextNode().Delta() * connection.Weight()
			}
			node.SetDelta(sumPreviousDeltasAndWeights * network.layers[i].ActivationDerivative()(node.Input()))
		}
	}
}

func accumulateGradients(network *layerCollection) {
	// going backwards from the last hidden layer to the input layer
	for i := len(network.layers) - 2; i >= 0; i-- {
		for _, node := range network.layers[i].Nodes() {
			for _, connection := range node.Connections() {
				connection.AddGradient(connection.NextNode().Delta() * node.Output())
			}
		}
	}
}

func updateWeights(network *layerCollection) {
	for i := 0; i < len(network.layers)-1; i++ {
		for _, node := range network.layers[i].Nodes() {
			for _, connection := range node.Connections() {
				connection.UpdateWeight(learningRate)
			}
		}
	}
}

func accumulateError(outputLayer *layer.Layer, expectedOutput []float64) float64 {
	err := 0.0
	for outputNodeIndex, outputNode := range outputLayer.Nodes() {
		err += outputNode.Output() - expectedOutput[outputNodeIndex]
	}
	return err
}

func test(network *layerCollection, inputs [][]float64) {
	fmt.Println("Results:")

    for _, input := range inputs {
    	forwardPass(network, input)

    	outputs := make([]float64, len(network.outputLayer().Nodes()))
    	for i, outputNode := range network.outputLayer().Nodes() {
    		outputs[i] = outputNode.Output()
		}
    	fmt.Println(input, "=>", outputs)
	}
}
