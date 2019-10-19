package algorithm

import (
	"fmt"
	"github.com/gokadin/ai-backpropagation/layer"
	"math"
)

func Learn(network *layer.Collection, inputs [][]float64, expectedOutputs [][]float64, learningRate float64) {
	counter := 0
	for {
		counter++
		err := 0.0
		for inputIndex, input := range inputs {
			forwardPass(network, input)
			backpropagate(network, expectedOutputs[inputIndex])
			err += math.Pow(accumulateError(network.OutputLayer(), expectedOutputs[inputIndex]), 2)
		}

		err /= 2 * float64(len(network.OutputLayer().Nodes()))
		fmt.Println("error:", err)
		if err < 0.0001 {
			fmt.Println("network finished learning after", counter, "iterations")
			break
		}

		updateWeights(network, learningRate)
	}
}

func forwardPass(network *layer.Collection, input []float64) {
	network.InputLayer().ResetInputs()
	network.InputLayer().SetInputs(input)
	network.InputLayer().Activate()
}

func backpropagate(network *layer.Collection, expectedOutput []float64) {
	calculateDeltas(network, expectedOutput)
	accumulateGradients(network)
}

func calculateDeltas(network *layer.Collection, expectedOutput []float64) {
	calculateOutputDeltas(network.OutputLayer(), expectedOutput)
	calculateHiddenDeltas(network)
}

func calculateOutputDeltas(outputLayer *layer.Layer, expectedOutput []float64) {
	for nodeIndex, outputNode := range outputLayer.Nodes() {
		outputNode.SetDelta(outputNode.Output() - expectedOutput[nodeIndex])
	}
}

func calculateHiddenDeltas(network *layer.Collection) {
	// going backwards from the last hidden layer to the first hidden layer
	for i := len(network.Layers) - 2; i > 0; i-- {
		for _, node := range network.Layers[i].Nodes() {
			sumPreviousDeltasAndWeights := 0.0
			for _, connection := range node.Connections() {
				sumPreviousDeltasAndWeights += connection.NextNode().Delta() * connection.Weight()
			}
			node.SetDelta(sumPreviousDeltasAndWeights * network.Layers[i].ActivationDerivative()(node.Input()))
		}
	}
}

func accumulateGradients(network *layer.Collection) {
	// going backwards from the last hidden layer to the input layer
	for i := len(network.Layers) - 2; i >= 0; i-- {
		for _, node := range network.Layers[i].Nodes() {
			for _, connection := range node.Connections() {
				connection.AddGradient(connection.NextNode().Delta() * node.Output())
			}
		}
	}
}

func updateWeights(network *layer.Collection, learningRate float64) {
	for i := 0; i < len(network.Layers)-1; i++ {
		for _, node := range network.Layers[i].Nodes() {
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

func Test(network *layer.Collection, inputs [][]float64) {
	fmt.Println("Results:")

	for _, input := range inputs {
		forwardPass(network, input)

		outputs := make([]float64, len(network.OutputLayer().Nodes()))
		for i, outputNode := range network.OutputLayer().Nodes() {
			outputs[i] = outputNode.Output()
		}
		fmt.Println(input, "=>", outputs)
	}
}
