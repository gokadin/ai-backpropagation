package main

import (
	"github.com/gokadin/ann-core/layer"
	"math"
	"math/rand"
	"testing"
)

func buildSimpleTestNetwork(inputCount, hiddenCount, outputCount int, activationFunction string) *layerCollection {
	inputLayer := layer.NewLayer(inputCount, layer.FunctionIdentity)
	hiddenLayer := layer.NewLayer(hiddenCount, activationFunction)
	outputLayer := layer.NewLayer(outputCount, layer.FunctionIdentity)
	inputLayer.ConnectTo(hiddenLayer)
	hiddenLayer.ConnectTo(outputLayer)

	network := newLayerCollection()
	network.layers = append(network.layers, inputLayer)
	network.layers = append(network.layers, hiddenLayer)
	network.layers = append(network.layers, outputLayer)

	return network
}

func generateSimpleData(inputCount, outputCount, associations int) ([][]float64, [][]float64) {
	inputs := make([][]float64, associations)
	for i := 0; i < associations; i++ {
		input := make([]float64, inputCount)
		for j := 0; j < inputCount; j++ {
			input[j] = rand.Float64()
		}
		inputs[i] = input
	}

	outputs := make([][]float64, associations)
	for i := 0; i < associations; i++ {
		output := make([]float64, inputCount)
		for j := 0; j < outputCount; j++ {
			output[j] = rand.Float64()
		}
		outputs[i] = output
	}

	return inputs, outputs
}

func Test_forwardPass_setsCorrectInputValues(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, _ := generateSimpleData(1, 1, 1)

	net.inputLayer().SetValues(inputs[0])

	for i, value := range inputs[0] {
		if net.inputLayer().Nodes()[i].Value() != value {
			t.Fatalf("Expected %f, got %f", value, net.inputLayer().Nodes()[i].Value())
		}
	}
}

func Test_forwardPass_outputIsCorrect(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, _ := generateSimpleData(1, 1, 1)

	net.inputLayer().ResetValues()
	net.inputLayer().SetValues(inputs[0])
	net.inputLayer().Activate()

	expected := inputs[0][0] * net.inputLayer().Nodes()[0].Connection(0).Weight()
	expected = expected * net.layers[1].Nodes()[0].Connection(0).Weight()
	if net.outputLayer().Nodes()[0].Value() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.outputLayer().Nodes()[0].Value())
	}
}

func Test_forwardPass_outputIsCorrectWithSigmoidActivationHiddenNode(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionSigmoid)
	inputs, _ := generateSimpleData(1, 1, 1)

	net.inputLayer().ResetValues()
	net.inputLayer().SetValues(inputs[0])
	net.inputLayer().Activate()

	expected := inputs[0][0] * net.inputLayer().Nodes()[0].Connection(0).Weight()
	expected = 1 / (1 + math.Pow(math.E, -expected))
	expected = expected * net.layers[1].Nodes()[0].Connection(0).Weight()
	if net.outputLayer().Nodes()[0].Value() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.outputLayer().Nodes()[0].Value())
	}
}

func Test_forwardPass_outputIsCorrectWithTwoInputNodes(t *testing.T) {
	net := buildSimpleTestNetwork(2, 1, 1, layer.FunctionIdentity)
	inputs, _ := generateSimpleData(2, 1, 1)

	net.inputLayer().ResetValues()
	net.inputLayer().SetValues(inputs[0])
	net.inputLayer().Activate()

	h1 := inputs[0][0] * net.inputLayer().Nodes()[0].Connection(0).Weight()
	h1 += inputs[0][1] * net.inputLayer().Nodes()[1].Connection(0).Weight()
	expected := h1 * net.layers[1].Nodes()[0].Connection(0).Weight()
	if net.outputLayer().Nodes()[0].Value() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.outputLayer().Nodes()[0].Value())
	}
}

func Test_forwardPass_outputIsCorrectWithTwoInputNodesAndTwoHiddenNodes(t *testing.T) {
	net := buildSimpleTestNetwork(2, 2, 1, layer.FunctionIdentity)
	inputs, _ := generateSimpleData(2, 2, 1)

	net.inputLayer().ResetValues()
	net.inputLayer().SetValues(inputs[0])
	net.inputLayer().Activate()

	h1 := inputs[0][0] * net.inputLayer().Nodes()[0].Connection(0).Weight()
	h1 += inputs[0][1] * net.inputLayer().Nodes()[1].Connection(0).Weight()
	h2 := inputs[0][0] * net.inputLayer().Nodes()[0].Connection(1).Weight()
	h2 += inputs[0][1] * net.inputLayer().Nodes()[1].Connection(1).Weight()
	expected := h1 * net.layers[1].Nodes()[0].Connection(0).Weight()
	expected += h2 * net.layers[1].Nodes()[1].Connection(0).Weight()
	if net.outputLayer().Nodes()[0].Value() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.outputLayer().Nodes()[0].Value())
	}
}
