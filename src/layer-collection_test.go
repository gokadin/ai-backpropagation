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

	net.inputLayer().SetInputs(inputs[0])

	for i, value := range inputs[0] {
		if net.inputLayer().Nodes()[i].Input() != value {
			t.Fatalf("Expected %f, got %f", value, net.inputLayer().Nodes()[i].Input())
		}
	}
}

func Test_forwardPass_outputIsCorrect(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, _ := generateSimpleData(1, 1, 1)

	net.inputLayer().ResetInputs()
	net.inputLayer().SetInputs(inputs[0])
	net.inputLayer().Activate()

	expected := inputs[0][0] * net.inputLayer().Nodes()[0].Connection(0).Weight()
	expected = expected * net.layers[1].Nodes()[0].Connection(0).Weight()
	if net.outputLayer().Nodes()[0].Output() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.outputLayer().Nodes()[0].Output())
	}
}

func Test_forwardPass_outputIsCorrectWithSigmoidActivationHiddenNode(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionSigmoid)
	inputs, _ := generateSimpleData(1, 1, 1)

	net.inputLayer().ResetInputs()
	net.inputLayer().SetInputs(inputs[0])
	net.inputLayer().Activate()

	expected := inputs[0][0] * net.inputLayer().Nodes()[0].Connection(0).Weight()
	expected = 1 / (1 + math.Pow(math.E, -expected))
	expected = expected * net.layers[1].Nodes()[0].Connection(0).Weight()
	if net.outputLayer().Nodes()[0].Output() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.outputLayer().Nodes()[0].Output())
	}
}

func Test_forwardPass_outputIsCorrectWithTwoInputNodes(t *testing.T) {
	net := buildSimpleTestNetwork(2, 1, 1, layer.FunctionIdentity)
	inputs, _ := generateSimpleData(2, 1, 1)

	net.inputLayer().ResetInputs()
	net.inputLayer().SetInputs(inputs[0])
	net.inputLayer().Activate()

	h1 := inputs[0][0] * net.inputLayer().Nodes()[0].Connection(0).Weight()
	h1 += inputs[0][1] * net.inputLayer().Nodes()[1].Connection(0).Weight()
	expected := h1 * net.layers[1].Nodes()[0].Connection(0).Weight()
	if net.outputLayer().Nodes()[0].Output() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.outputLayer().Nodes()[0].Output())
	}
}

func Test_forwardPass_outputIsCorrectWithTwoInputNodesAndTwoHiddenNodes(t *testing.T) {
	net := buildSimpleTestNetwork(2, 2, 1, layer.FunctionIdentity)
	inputs, _ := generateSimpleData(2, 2, 1)

	net.inputLayer().ResetInputs()
	net.inputLayer().SetInputs(inputs[0])
	net.inputLayer().Activate()

	h1 := inputs[0][0] * net.inputLayer().Nodes()[0].Connection(0).Weight()
	h1 += inputs[0][1] * net.inputLayer().Nodes()[1].Connection(0).Weight()
	h2 := inputs[0][0] * net.inputLayer().Nodes()[0].Connection(1).Weight()
	h2 += inputs[0][1] * net.inputLayer().Nodes()[1].Connection(1).Weight()
	expected := h1 * net.layers[1].Nodes()[0].Connection(0).Weight()
	expected += h2 * net.layers[1].Nodes()[1].Connection(0).Weight()
	if net.outputLayer().Nodes()[0].Output() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.outputLayer().Nodes()[0].Output())
	}
}

// accumulate output deltas

func Test_accumulateOutputDeltas_setsTheCorrectValues(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, outputs := generateSimpleData(1, 1, 1)
	net.inputLayer().ResetInputs()
	net.inputLayer().SetInputs(inputs[0])
	net.inputLayer().Activate()

	calculateOutputDeltas(net.outputLayer(), outputs[0])

	expected := net.outputLayer().Nodes()[0].Output() - outputs[0][0]
	if net.outputLayer().Nodes()[0].Delta() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.outputLayer().Nodes()[0].Delta())
	}
}

func Test_accumulateOutputDeltas_accumulatesTheCorrectValuesWithTwoAssociations(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, outputs := generateSimpleData(1, 1, 2)

	expected := 0.0
	for i := 0; i < len(inputs); i++ {
		net.inputLayer().ResetInputs()
		net.inputLayer().SetInputs(inputs[i])
		net.inputLayer().Activate()

		calculateOutputDeltas(net.outputLayer(), outputs[i])
		expected += net.outputLayer().Nodes()[0].Output() - outputs[i][0]
	}

	if net.outputLayer().Nodes()[0].Delta() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.outputLayer().Nodes()[0].Delta())
	}
}

func Test_accumulateOutputDeltas_accumulatesTheCorrectValuesWithTwoAssociationsAndMultipleOutputNodes(t *testing.T) {
	net := buildSimpleTestNetwork(2, 2, 2, layer.FunctionIdentity)
	inputs, outputs := generateSimpleData(2, 2, 2)

	expected := make([]float64, len(outputs))
	for i := 0; i < len(inputs); i++ {
		net.inputLayer().ResetInputs()
		net.inputLayer().SetInputs(inputs[i])
		net.inputLayer().Activate()

		calculateOutputDeltas(net.outputLayer(), outputs[i])
		for outputNodeIndex, outputNode := range net.outputLayer().Nodes() {
			expected[outputNodeIndex] += outputNode.Output() - outputs[i][outputNodeIndex]
		}
	}

	for outputNodeIndex, outputNode := range net.outputLayer().Nodes() {
		if outputNode.Delta() != expected[outputNodeIndex] {
			t.Fatalf("Expected %f, got %f", expected[outputNodeIndex], outputNode.Delta())
		}
	}
}

// hidden deltas

func Test_accumulateDeltas_setsTheCorrectValues(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, outputs := generateSimpleData(1, 1, 1)
	net.inputLayer().ResetInputs()
	net.inputLayer().SetInputs(inputs[0])
	net.inputLayer().Activate()
	calculateOutputDeltas(net.outputLayer(), outputs[0])

	calculateHiddenDeltas(net)

	expected := net.layers[1].Nodes()[0].Output() * net.outputLayer().Nodes()[0].Delta() * net.layers[1].Nodes()[0].Connection(0).Weight()
	if net.layers[1].Nodes()[0].Delta() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.layers[1].Nodes()[0].Delta())
	}
}

func Test_accumulateDeltas_accumulatesTheCorrectValuesWithTwoAssociations(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, outputs := generateSimpleData(1, 1, 2)

	expected := 0.0
	for i := 0; i < len(inputs); i++ {
		net.inputLayer().ResetInputs()
		net.inputLayer().SetInputs(inputs[i])
		net.inputLayer().Activate()
		calculateOutputDeltas(net.outputLayer(), outputs[i])

		calculateHiddenDeltas(net)
		expected += net.layers[1].Nodes()[0].Output() * net.outputLayer().Nodes()[0].Delta() * net.layers[1].Nodes()[0].Connection(0).Weight()
	}

	if math.Abs(net.layers[1].Nodes()[0].Delta()) - math.Abs(expected) > 0.00001 {
		t.Fatalf("Expected %f, got %f", expected, net.layers[1].Nodes()[0].Delta())
	}
}

// accumulate gradients

func Test_accumulateGradients_setsTheCorrectValues(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, outputs := generateSimpleData(1, 1, 1)
	net.inputLayer().ResetInputs()
	net.inputLayer().SetInputs(inputs[0])
	net.inputLayer().Activate()
	calculateOutputDeltas(net.outputLayer(), outputs[0])

	accumulateGradients(net)

	expected := net.outputLayer().Nodes()[0].Delta() * net.layers[1].Nodes()[0].Output()
	if net.layers[1].Nodes()[0].Connection(0).Gradient() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.layers[1].Nodes()[0].Connection(0).Gradient())
	}
}

func Test_accumulateGradients_accumulatesTheCorrectValuesWithTwoAssociations(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, outputs := generateSimpleData(1, 1, 2)

	expected := 0.0
	for i := 0; i < len(inputs); i++ {
		net.inputLayer().ResetInputs()
		net.inputLayer().SetInputs(inputs[i])
		net.inputLayer().Activate()
		calculateOutputDeltas(net.outputLayer(), outputs[i])

		accumulateGradients(net)
		expected += net.outputLayer().Nodes()[0].Delta() * net.layers[1].Nodes()[0].Output()
	}

	if net.layers[1].Nodes()[0].Connection(0).Gradient() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.layers[1].Nodes()[0].Connection(0).Gradient())
	}
}
