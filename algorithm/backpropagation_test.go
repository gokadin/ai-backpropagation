package algorithm

import (
	"github.com/gokadin/ai-backpropagation/layer"
	"math"
	"math/rand"
	"testing"
)

func buildSimpleTestNetwork(inputCount, hiddenCount, outputCount int, activationFunction string) *layer.Collection {
	inputLayer := layer.NewLayer(inputCount, layer.FunctionIdentity)
	hiddenLayer := layer.NewLayer(hiddenCount, activationFunction)
	outputLayer := layer.NewLayer(outputCount, layer.FunctionIdentity)
	inputLayer.ConnectTo(hiddenLayer)
	hiddenLayer.ConnectTo(outputLayer)

	network := layer.NewCollection()
	network.Layers = append(network.Layers, inputLayer)
	network.Layers = append(network.Layers, hiddenLayer)
	network.Layers = append(network.Layers, outputLayer)

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

func Test_accumulateOutputDeltas_setsTheCorrectValues(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, outputs := generateSimpleData(1, 1, 1)
	net.InputLayer().ResetInputs()
	net.InputLayer().SetInputs(inputs[0])
	net.InputLayer().Activate()

	calculateOutputDeltas(net.OutputLayer(), outputs[0])

	expected := net.OutputLayer().Nodes()[0].Output() - outputs[0][0]
	if net.OutputLayer().Nodes()[0].Delta() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.OutputLayer().Nodes()[0].Delta())
	}
}

func Test_accumulateOutputDeltas_accumulatesTheCorrectValuesWithTwoAssociations(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, outputs := generateSimpleData(1, 1, 2)

	expected := 0.0
	for i := 0; i < len(inputs); i++ {
		net.InputLayer().ResetInputs()
		net.InputLayer().SetInputs(inputs[i])
		net.InputLayer().Activate()

		calculateOutputDeltas(net.OutputLayer(), outputs[i])
		expected += net.OutputLayer().Nodes()[0].Output() - outputs[i][0]
	}

	if net.OutputLayer().Nodes()[0].Delta() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.OutputLayer().Nodes()[0].Delta())
	}
}

func Test_accumulateOutputDeltas_accumulatesTheCorrectValuesWithTwoAssociationsAndMultipleOutputNodes(t *testing.T) {
	net := buildSimpleTestNetwork(2, 2, 2, layer.FunctionIdentity)
	inputs, outputs := generateSimpleData(2, 2, 2)

	expected := make([]float64, len(outputs))
	for i := 0; i < len(inputs); i++ {
		net.InputLayer().ResetInputs()
		net.InputLayer().SetInputs(inputs[i])
		net.InputLayer().Activate()

		calculateOutputDeltas(net.OutputLayer(), outputs[i])
		for outputNodeIndex, outputNode := range net.OutputLayer().Nodes() {
			expected[outputNodeIndex] += outputNode.Output() - outputs[i][outputNodeIndex]
		}
	}

	for outputNodeIndex, outputNode := range net.OutputLayer().Nodes() {
		if outputNode.Delta() != expected[outputNodeIndex] {
			t.Fatalf("Expected %f, got %f", expected[outputNodeIndex], outputNode.Delta())
		}
	}
}

// hidden deltas

func Test_accumulateDeltas_setsTheCorrectValues(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, outputs := generateSimpleData(1, 1, 1)
	net.InputLayer().ResetInputs()
	net.InputLayer().SetInputs(inputs[0])
	net.InputLayer().Activate()
	calculateOutputDeltas(net.OutputLayer(), outputs[0])

	calculateHiddenDeltas(net)

	expected := net.Layers[1].Nodes()[0].Output() * net.OutputLayer().Nodes()[0].Delta() * net.Layers[1].Nodes()[0].Connection(0).Weight()
	if net.Layers[1].Nodes()[0].Delta() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.Layers[1].Nodes()[0].Delta())
	}
}

func Test_accumulateDeltas_accumulatesTheCorrectValuesWithTwoAssociations(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, outputs := generateSimpleData(1, 1, 2)

	expected := 0.0
	for i := 0; i < len(inputs); i++ {
		net.InputLayer().ResetInputs()
		net.InputLayer().SetInputs(inputs[i])
		net.InputLayer().Activate()
		calculateOutputDeltas(net.OutputLayer(), outputs[i])

		calculateHiddenDeltas(net)
		expected += net.Layers[1].Nodes()[0].Output() * net.OutputLayer().Nodes()[0].Delta() * net.Layers[1].Nodes()[0].Connection(0).Weight()
	}

	if math.Abs(net.Layers[1].Nodes()[0].Delta()) - math.Abs(expected) > 0.00001 {
		t.Fatalf("Expected %f, got %f", expected, net.Layers[1].Nodes()[0].Delta())
	}
}

// accumulate gradients

func Test_accumulateGradients_setsTheCorrectValues(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, outputs := generateSimpleData(1, 1, 1)
	net.InputLayer().ResetInputs()
	net.InputLayer().SetInputs(inputs[0])
	net.InputLayer().Activate()
	calculateOutputDeltas(net.OutputLayer(), outputs[0])

	accumulateGradients(net)

	expected := net.OutputLayer().Nodes()[0].Delta() * net.Layers[1].Nodes()[0].Output()
	if net.Layers[1].Nodes()[0].Connection(0).Gradient() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.Layers[1].Nodes()[0].Connection(0).Gradient())
	}
}

func Test_accumulateGradients_accumulatesTheCorrectValuesWithTwoAssociations(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, outputs := generateSimpleData(1, 1, 2)

	expected := 0.0
	for i := 0; i < len(inputs); i++ {
		net.InputLayer().ResetInputs()
		net.InputLayer().SetInputs(inputs[i])
		net.InputLayer().Activate()
		calculateOutputDeltas(net.OutputLayer(), outputs[i])

		accumulateGradients(net)
		expected += net.OutputLayer().Nodes()[0].Delta() * net.Layers[1].Nodes()[0].Output()
	}

	if net.Layers[1].Nodes()[0].Connection(0).Gradient() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.Layers[1].Nodes()[0].Connection(0).Gradient())
	}
}
