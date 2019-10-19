package layer

import (
	"github.com/gokadin/ann-core/layer"
	"math"
	"math/rand"
	"testing"
)

func buildSimpleTestNetwork(inputCount, hiddenCount, outputCount int, activationFunction string) *Collection {
	inputLayer := NewLayer(inputCount, layer.FunctionIdentity)
	hiddenLayer := NewLayer(hiddenCount, activationFunction)
	outputLayer := NewLayer(outputCount, layer.FunctionIdentity)
	inputLayer.ConnectTo(hiddenLayer)
	hiddenLayer.ConnectTo(outputLayer)

	network := NewCollection()
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

func Test_forwardPass_setsCorrectInputValues(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, _ := generateSimpleData(1, 1, 1)

	net.InputLayer().SetInputs(inputs[0])

	for i, value := range inputs[0] {
		if net.InputLayer().Nodes()[i].Input() != value {
			t.Fatalf("Expected %f, got %f", value, net.InputLayer().Nodes()[i].Input())
		}
	}
}

func Test_forwardPass_outputIsCorrect(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, _ := generateSimpleData(1, 1, 1)

	net.InputLayer().ResetInputs()
	net.InputLayer().SetInputs(inputs[0])
	net.InputLayer().Activate()

	expected := inputs[0][0] * net.InputLayer().Nodes()[0].Connection(0).Weight()
	expected = expected * net.Layers[1].Nodes()[0].Connection(0).Weight()
	if net.OutputLayer().Nodes()[0].Output() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.OutputLayer().Nodes()[0].Output())
	}
}

func Test_forwardPass_outputIsCorrectWithSigmoidActivationHiddenNode(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionSigmoid)
	inputs, _ := generateSimpleData(1, 1, 1)

	net.InputLayer().ResetInputs()
	net.InputLayer().SetInputs(inputs[0])
	net.InputLayer().Activate()

	expected := inputs[0][0] * net.InputLayer().Nodes()[0].Connection(0).Weight()
	expected = 1 / (1 + math.Pow(math.E, -expected))
	expected = expected * net.Layers[1].Nodes()[0].Connection(0).Weight()
	if net.OutputLayer().Nodes()[0].Output() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.OutputLayer().Nodes()[0].Output())
	}
}

func Test_forwardPass_outputIsCorrectWithTwoInputNodes(t *testing.T) {
	net := buildSimpleTestNetwork(2, 1, 1, layer.FunctionIdentity)
	inputs, _ := generateSimpleData(2, 1, 1)

	net.InputLayer().ResetInputs()
	net.InputLayer().SetInputs(inputs[0])
	net.InputLayer().Activate()

	h1 := inputs[0][0] * net.InputLayer().Nodes()[0].Connection(0).Weight()
	h1 += inputs[0][1] * net.InputLayer().Nodes()[1].Connection(0).Weight()
	expected := h1 * net.Layers[1].Nodes()[0].Connection(0).Weight()
	if net.OutputLayer().Nodes()[0].Output() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.OutputLayer().Nodes()[0].Output())
	}
}

func Test_forwardPass_outputIsCorrectWithTwoInputNodesAndTwoHiddenNodes(t *testing.T) {
	net := buildSimpleTestNetwork(2, 2, 1, layer.FunctionIdentity)
	inputs, _ := generateSimpleData(2, 2, 1)

	net.InputLayer().ResetInputs()
	net.InputLayer().SetInputs(inputs[0])
	net.InputLayer().Activate()

	h1 := inputs[0][0] * net.InputLayer().Nodes()[0].Connection(0).Weight()
	h1 += inputs[0][1] * net.InputLayer().Nodes()[1].Connection(0).Weight()
	h2 := inputs[0][0] * net.InputLayer().Nodes()[0].Connection(1).Weight()
	h2 += inputs[0][1] * net.InputLayer().Nodes()[1].Connection(1).Weight()
	expected := h1 * net.Layers[1].Nodes()[0].Connection(0).Weight()
	expected += h2 * net.Layers[1].Nodes()[1].Connection(0).Weight()
	if net.OutputLayer().Nodes()[0].Output() != expected {
		t.Fatalf("Expected %f, got %f", expected, net.OutputLayer().Nodes()[0].Output())
	}
}
