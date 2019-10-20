package layer

import (
	"github.com/gokadin/ai-backpropagation/node"
	"log"
)

const defaultBiasValue = 1.0

type Layer struct {
	nodes     []*node.Node
	bias *node.Node
	nextLayer *Layer
	activationFunction func(x float64) float64
	activationFunctionDerivative func(x float64) float64
	isOutputLayer bool
}

func NewLayer(size int, activationFunctionName string) *Layer {
	return &Layer{
		nodes: initializeNodes(size),
		bias: node.NewBiasNode(defaultBiasValue),
		activationFunction: getActivationFunction(activationFunctionName),
		activationFunctionDerivative: getActivationFunctionDerivative(activationFunctionName),
		isOutputLayer: false,
	}
}

func NewOutputLayer(size int, activationFunctionName string) *Layer {
	layer := NewLayer(size, activationFunctionName)
	layer.isOutputLayer = true
	return layer
}

func initializeNodes(size int) []*node.Node {
	nodes := make([]*node.Node, size)
	for i := range nodes {
		nodes[i] = node.NewNode()
	}
	return nodes
}

func (l *Layer) Size() int {
	return len(l.nodes)
}

func (l *Layer) IsOutputLayer() bool {
	return l.isOutputLayer
}

func (l *Layer) ConnectTo(nextLayer *Layer) {
	l.nextLayer = nextLayer

	// connect all layer nodes to the next layer nodes
	for _, n := range l.nodes {
		for _, nextNode := range nextLayer.nodes {
			n.ConnectTo(nextNode)
		}
	}

	// connect bias to the next layer
	for _, nextNode := range nextLayer.nodes {
		l.bias.ConnectTo(nextNode)
	}
}

func (l *Layer) Nodes() []*node.Node {
	return l.nodes
}

func (l *Layer) Node(index int) *node.Node {
	return l.nodes[index]
}

func (l *Layer) Parameters() []*node.Node {
	if l.isOutputLayer {
		return l.nodes
	}

	return append(l.nodes, l.bias)
}

func (l *Layer) Bias() *node.Node {
	return l.bias
}

func (l *Layer) SetInputs(values []float64) {
	if len(values) != l.Size() {
		log.Fatal("Cannot set values, size mismatch:", len(values), "!=", l.Size())
	}

	for i, value := range values {
		l.nodes[i].SetInput(value)
	}
}

func (l *Layer) ResetInputs() {
	for _, n := range l.nodes {
		n.ResetInput()
	}

	if l.nextLayer != nil {
		l.nextLayer.ResetInputs()
	}
}

func (l *Layer) Activate() {
	for _, n := range l.nodes {
		n.Activate(l.activationFunction)
	}
	l.bias.Activate(nil)

	if l.nextLayer != nil {
		l.nextLayer.Activate()
	}
}

func (l *Layer) ActivationDerivative() func (x float64) float64 {
	return l.activationFunctionDerivative
}

