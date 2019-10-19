package layer

import (
	"github.com/gokadin/ai-backpropagation/node"
	"log"
)

type Layer struct {
	nodes     []*node.Node
	nextLayer *Layer
	activationFunction func(x float64) float64
	activationFunctionDerivative func(x float64) float64
	isOutputLayer bool
}

func NewLayer(size int, activationFunctionName string) *Layer {
	nodes := make([]*node.Node, size + 1) // +1 for bias
	for i := range nodes {
		nodes[i] = node.NewNode()
	}
	return &Layer{
		nodes: nodes,
		activationFunction: getActivationFunction(activationFunctionName),
		activationFunctionDerivative: getActivationFunctionDerivative(activationFunctionName),
		isOutputLayer: false,
	}
}

func NewOutputLayer(size int, activationFunctionName string) *Layer {
	nodes := make([]*node.Node, size)
	for i := range nodes {
		nodes[i] = node.NewNode()
	}
	return &Layer{
		nodes: nodes,
		activationFunction: getActivationFunction(activationFunctionName),
		activationFunctionDerivative: getActivationFunctionDerivative(activationFunctionName),
		isOutputLayer: true,
	}
}

func (l *Layer) Size() int {
	return len(l.nodes)
}

func (l *Layer) IsOutputLayer() bool {
	return l.isOutputLayer
}

func (l *Layer) ConnectTo(nextLayer *Layer) {
	l.nextLayer = nextLayer

	for _, n := range l.nodes {
		for _, nextNode := range nextLayer.nodes {
			n.ConnectTo(nextNode)
		}
	}
}

func (l *Layer) Nodes() []*node.Node {
	return l.nodes
}

func (l *Layer) Node(index int) *node.Node {
	return l.nodes[index]
}

func (l *Layer) SetInputs(values []float64) {
	if len(values) != l.Size() - 1 {
		log.Fatal("Cannot set values, size mismatch:", len(values), "!=", l.Size())
	}

	for i, value := range values {
		l.nodes[i].SetInput(value)
	}

	if !l.isOutputLayer {
		l.nodes[len(values)].SetInput(1.0)
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

	if l.nextLayer != nil {
		l.nextLayer.Activate()
	}
}

func (l *Layer) ActivationDerivative() func (x float64) float64 {
	return l.activationFunctionDerivative
}

