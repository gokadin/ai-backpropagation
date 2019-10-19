package layer

import (
	"github.com/gokadin/ai-backpropagation/node"
	"log"
	"math"
	"math/rand"
)

type Layer struct {
	nodes     []*node.Node
	nextLayer *Layer
    activationFunctionName string
	isOutputLayer bool
}

func NewLayer(size int, activationFunctionName string) *Layer {
	nodes := make([]*node.Node, size + 1) // +1 for bias
	for i := range nodes {
		nodes[i] = node.NewNode()
	}
	return &Layer{
		nodes: nodes,
        activationFunctionName: activationFunctionName,
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
        activationFunctionName: activationFunctionName,
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

			/* Better weight initialization */

			weight := rand.NormFloat64() / math.Sqrt(float64(l.Size()))
			n.ConnectTo(nextNode, weight)
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
	switch l.activationFunctionName {
	case FunctionSoftmax:
		l.activateSoftmax()
		break
	default:
		for _, n := range l.nodes {
			n.Activate(getActivationFunction(l.activationFunctionName))
		}
		break
	}

	if l.nextLayer != nil {
		l.nextLayer.Activate()
	}
}

func (l *Layer) ActivationDerivative() func (x float64) float64 {
	return getActivationFunctionDerivative(l.activationFunctionName)
}

func (l *Layer) activateSoftmax() {
	sum := 0.0
	for _, n := range l.nodes {
		sum += math.Pow(math.E, n.Input())
	}
	for _, n := range l.nodes {
		inputExp := math.Pow(math.E, n.Input())
		partialSum := sum - inputExp
		n.SetOutput(inputExp / partialSum)
	}
}
