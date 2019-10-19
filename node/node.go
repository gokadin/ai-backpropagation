package node

type Node struct {
	input       float64
	output      float64
	delta       float64
	connections []*connection
}

func NewNode() *Node {
	return &Node{
		connections: make([]*connection, 0),
	}
}

func (n *Node) ConnectTo(nextNode *Node, weight float64) {
	n.connections = append(n.connections, newConnection(nextNode, weight))
}

func (n *Node) Connections() []*connection {
	return n.connections
}

func (n *Node) Connection(index int) *connection {
	return n.connections[index]
}

func (n *Node) ResetInput() {
	n.input = 0.0
}

func (n *Node) Input() float64 {
	return n.input
}

func (n *Node) Output() float64 {
	return n.output
}

func (n *Node) SetOutput(value float64) {
    n.output = value
}

func (n *Node) SetInput(value float64) {
	n.input = value
}

func (n *Node) AddInput(value float64) {
	n.input += value
}

func (n *Node) SetDelta(delta float64) {
	n.delta = delta
}

func (n *Node) Delta() float64 {
    return n.delta
}

func (n *Node) Activate(activationFunction func(x float64) float64) {
	n.output = activationFunction(n.input)

	for _, connection := range n.connections {
		connection.nextNode.AddInput(n.output * connection.weight)
	}
}
