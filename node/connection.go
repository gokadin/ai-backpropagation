package node

type connection struct {
	nextNode *Node
	weight   float64
	gradient float64
	velocity float64
	sqrt float64
}

func newConnection(nextNode *Node, weight float64) *connection {
	return &connection{
		nextNode: nextNode,
		weight:   weight,
	}
}

func (c *connection) Weight() float64 {
	return c.weight
}

func (c *connection) NextNode() *Node {
	return c.nextNode
}

func (c *connection) AddGradient(value float64) {
	c.gradient += value
}

func (c *connection) Gradient() float64 {
	return c.gradient
}

func (c *connection) ResetGradient() {
	c.gradient = 0.0
}

func (c *connection) GetWeight() float64 {
	return c.weight
}

func (c *connection) SetWeight(weight float64) {
	c.weight = weight
}

func (c *connection) GetGradient() float64 {
	return c.gradient
}

func (c *connection) GetVelocity() float64 {
	return c.velocity
}

func (c *connection) SetVelocity(velocity float64) {
	c.velocity = velocity
}

func (c *connection) GetSqrt() float64 {
	return c.sqrt
}

func (c *connection) SetSqrt(sqrt float64) {
	c.sqrt = sqrt
}
