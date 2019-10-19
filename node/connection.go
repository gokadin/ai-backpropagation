package node

import "math/rand"

type connection struct {
	nextNode *Node
	weight   float64
	gradient float64
	gradientCounter int
}

func newConnection(nextNode *Node) *connection {
	return &connection{
		nextNode: nextNode,
		weight:   rand.Float64(),
	}
}

func newConnectionWithWeight(nextNode *Node, weight float64) *connection {
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
	c.gradientCounter++
}

func (c *connection) Gradient() float64 {
	return c.gradient
}

func (c *connection) UpdateWeight(learningRate float64) {
	c.weight -= learningRate * c.gradient / float64(c.gradientCounter)
	c.gradient = 0.0
	c.gradientCounter = 0
}

