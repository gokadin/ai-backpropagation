package main

import "github.com/gokadin/ann-core/layer"

type network struct {
	layers []*layer.Layer
}

func newNetwork() *network {
	return &network{
		layers: make([]*layer.Layer, 0),
	}
}

func (n *network) outputLayer() *layer.Layer {
	return n.layers[len(n.layers) - 1]
}

func (n *network) inputLayer() *layer.Layer {
	return n.layers[0]
}
