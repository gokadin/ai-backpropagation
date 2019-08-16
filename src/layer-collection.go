package main

import "github.com/gokadin/ann-core/layer"

type layerCollection struct {
	layers []*layer.Layer
}

func newLayerCollection() *layerCollection {
	return &layerCollection{
		layers: make([]*layer.Layer, 0),
	}
}

func (n *layerCollection) outputLayer() *layer.Layer {
	return n.layers[len(n.layers)-1]
}

func (n *layerCollection) inputLayer() *layer.Layer {
	return n.layers[0]
}
