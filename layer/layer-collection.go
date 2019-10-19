package layer

type Collection struct {
	Layers []*Layer
}

func NewCollection() *Collection {
	return &Collection{
		Layers: make([]*Layer, 0),
	}
}

func (n *Collection) OutputLayer() *Layer {
	return n.Layers[len(n.Layers)-1]
}

func (n *Collection) InputLayer() *Layer {
	return n.Layers[0]
}
