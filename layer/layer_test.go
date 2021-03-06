package layer

import "testing"

const someLayerSize = 10

func Test_layer_initializes_with_correct_size(t *testing.T) {
	layer := NewLayer(someLayerSize, FunctionIdentity)

	if len(layer.nodes) != someLayerSize {
		t.Fatalf("Layer initialized with wrong size. Expected %d, got %d", someLayerSize, len(layer.nodes))
	}
}
