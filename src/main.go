package main

import (
	"github.com/gokadin/ann-core/core"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().UTC().UnixNano())

	n := core.NewNetwork()
	n.AddInputLayer(2).
		AddHiddenLayer(2).
		AddOutputLayer(2)
}
