package layer

import "math"

const (
	FunctionIdentity = "functionIdentity"
	FunctionIdentityDerivative = "functionIdentityDerivative"
	FunctionSigmoid = "functionSigmoid"
	FunctionSigmoidDerivative = "functionSigmoidDerivative"
	FunctionRelu = "functionRelu"
	FunctionReluDerivative = "functionReluDerivative"
	FunctionLeakyRelu = "functionLeakyRelu"
	FunctionLeakyReluDerivative = "functionLeakyReluDerivative"
	FunctionSoftmax = "functionSoftmax"
)

func getActivationFunction(name string) func(x float64) float64 {
	switch name {
	case FunctionIdentity:
		return Identity
	case FunctionIdentityDerivative:
		return IdentityDerivative
	case FunctionSigmoid:
		return Sigmoid
	case FunctionSigmoidDerivative:
		return SigmoidDerivative
	case FunctionRelu:
		return Relu
	case FunctionReluDerivative:
		return ReluDerivative
	case FunctionLeakyRelu:
		return LeakyRelu
	case FunctionLeakyReluDerivative:
		return LeakyReluDerivative
	default:
		return nil
	}
}

func getActivationFunctionDerivative(name string) func(x float64) float64 {
	return getActivationFunction(name + "Derivative")
}

func Identity(x float64) float64 {
	return x
}

func IdentityDerivative(x float64) float64 {
	return 1
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}

func SigmoidDerivative(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

func Relu(x float64) float64 {
	return math.Max(0.0, x)
}

func ReluDerivative(x float64) float64 {
    if x > 0.0 {
    	return 1.0
	}

    return 0.0
}

func LeakyRelu(x float64) float64 {
	if x >= 0 {
		return x
	}

	return 0.01 * x
}

func LeakyReluDerivative(x float64) float64 {
	if x >= 0.0 {
		return 1.0
	}

	return 0.01
}
