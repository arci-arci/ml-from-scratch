package example

import (
	"fmt"
	"math"
	linear "ml/regression"
)

func RunLinearRegression() {
	dataset := linear.ReadCSV()
	distribution := linear.Distribution{
		Train: 0.7,
		Test:  0.3,
	}
	seed := linear.Seed{
		FirstSeed: 42,
		SeconSeed: 1024,
	}

	target := 9 // color_intensity
	train, test := linear.SplitDataset(dataset, distribution, seed)
	yTrain := linear.GetTarget(train, target)

	scaler := linear.FitStandardScaler(train, target)
	scaledTrain := linear.Transform(scaler, train, target)
	scaledTest := linear.Transform(scaler, test, target)

	options := linear.LinearRegressionOpt{
		Train:        scaledTrain,
		LearningRate: 0.0002,
		InputFeature: []int{0, 1, 4},
		Target:       target,
		BatchSize:    10,
		Alpha:        0.01,
		Epochs:       600,
	}

	model := linear.Train(options)

	fmt.Print("\n\n------- Test step -------\n\n")

	predictions := make([]float64, len(scaledTest))
	trueValues := make([]float64, len(scaledTest))

	for i, example := range scaledTest {
		scaledPred := linear.Fit(model, example)
		pred := linear.InverseTransform(scaler, scaledPred, target)
		trueV := yTrain[i]

		predictions[i] = pred
		trueValues[i] = trueV

		fmt.Printf("Pred: %v, Real: %v\n", pred, trueV)
	}

	fmt.Print("\n\n------- MAE -------\n\n")
	fmt.Printf("MAE: %v", linear.Mae(predictions, trueValues))

}

// Test correlation
func Correlation(x, y []float64) float64 {
	n := float64(len(x))
	meanX, meanY := 0.0, 0.0
	for i := range x {
		meanX += x[i]
		meanY += y[i]
	}
	meanX /= n
	meanY /= n

	num, denX, denY := 0.0, 0.0, 0.0
	for i := range x {
		dx := x[i] - meanX
		dy := y[i] - meanY
		num += dx * dy
		denX += dx * dx
		denY += dy * dy
	}
	return num / math.Sqrt(denX*denY)
}
