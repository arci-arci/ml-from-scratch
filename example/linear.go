package example

import (
	"fmt"
	"math"
	linear "ml/regression"
)

func RunLinearRegression() {
	dataset := linear.ReadCSV()
	distribution := linear.Distribution{
		Train: 0.85,
		Test:  0.15,
	}
	seed := linear.Seed{
		FirstSeed: 42,
		SeconSeed: 1024,
	}

	target := 5 // Total phenols
	train, test := linear.SplitDataset(dataset, distribution, seed)
	yTrain := linear.GetFeatureValues(train, target)

	scaler := linear.FitStandardScaler(train, target)
	scaledTrain := linear.Transform(scaler, train, target)
	scaledTest := linear.Transform(scaler, test, target)

	// Pearson's correlation coefficient
	// for the top 3 input features
	//
	// r       feature target
	// 0.86    6       5
	// 0.70    11      5
	// 0.55    12      5

	options := linear.LinearRegressionOpt{
		Train:        scaledTrain,
		LearningRate: 0.0002,
		InputFeature: []int{6, 11, 12},
		Target:       target,
		BatchSize:    30,
		Alpha:        0.5,
		Epochs:       550,
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

		fmt.Printf("Pred: %.2f, Real: %v\n", pred, trueV)
	}

	fmt.Print("\n\n------- MAE -------\n\n")
	fmt.Printf("MAE: %.2f\n\n", linear.Mae(predictions, trueValues))

}

func FindCorrelations(target int) {
	dataset := linear.ReadCSV()
	distribution := linear.Distribution{
		Train: 0.85,
		Test:  0.15,
	}
	seed := linear.Seed{
		FirstSeed: 42,
		SeconSeed: 1024,
	}

	train, _ := linear.SplitDataset(dataset, distribution, seed)
	yTrain := linear.GetFeatureValues(train, target)
	nFeature := len(train[0])

	fmt.Printf("r\tfeature\ttarget\n")

	for feature := range nFeature {
		if feature == target {
			continue
		}

		fValues := linear.GetFeatureValues(train, feature)
		r := correlation(fValues, yTrain)

		fmt.Printf("%.2f\t%v\t%v\n", r, feature, target)
	}
}

func correlation(x, y []float64) float64 {
	// Correlation calculated based on
	// Pearson's correlation coefficient

	n := len(x)
	meanX := 0.0
	meanY := 0.0

	for i := range x {
		meanX += x[i]
		meanY += y[i]
	}

	meanX /= float64(n)
	meanY /= float64(n)

	num := 0.0
	denX := 0.0
	denY := 0.0

	for i := range x {
		dx := x[i] - meanX
		dy := y[i] - meanY

		num += dx * dy
		denX += math.Pow(dx, 2)
		denY += math.Pow(dy, 2)
	}

	return num / math.Sqrt(denX*denY)
}
