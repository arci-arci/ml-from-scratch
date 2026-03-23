package example

import (
	"fmt"
	linear "ml/regression"
)

func RunLinearRegression() {
	dataset := linear.ReadCSV()
	distribution := linear.Distribution{
		Train:      0.7,
		Validation: 0.2,
		Test:       0.1,
	}
	seed := linear.Seed{
		FirstSeed: 42,
		SeconSeed: 1024,
	}

	target := 9 // color_intensity

	// for i, e := range dataset {
	// 	if i%10 != 0 || i == 0 {
	// 		fmt.Printf("%v  ", e[target])
	// 	} else {
	// 		fmt.Printf("%v\n", e[target])
	// 	}
	// }

	// fmt.Printf("\n\n-------------------\n\n")

	train, val, test := linear.SplitDataset(dataset, distribution, seed)
	options := linear.LinearRegressionOpt{
		Train:        train,
		Validation:   val,
		LearningRate: 0.05,
		InputFeature: []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11},
		Target:       target,
		BatchSize:    5,
		BatchSeed: linear.Seed{
			FirstSeed: 10,
			SeconSeed: 1000,
		},
	}

	model := linear.Train(options)
	example := test[6]
	pred := linear.Fit(model, example, target)
	fmt.Printf("Prediction %v => Real Value %v\n", pred, example[target])
}
