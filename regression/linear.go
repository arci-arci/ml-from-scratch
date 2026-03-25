package linear

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"strconv"
	"strings"
)

type CSVData = [][]float64
type Distribution struct {
	Train float64
	Test  float64
}

type Seed struct {
	FirstSeed uint64
	SeconSeed uint64
}

type LinearRegressionOpt struct {
	Train        CSVData
	InputFeature []int
	LearningRate float64
	Target       int
	BatchSize    int
	Alpha        float64
	Epochs       int
}

type LinearRegressionModel struct {
	Weights []float64
	Xs      []int
}

type StandardScaler struct {
	Mean []float64
	Std  []float64
}

func Train(options LinearRegressionOpt) LinearRegressionModel {
	batches := int(math.Floor(float64(len(options.Train)) / float64(options.BatchSize)))
	fmt.Printf("Batch amount for each epoch: %v\n", batches)

	// Adding 1 so I can take into account the intercept
	Xs := addIntercept(options.InputFeature)
	weights := initWeights(len(Xs))
	fmt.Printf("Genereted weights: %v\n", weights)

	currentBatch := 1
	currentEpoch := 1

	for currentEpoch <= options.Epochs {
		rand.Shuffle(options.BatchSize, func(i, j int) {
			options.Train[i], options.Train[j] = options.Train[j], options.Train[i]
		})

		for currentBatch <= batches {
			batchWeights := initBatchWeights(len(Xs))
			batch := createBatch(options.Train, options.BatchSize)

			for _, e := range batch {
				// Adding X0 value
				example := append([]float64{1.0}, e...)
				err := prediction(example, weights, Xs) - example[options.Target+1]

				for i, idx := range Xs {
					batchWeights[i] = batchWeights[i] + err*example[idx]
				}
			}

			for i := range Xs {
				weightMean := float64(batchWeights[i] / float64(options.BatchSize))
				penality := options.Alpha * weights[i]
				weights[i] = weights[i] - options.LearningRate*(weightMean+penality)
			}

			currentBatch += 1
		}

		currentEpoch += 1
	}

	fmt.Printf("Final Weight: %v\n", weights)

	return LinearRegressionModel{
		Weights: weights,
		Xs:      Xs,
	}
}

func Fit(model LinearRegressionModel, example []float64) float64 {
	example = append([]float64{1.0}, example...)
	return prediction(example, model.Weights, model.Xs)
}

func Mae(preds []float64, real []float64) float64 {
	sum := 0.0
	for i := range preds {
		sum += math.Abs(preds[i] - real[i])
	}

	return sum / float64(len(preds))
}

func prediction(example, weights []float64, inputFeatures []int) float64 {
	var targetPred float64

	for i, idx := range inputFeatures {
		targetPred = targetPred + weights[i]*example[idx]
	}

	return targetPred
}

func addIntercept(input []int) []int {
	Xs := make([]int, 0, len(input)+1)
	Xs = append(Xs, 0)

	for _, idx := range input {
		Xs = append(Xs, idx+1)
	}

	return Xs
}

func initBatchWeights(nWeights int) []float64 {
	batchWeights := make([]float64, 0, nWeights)
	for range nWeights {
		batchWeights = append(batchWeights, 0.0)
	}

	return batchWeights
}

func initWeights(nWeights int) []float64 {
	weights := make([]float64, 0, nWeights)

	for range nWeights {
		w := rand.NormFloat64()
		weights = append(weights, w)
	}

	return weights
}

func createBatch(dataset CSVData, batchSize int) [][]float64 {
	acc := make([]int, 0, batchSize)
	batch := make([][]float64, 0, batchSize)

	for len(acc) != batchSize {
		idx := rand.IntN(len(dataset))
		if slices.Contains(acc, idx) {
			continue
		}

		acc = append(acc, idx)
		batch = append(batch, dataset[idx])
	}

	return batch

}

func ReadCSV() CSVData {
	f, err := os.Open("regression/wine.csv")

	if err != nil {
		panic(err)
	}

	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		panic("Unable to parse input file")
	}

	parsedRecord := make(CSVData, 0, len(records))
	for _, record := range records {
		data := make([]float64, 0, len(record)-1)
		for i := 1; i < len(record); i++ {
			v, err := strconv.ParseFloat(strings.TrimSpace(record[i]), 64)
			if err != nil {
				panic("Invalid number")
			}

			data = append(data, v)
		}

		parsedRecord = append(parsedRecord, data)
	}

	return parsedRecord
}

func SplitDataset(dataset CSVData, dist Distribution, seed Seed) (trainSet, testSet CSVData) {
	source := rand.NewPCG(seed.FirstSeed, seed.SeconSeed)
	randGen := rand.New(source)

	rand.Shuffle(len(dataset), func(i, j int) {
		dataset[i], dataset[j] = dataset[j], dataset[i]
	})

	total := len(dataset)
	acc := make([]int, 0, total)
	trainSize := int(math.Floor(float64(total) * dist.Train))
	testSize := int(math.Floor(float64(total) * dist.Test))

	trainSetIdxs, acc := generate(trainSize, total, randGen, acc)
	testSetIdx, acc := generate(testSize, total, randGen, acc)

	if len(acc) < total {
		sets := [][]int{trainSetIdxs, testSetIdx}

		// Distibute missing values
		// using a round-robing strategy
		sets = distributeMissingValues(sets, acc)

		trainSetIdxs = sets[0]
		testSetIdx = sets[1]
	}

	trainSet = createSet(dataset, trainSetIdxs)
	testSet = createSet(dataset, testSetIdx)

	return trainSet, testSet
}

func GetFeatureValues(train CSVData, feature int) []float64 {
	values := make([]float64, len(train))

	for i, row := range train {
		values[i] = row[feature]
	}

	return values
}

func generate(size int, total int, randGen *rand.Rand, acc []int) (elements, accR []int) {
	elements = make([]int, 0, size)
	i := 0

	for i < size {
		idx := randGen.IntN(total)
		isNotGenerated := !slices.Contains(elements, idx)
		isNotPreviouslyGenerated := !slices.Contains(acc, idx)

		if isNotGenerated && isNotPreviouslyGenerated {
			elements = append(elements, idx)
			acc = append(acc, idx)
			i += 1
		}
	}

	return elements, acc
}

func createSet(dataset CSVData, idxs []int) CSVData {
	set := make(CSVData, 0, len(idxs))

	for _, idx := range idxs {
		set = append(set, dataset[idx])
	}

	return set
}

func distributeMissingValues(sets [][]int, acc []int) [][]int {
	missing := cap(acc) - len(acc)
	setIdx := 0

	for i := range cap(acc) {
		if slices.Contains(acc, i) {
			continue
		}

		set := sets[setIdx%len(sets)]
		set = append(set, i)
		sets[setIdx%len(sets)] = set

		setIdx += 1
		missing -= 1

		if missing == 0 {
			break
		}
	}

	return sets
}

func FitStandardScaler(train CSVData, target int) StandardScaler {
	n := len(train[0])
	m := len(train)
	mean := make([]float64, n)
	std := make([]float64, n)

	// Compute mean
	for _, row := range train {
		for j, v := range row {
			mean[j] += v
		}
	}

	for j := range mean {
		mean[j] /= float64(m)
	}

	// Compute std deviation
	for _, row := range train {
		for j, v := range row {
			diff := v - mean[j]
			std[j] += diff * diff
		}
	}

	for j := range std {
		std[j] = math.Sqrt(std[j] / float64(m))
	}

	return StandardScaler{
		Mean: mean,
		Std:  std,
	}
}

func Transform(s StandardScaler, train CSVData, target int) CSVData {
	scaled := make([][]float64, len(train))

	for i, row := range train {
		scaled[i] = make([]float64, len(row))
		for j, v := range row {
			if s.Std[j] == 0 {
				scaled[i][j] = 0
			} else {
				scaled[i][j] = (v - s.Mean[j]) / s.Std[j]
			}
		}
	}

	return scaled
}

func InverseTransform(s StandardScaler, y float64, targert int) float64 {
	return y*s.Std[targert] + s.Mean[targert]
}
