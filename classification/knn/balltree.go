package knn

import (
	"math"
	"math/rand/v2"
	"ml/common"
	"slices"
	"sort"
)

const initialCap int = 100
const minNodes int = 60

type SplitFn = func(point NormalizedDocument, dimension string, median float64) bool

type VarianceData struct {
	Variance float64
	Term     string
}

type Ball struct {
	Center    map[string]float64
	Threshold float64
	Radius    float64
	Left      *Ball
	Right     *Ball
	Items     []NormalizedDocument
}

type BallTree struct {
	Root *Ball
}

func BuildBallTree(db []NormalizedDocument, v *common.Vocabulary, leafSize int) *BallTree {
	dbSize := len(db)
	rand.Shuffle(dbSize, func(i, j int) {
		db[i], db[j] = db[j], db[i]
	})

	tree := BallTree{
		Root: &Ball{},
	}

	buildBallTreeRecursive(tree.Root, db, v, leafSize)

	return &tree
}

func IsLeaf(node *Ball) bool {
	return node.Left == nil && node.Right == nil
}

func buildBallTreeRecursive(node *Ball, db []NormalizedDocument, v *common.Vocabulary, leafSize int) {
	if len(db) <= leafSize {
		node.Items = db
		return
	}

	dbSize := len(db)

	// use only 10% of data
	n := int(math.Floor(float64(dbSize) * 0.1))
	points := selectPoints(dbSize, n)
	spreads := calcSpread(db, v)
	splitValue, dimensionIdx := findBestSplit(db, points, spreads)

	if splitValue == -1.0 {
		// Make a leaf
		node.Items = db
		return
	}

	dimension := spreads[dimensionIdx].Term
	radius := findRadius(db, splitValue, dimension)

	leftDb := createSubTree(db, dimension, splitValue, left)
	rightDb := createSubTree(db, dimension, splitValue, right)

	node.Threshold = splitValue
	node.Radius = radius
	node.Center = centroide(db, v)

	if len(leftDb) > 0 {
		node.Left = &Ball{}
		buildBallTreeRecursive(node.Left, leftDb, v, leafSize)
	}

	if len(rightDb) > 0 {
		node.Right = &Ball{}
		buildBallTreeRecursive(node.Right, rightDb, v, leafSize)
	}
}

func centroide(db []NormalizedDocument, v *common.Vocabulary) map[string]float64 {
	centroide := make(map[string]float64)
	dbSize := len(db)

	for token := range *v {
		for _, point := range db {
			v := (*point.WBow)[token]
			centroide[token] += v
		}
	}

	for token := range *v {
		centroide[token] /= float64(dbSize)
	}

	return centroide
}

func createSubTree(db []NormalizedDocument, dimension string, split float64, fn SplitFn) []NormalizedDocument {
	subDb := make([]NormalizedDocument, 0, initialCap)

	for _, point := range db {
		if fn(point, dimension, split) {
			subDb = append(subDb, point)
		}
	}

	return subDb
}

func findRadius(db []NormalizedDocument, split float64, dimension string) float64 {
	maxDist := -1.0

	for _, point := range db {
		currentDist := math.Pow(split-(*point.WBow)[dimension], 2)
		if currentDist > maxDist {
			maxDist = currentDist
		}
	}

	return maxDist
}

func left(point NormalizedDocument, dimension string, split float64) bool {
	return (*point.WBow)[dimension] < split
}

func right(point NormalizedDocument, dimension string, split float64) bool {
	return (*point.WBow)[dimension] >= split
}

func calcSpread(db []NormalizedDocument, v *common.Vocabulary) (variances []VarianceData) {
	variances = make([]VarianceData, 0, len(*v))

	for d := range *v {
		variance := calcVariance(db, d)
		variances = append(variances, VarianceData{Variance: variance, Term: d})
	}

	sort.Slice(variances, func(i, j int) bool {
		return variances[i].Variance > variances[j].Variance
	})

	return variances
}

func findSplitValue(db []NormalizedDocument, points []int, dimension string) float64 {
	sum := 0.0
	for _, pointIdx := range points {
		point := db[pointIdx]
		sum += (*point.WBow)[dimension]
	}

	return sum / float64(len(points))
}

func findBestSplit(db []NormalizedDocument, points []int, spreads []VarianceData) (float64, int) {
	splitValue := -1.0
	dimensionIdx := 0

	for idx, spread := range spreads {
		currentSplitValue := findSplitValue(db, points, spread.Term)

		if currentSplitValue > splitValue && currentSplitValue > 0 {
			splitValue = currentSplitValue
			dimensionIdx = idx
		}
	}

	return splitValue, dimensionIdx
}

func calcVariance(db []NormalizedDocument, dimension string) float64 {
	totalCount := len(db)
	if totalCount == 0 {
		return 0
	}

	values := make([]float64, 0, initialCap)

	for _, point := range db {
		v := (*point.WBow)[dimension]

		if v == 0 {
			continue
		}

		values = append(values, v)
	}

	nonZeroCount := len(values)
	if nonZeroCount == 0 {
		return 0
	}

	density := float64(nonZeroCount) / float64(totalCount)
	mean := calcMean(values)
	deviations := make([]float64, 0, initialCap)

	for _, v := range values {
		dev := math.Pow(v-mean, 2)
		deviations = append(deviations, dev)
	}

	variance := calcMean(deviations)
	return float64(density) * variance
}

func calcMean(values []float64) float64 {
	sumV := 0.0
	for _, v := range values {
		sumV += v
	}

	return sumV / float64(len(values))
}

func selectPoints(dbSize int, nPoints int) []int {
	points := make([]int, nPoints)
	generetedPoints := 0

	for generetedPoints < nPoints {
		idx := rand.IntN(dbSize)

		if slices.Contains(points, idx) {
			continue
		}

		points[generetedPoints] = idx
		generetedPoints += 1
	}

	return points
}
