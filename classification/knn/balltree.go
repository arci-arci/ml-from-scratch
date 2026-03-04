package knn

import (
	"math"
	"ml/common"
	"slices"
)

type Ball struct {
	Pivot  float64
	Center *WeightedBoW
	Radius float64
}

type BallNode struct {
	Ball  *Ball
	Left  *BallNode
	Right *BallNode
}

type BallTree struct {
	Root *BallNode
}

type Population = []float64
type SplitFn = func(point NormalizedDocument, dimension string, splitValue float64) bool

func CreateBallTree(v *common.Vocabulary, points []NormalizedDocument) BallTree {
	root := createBallTreeRecursive(v, points)
	tree := BallTree{
		Root: &root,
	}

	return tree
}

func createBallTreeRecursive(v *common.Vocabulary, points []NormalizedDocument) BallNode {
	if len(points) == 1 {
		leafRadius := calculateRadius(points[0].WBow, points, v)

		return BallNode{
			Ball:  &Ball{Center: points[0].WBow, Pivot: -1, Radius: leafRadius},
			Left:  nil,
			Right: nil,
		}
	}

	dimension := findDimension(v, points)
	population := make(Population, 0, len(*v))
	for _, point := range points {
		population = append(population, (*point.WBow)[dimension])
	}

	// More text preprocessing

	splittingValue := median(population)
	centroid := findCentroid(v, points)
	radius := calculateRadius(&centroid, points, v)

	leftPoints := splitPoints(points, dimension, splittingValue, leftFn)
	rightPoints := splitPoints(points, dimension, splittingValue, rightFn)
	leftNode := createBallTreeRecursive(v, leftPoints)
	rightNode := createBallTreeRecursive(v, rightPoints)

	node := BallNode{
		Ball:  &Ball{Center: nil, Pivot: splittingValue, Radius: radius},
		Left:  &leftNode,
		Right: &rightNode,
	}

	return node
}

func splitPoints(points []NormalizedDocument, dimension string, splitValue float64, splitFn SplitFn) []NormalizedDocument {
	newPoints := make([]NormalizedDocument, 0)

	for _, point := range points {
		if splitFn(point, dimension, splitValue) {
			newPoints = append(newPoints, point)
		}
	}

	return newPoints
}

func leftFn(point NormalizedDocument, dimension string, splitValue float64) bool {
	return (*point.WBow)[dimension] < splitValue
}

func rightFn(point NormalizedDocument, dimension string, splitValue float64) bool {
	return (*point.WBow)[dimension] > splitValue
}

func findDimension(v *common.Vocabulary, points []NormalizedDocument) string {
	var maxStd float64
	var needle string

	for token := range *v {
		population := make(Population, 0, len(*v))
		for _, point := range points {
			population = append(population, (*point.WBow)[token])
		}

		tokenAvg := avg(population)
		tokenStd := std(population, tokenAvg)

		if tokenStd > maxStd {
			maxStd = tokenStd
			needle = token
		}
	}

	return needle
}

func findCentroid(v *common.Vocabulary, points []NormalizedDocument) WeightedBoW {
	centroid := WeightedBoW{}

	for token := range *v {
		for _, p := range points {
			centroid[token] += (*p.WBow)[token]
		}
	}

	for token := range *v {
		centroid[token] /= float64(len(points))
	}

	return centroid
}

func calculateRadius(centroid *WeightedBoW, points []NormalizedDocument, v *common.Vocabulary) float64 {
	distances := make([]float64, 0, len(*centroid))

	for _, p := range points {
		d := EuclideanDistance(centroid, p.WBow, v)
		distances = append(distances, d)
	}

	return slices.Max(distances)
}

func std(population Population, avg float64) float64 {
	var sum float64
	for _, v := range population {
		sum += math.Pow(v-avg, 2)
	}

	variance := sum / float64(len(population))
	return math.Sqrt(variance)
}

func avg(population Population) float64 {
	var sum float64

	for _, v := range population {
		sum += v
	}

	return sum / float64(len(population))
}

func median(population Population) float64 {
	slices.Sort(population)

	if len(population)%2 == 0 {
		first := (len(population) / 2) - 1
		second := first + 1

		return (population[first] + population[second]) / 2
	}

	middle := len(population) / 2
	return population[middle]
}
