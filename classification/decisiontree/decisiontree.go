package decisiontree

import (
	"cmp"
	"math"
	"ml/common"
	"os"
	"path"
	"slices"
	"strings"
)

type DocumentData struct {
	DocumentName string
	Bow          *common.BoW
	Class        string
}

type CandidateSplit struct {
	quality float64
	feature string
}

type Node struct {
	Left              *Node
	Right             *Node
	Threshold         float64
	ClassDistribution ClassDistribution
	Majority          string
	SplitFeature      string
	Label             string
	IsLeaf            bool
}

type DecisionTree struct {
	Root *Node
}

type ClassDistribution = map[string]float64

type DecisionTreeOptions struct {
	Folders  []string
	Classes  []string
	MaxDepth int
	LeafSize int
}

func Train(options DecisionTreeOptions) DecisionTree {
	v := common.Vocabulary{}
	db := createDatabase(options.Folders, options.Classes)
	defineVocabulary(db, &v)
	tree := createDecisionTree(db, v, options)

	return tree
}

func Fit(model *DecisionTree, target *common.BoW) (string, ClassDistribution) {
	return runFit(model.Root, target)
}

func runFit(root *Node, target *common.BoW) (string, ClassDistribution) {
	stack := []*Node{root}
	distributions := ClassDistribution{}
	var class string

	for len(stack) > 0 {
		top := stack[len(stack)-1]
		stack = stack[:len(stack)-1] // remove top element

		if top == nil {
			continue
		}

		if top.IsLeaf {
			distributions = top.ClassDistribution
			class = top.Label
			break
		}

		tokenValue, ok := (*target)[top.SplitFeature]

		if ok {
			if float64(tokenValue) <= top.Threshold {
				stack = append(stack, top.Left)
			} else {
				stack = append(stack, top.Right)
			}
		} else {
			distributions = top.ClassDistribution
			class = top.Majority
			break
		}
	}

	return class, distributions
}

func createDecisionTree(db []DocumentData, v common.Vocabulary, options DecisionTreeOptions) DecisionTree {
	root := createDecisionTreeRecursive(db, v, options.LeafSize, options.MaxDepth, 0)
	return DecisionTree{
		Root: &root,
	}
}

func createDecisionTreeRecursive(db []DocumentData, v common.Vocabulary, leafSize, maxDepth, depth int) Node {
	classDistribution := calculateClassDistribution(db)

	if depth == maxDepth {
		class := getMostCommonClass(db)

		return Node{
			Label:             class,
			IsLeaf:            true,
			ClassDistribution: classDistribution,
		}
	}

	if len(db) <= leafSize {
		class := getMostCommonClass(db)
		return Node{
			Label:             class,
			IsLeaf:            true,
			ClassDistribution: classDistribution,
		}
	}

	if allSameClass(db) {
		// Entropy equals to zero

		return Node{
			Label:             db[0].Class,
			IsLeaf:            true,
			ClassDistribution: classDistribution,
		}
	}

	if len(v) == 0 {
		class := getMostCommonClass(db)

		return Node{
			Label:             class,
			IsLeaf:            true,
			ClassDistribution: classDistribution,
		}
	}

	candidates := make([]CandidateSplit, 0, len(v))

	for token := range v {
		left, right, _ := split(db, token)
		leftEntropy := calculateEntropy(left)
		rightEntropy := calculateEntropy(right)

		leftQuality := (float64(len(left)) / float64(len(db))) * leftEntropy
		rightQuality := (float64(len(right)) / float64(len(db))) * rightEntropy
		splitQuality := leftQuality + rightQuality

		candidates = append(candidates, CandidateSplit{
			quality: splitQuality,
			feature: token,
		})
	}

	minImpurity := slices.MinFunc(candidates, func(a, b CandidateSplit) int {
		return cmp.Compare(a.quality, b.quality)
	})

	left, right, threshold := split(db, minImpurity.feature)
	delete(v, minImpurity.feature)

	if len(left) == 0 || len(right) == 0 {
		// Create a leaf node using his parent data

		class := getMostCommonClass(db)
		return Node{
			Label:             class,
			IsLeaf:            true,
			ClassDistribution: classDistribution,
		}
	}

	leftNode := createDecisionTreeRecursive(left, v, leafSize, maxDepth, depth+1)
	rightNode := createDecisionTreeRecursive(right, v, leafSize, maxDepth, depth+1)
	majority := calculateMajorityClass(classDistribution)

	parent := Node{
		ClassDistribution: classDistribution,
		SplitFeature:      minImpurity.feature,
		Majority:          majority,
		Threshold:         threshold,
		Left:              &leftNode,
		Right:             &rightNode,
		IsLeaf:            false,
	}

	return parent
}

func allSameClass(db []DocumentData) bool {
	classes := getAllClassesFromDB(db)
	return len(classes) == 1
}

func getMostCommonClass(db []DocumentData) string {
	classes := getAllClassesFromDB(db)
	mostCommonClass := classes[0]
	maxCounter := 0

	for _, class := range classes {
		counter := 0

		for _, doc := range db {
			if doc.Class == class {
				counter += 1
			}
		}

		if counter > maxCounter {
			maxCounter = counter
			mostCommonClass = class
		}
	}

	return mostCommonClass
}

func defineVocabulary(db []DocumentData, v *common.Vocabulary) {
	for _, doc := range db {
		common.CreateVocabulary(doc.Bow, v)
	}
}

func createDatabase(folders []string, classes []string) []DocumentData {
	size, err := common.GetCollectionSize(folders, classes)
	if err != nil {
		panic(err)
	}

	documentsData := make([]DocumentData, 0, size)

	for _, folder := range folders {
		for _, class := range classes {
			documents, err := os.ReadDir(path.Join(folder, class))
			if err != nil {
				panic(err)
			}

			for _, document := range documents {
				if strings.ToLower(document.Name()) == "summary.txt" {
					continue
				}

				bow := common.BoW{}
				common.ReadClassDocument(folder, class, document.Name(), &bow)

				if len(bow) < common.AVG_TOKEN_AMOUNT {
					continue
				}

				documentsData = append(documentsData, DocumentData{DocumentName: document.Name(), Bow: &bow, Class: class})
			}
		}
	}

	return documentsData
}

func calculateEntropy(db []DocumentData) float64 {
	X := getAllClassesFromDB(db)
	proportions := map[string]float64{}
	for _, x := range X {
		proportions[x] = calculateProportion(db, x)
	}

	var entropy float64
	for _, p := range proportions {
		entropy = entropy + (-p * math.Log2(p))
	}

	return entropy
}

func getAllClassesFromDB(db []DocumentData) []string {
	classes := []string{}

	for _, doc := range db {
		if len(classes) == 0 {
			classes = append(classes, doc.Class)
			continue
		}

		if slices.Contains(classes, doc.Class) {
			continue
		}

		classes = append(classes, doc.Class)
	}

	return classes
}

func calculateProportion(db []DocumentData, class string) float64 {
	var elements int

	for _, doc := range db {
		if doc.Class == class {
			elements += 1
		}
	}

	return float64(elements) / float64(len(db))
}

func split(db []DocumentData, feature string) (leftDB, rightDB []DocumentData, threshold float64) {
	featureValues := []int64{}

	for _, doc := range db {
		if len(featureValues) == 0 {
			featureValues = append(featureValues, (*doc.Bow)[feature])
			continue
		}

		if slices.Contains(featureValues, (*doc.Bow)[feature]) {
			continue
		}

		featureValues = append(featureValues, (*doc.Bow)[feature])
	}

	leftDB = make([]DocumentData, 0, len(db)/2)
	rightDB = make([]DocumentData, 0, len(db)/2)
	threshold = findThreshold(featureValues)

	for _, doc := range db {
		value := float64((*doc.Bow)[feature])
		if value <= threshold {
			leftDB = append(leftDB, doc)
		} else {
			rightDB = append(rightDB, doc)
		}
	}

	return leftDB, rightDB, threshold
}

func findThreshold(population []int64) float64 {
	slices.Sort(population)
	middle := len(population) / 2

	if len(population)%2 == 0 {
		first := population[middle-1]
		second := population[middle]

		return float64((first + second)) / 2.0
	}

	return float64(population[middle])
}

func calculateClassDistribution(db []DocumentData) ClassDistribution {
	classes := getAllClassesFromDB(db)
	distributions := ClassDistribution{}

	for _, class := range classes {
		counter := 0

		for _, doc := range db {
			if doc.Class == class {
				counter += 1
			}
		}

		distributions[class] = float64(counter) / float64(len(db))
	}

	return distributions
}

func calculateMajorityClass(classDistribution ClassDistribution) string {
	maxDist := -1.0
	var majority string

	for class := range classDistribution {
		if classDistribution[class] > maxDist {
			majority = class
			maxDist = classDistribution[class]
		}
	}

	return majority
}
