package knn

import (
	"math"
	"ml/common"
	"os"
	"path"
	"strings"
)

type Point struct {
	Document *common.BoW
	Class    string
}

type KNNModel struct {
	Points     []NormalizedDocument
	Vocabulary *common.Vocabulary
	Size       int
	Df         *DocumentFrequency
	Tree       *BallTree
}

type KNNOptions struct {
	Folders  []string
	Classes  []string
	MinDf    int
	MaxDf    int
	LeafSize int
}

type Neighbor struct {
	Distance     float64
	Index        int
	Class        string
	DocumentName string
}

type DocumentData struct {
	DocumentName string
	Bow          *common.BoW
	Class        string
}

type NormalizedDocument struct {
	DocumentName string
	WBow         *common.WeightedBoW
	Class        string
}

type DocumentFrequency = map[string]int

func Train(options KNNOptions) KNNModel {
	v := common.Vocabulary{}
	db, df := normalize(options)
	defineVocabulary(df, &v)

	tree := BuildBallTree(db, &v, options.LeafSize)

	return KNNModel{
		Points:     db,
		Vocabulary: &v,
		Size:       len(db),
		Df:         &df,
		Tree:       tree,
	}
}

func Fit(model *KNNModel, point *common.BoW, k int) string {
	if k <= 0 {
		panic("k parameter must be greater than 0")
	}

	Q := CreatePriorityQueue(k)
	target := getWeithedBoW(point, model.Df, model.Size)
	fitRecursive(&Q, model.Tree.Root, &target, k, model.Vocabulary)

	return vote(Q.Items)
}

func fitRecursive(Q *PriorityQueue, node *Ball, target *common.WeightedBoW, k int, v *common.Vocabulary) {
	first := Q.Min()
	distanceToMin := cosineSimilarity(target, first.WBow, v)

	// Resolve for first.WBoW what should I set

	if cosineSimilarity(target, &node.Center, v)-node.Radius >= distanceToMin {
		// Q was not modified
		return
	}

	if IsLeaf(node) {
		// Check each data point
		for _, p := range node.Items {
			distanceToP := cosineSimilarity(target, p.WBow, v)
			if distanceToP < distanceToMin {
				Q.Insert(p, distanceToP)

				if Q.Size() > k {
					Q.Delete()
				}
			}
		}

	} else {
		childOne := Ball{}
		childTwo := Ball{}
		distanceToLeft := cosineSimilarity(target, &node.Left.Center, v)
		distanceToRight := cosineSimilarity(target, &node.Right.Center, v)

		if distanceToLeft < distanceToRight {
			childOne = *node.Left
			childTwo = *node.Right
		} else {
			childOne = *node.Right
			childTwo = *node.Left
		}

		fitRecursive(Q, &childOne, target, k, v)
		fitRecursive(Q, &childTwo, target, k, v)
	}
}

func defineVocabulary(df DocumentFrequency, v *common.Vocabulary) {
	for term, df := range df {
		(*v)[term] = int64(df)
	}
}

func normalize(options KNNOptions) ([]NormalizedDocument, DocumentFrequency) {
	size, err := common.GetCollectionSize(options.Folders, options.Classes)
	df := DocumentFrequency{}
	documentsData := make([]DocumentData, 0, size)
	normalizedDocuments := make([]NormalizedDocument, 0, size)

	if err != nil {
		panic(err)
	}

	for _, folder := range options.Folders {
		for _, class := range options.Classes {
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

				addToDocFrequency(&df, &bow)
				documentsData = append(documentsData, DocumentData{DocumentName: document.Name(), Bow: &bow, Class: class})
			}
		}
	}

	for term, docF := range df {
		if docF < options.MinDf || docF > options.MaxDf {
			delete(df, term)
		}
	}

	for _, info := range documentsData {
		weightedBoW := getWeithedBoW(info.Bow, &df, size)
		nDoc := NormalizedDocument{
			DocumentName: info.DocumentName,
			Class:        info.Class,
			WBow:         &weightedBoW,
		}

		normalizedDocuments = append(normalizedDocuments, nDoc)
	}

	return normalizedDocuments, df
}

func calculateTermFrequency(bow *common.BoW) common.WeightedBoW {
	var totalFrequencies int64
	tfBoW := common.WeightedBoW{}

	for token := range *bow {
		totalFrequencies += (*bow)[token]
	}

	for token := range *bow {
		tfBoW[token] = float64((*bow)[token]) / float64(totalFrequencies)
	}

	return tfBoW
}

func getWeithedBoW(bow *common.BoW, df *DocumentFrequency, size int) common.WeightedBoW {
	weightedBoW := common.WeightedBoW{}
	tfBoW := calculateTermFrequency(bow)

	for token, tf := range tfBoW {
		// In case a term is not present in a document, I add 1 on both of the term
		idf := math.Log10(float64(1+size) / float64(1+(*df)[token]))
		weightedBoW[token] = tf * idf
	}

	return weightedBoW
}

func addToDocFrequency(df *DocumentFrequency, bow *common.BoW) {
	for token := range *bow {
		numDocs, ok := (*df)[token]

		if ok {
			(*df)[token] = numDocs + 1
		} else {
			(*df)[token] = 1
		}
	}
}

func vote(neighbors []PriorityItem) string {
	classes := []string{"ham", "spam"}
	foundClass := classes[0]
	maxFrequency := -1

	for _, class := range classes {
		frequency := 0

		for _, n := range neighbors {
			if class == n.Value.Class {
				frequency += 1
			}
		}

		if frequency > maxFrequency {
			foundClass = class
			maxFrequency = frequency
		}
	}

	return foundClass
}

func cosineSimilarity(target *common.WeightedBoW, q *common.WeightedBoW, v *common.Vocabulary) float64 {
	targetLen := magnitude(target, v)
	pointLen := magnitude(q, v)
	product := dot(target, q, v)
	distance := product / (targetLen * pointLen)

	return distance
}

func dot(target *common.WeightedBoW, q *common.WeightedBoW, v *common.Vocabulary) float64 {
	var sum float64

	for token := range *v {
		targetV := (*target)[token]
		pointV := (*q)[token]
		sum = sum + (targetV * pointV)
	}

	return sum
}

func magnitude(vector *common.WeightedBoW, v *common.Vocabulary) float64 {
	var sum float64

	for token := range *v {
		val := (*vector)[token]
		sum += math.Pow(val, 2)
	}

	return math.Sqrt(sum)
}
